from argparse import Namespace
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
import tqdm
from loader.data import TaskA_Dataset
from loader.spacy import SpacyFeatures
from models.lang_feats_mlp import SpacyFeaturesMLP, collate_fn
from util.checkpoints import ProgressTracker

from util.device import get_device
from torch.utils.data import DataLoader


def add_args(parser):
    def p(name): return "--lang-mlp-" + name
    group = parser.add_argument_group("Language Features MLP")
    group.add_argument(p("hidden-size"), type=int, default=128)
    group.add_argument(p("dropout"), type=float, default=0.0)
    group.add_argument(p("spacy-n-feats"), type=int, default=66)
    group.add_argument(p("spacy-train-feats"), type=str, default=0.0)
    group.add_argument(p("spacy-dev-feats"), type=str, default=0.0)
    group.add_argument(p("spacy-scale"), action="store_true", default=False)
    group.add_argument(p("n-epochs"), type=int, default=10)
    group.add_argument(p("save-every"), type=int, default=1)
    group.add_argument(p("train"), default=True, action="store_true")
    group.add_argument(p("checkpoint-prefix"),
                       type=str, default="lang-mlp")
    group.add_argument(p("load-model"), type=str, default=None)
    group.add_argument(p("batch-size"), type=int, default=32)
    group.add_argument(p("lr"), type=float, default=0.005)


def evaluate(model: SpacyFeaturesMLP, dev_loader, f1_only=True):
    model.eval()
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for spacy_feats, labels, _ in dev_loader:
            spacy_feats = spacy_feats.to(get_device())
            out = model(spacy_feats)
            pred = out.round()
            for i in range(out.shape[0]):
                y_pred.append(pred[i].item())
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_gold, y_pred, average="macro", zero_division=0.0)

    if f1_only:
        return f1

    return f1, r


@dataclass
class TrainingArguments:
    model: SpacyFeaturesMLP
    optimizer: torch.optim.Optimizer
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    n_epochs: int = 5
    save_every: int = 100
    checkpoint_prefix: str = "word2vec-classifier"
    criterion: torch.nn.Module = None


def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(
        args.checkpoint_prefix,
        evaluate_fn=evaluate,
        last_epoch_only=True,
        save_latest=False
    )
    losses = []
    for epoch in range(args.n_epochs):
        with tqdm.tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            args.model.train()
            for spacy_feats, labels, _ in args.train_loader:
                args.optimizer.zero_grad()
                spacy_feats = spacy_feats.to(get_device())
                labels = labels.to(get_device())
                out = args.model(spacy_feats)
                print(out)
                out = out.reshape(-1)
                loss = args.criterion(out, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(args.model.parameters(), 1.0)
                args.optimizer.step()
                losses.append(loss.item())
                pbar.update(1)
                i += 1

                if i % args.save_every == 0 and i > 0:
                    best, latest = pt.for_steps(args.model, args.dev_loader)
                    args.model.train()
                    l = sum(losses) / len(losses)
                    losses = []
                    pbar.set_postfix(
                        best=best, latest=latest, loss=l)

        pt.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
        args.model.train()


def entry(args: Namespace):
    def arg(name):
        return getattr(args, "lang_mlp_" + name.replace("-", "_"))
    spacy_features = SpacyFeatures(
        arg("spacy_train_feats"), arg("spacy_dev_feats"))
    if arg("spacy_scale"):
        spacy_features.scale()
    ds_train = TaskA_Dataset(
        split="train", spacy_features=spacy_features)
    ds_dev = TaskA_Dataset(
        split="dev", spacy_features=spacy_features)
    dl_train = DataLoader(ds_train, batch_size=arg("batch_size"),
                          shuffle=True, collate_fn=collate_fn)
    dl_dev = DataLoader(ds_dev, batch_size=arg("batch_size"),
                        shuffle=False, collate_fn=collate_fn)
    if arg("load-model"):
        model, checkpoint = SpacyFeaturesMLP.from_pretrained(arg("load-model"))
        print("Model loaded from", arg("load-model"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state dict")
    else:
        model = SpacyFeaturesMLP(
            arg("spacy_n_feats"), arg("hidden_size"), arg("dropout"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))
        model.to_device()
    print(model)

    training_args = TrainingArguments(
        model=model,
        optimizer=optimizer,
        train_loader=dl_train,
        dev_loader=dl_dev,
        n_epochs=arg("n_epochs"),
        save_every=arg("save_every"),
        checkpoint_prefix=arg("checkpoint_prefix"),
        criterion=torch.nn.BCELoss()
    )

    if arg("train"):
        train(training_args)
