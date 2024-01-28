from argparse import Namespace
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from loader.data import TaskA_Dataset
from loader.uar import UAR
from models.joint_model import JointModel, JointModelPreprocessor
from util.checkpoints import ProgressTracker

from util.device import get_device


def add_args(parser):
    def p(name): return "--joint-model-" + name
    group = parser.add_argument_group("Word2Vec Classifier")
    group.add_argument(p("train"), action="store_true", default=False)
    group.add_argument(p("predict"), action="store_true", default=False)

    args = [
        ("cc-model-path", str, None),
        ("cc-tokenizer-path", str, None),
        ("w2v-model-path", str, None),
        ("w2v-tokenizer-path", str, None),
        ("cc-max-len", int, None),
        ("w2v-max-len", int, None),
        ("hidden-size", int, 128),
        ("dropout", float, 0.2),
        ("load-model", str, None),
        ("n-epochs", int, 10),
        ("batch-size", int, 32),
        ("lr", float, 0.005),
        ("save-every", int, 100),
        ("checkpoint-prefix", str, "joint-model-test"),
    ]

    for name, type, default_value in args:
        group.add_argument(p(name), type=type, default=default_value)


def evaluate(model, dev_loader, f1_only=True):
    model.eval()
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for cc, w2v, labels in dev_loader:
            cc = cc.to(get_device())
            w2v = w2v.to(get_device())
            out = model(cc, w2v)
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
    model: JointModel = None
    optimizer: torch.optim.Optimizer = None
    criterion: nn.Module = None
    train_loader: DataLoader = None
    dev_loader: DataLoader = None
    n_epochs: int = 10
    save_every: int = 100
    checkpoint_prefix: str = "joint-model-test"


def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate,
                         last_epoch_only=True, save_latest=False)
    losses = []
    for epoch in range(args.n_epochs):
        with tqdm.tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            args.model.train()
            for cc, w2v, labels in args.train_loader:
                args.optimizer.zero_grad()
                cc = cc.to(get_device())
                w2v = w2v.to(get_device())
                labels = labels.to(get_device())
                out = args.model(cc, w2v)
                out = out.squeeze()
                loss = args.criterion(out, labels)
                loss.backward()
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
        return getattr(args, "joint_model_" + name.replace("-", "_"))

    preprocessor = JointModelPreprocessor(
        cc_model_path=arg("cc_model_path"),
        cc_tokenizer_path=arg("cc_tokenizer_path"),
        cc_max_len=arg("cc-max-len"),
        w2v_model_path=arg("w2v_model_path"),
        w2v_tokenizer_path=arg("w2v_tokenizer_path"),
        w2v_max_len=arg("w2v-max-len"),
    )

    if arg("load-model"):
        model, checkpoint = JointModel.from_pretrained(arg("load-model"))
        optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model = JointModel(
            cc_size=preprocessor.cc_classifier.hidden_size,
            uar_size=512,
            w2v_size=preprocessor.w2v_classifier.hidden_size,
            hidden_size=arg("hidden_size"),
            dropout=arg("dropout")
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))

    model.to_device()
    print(model)

    if arg("train"):
        ds = TaskA_Dataset(split="train")
        ds_dev = TaskA_Dataset(split="dev")
        train_loader = DataLoader(
            ds,
            batch_size=arg("batch_size"),
            shuffle=True,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
        )
        dev_loader = DataLoader(
            ds_dev,
            batch_size=arg("batch_size"),
            shuffle=False,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
        )
        training_arguments = TrainingArguments(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            dev_loader=dev_loader,
            n_epochs=arg("n_epochs"),
            save_every=arg("save_every"),
            checkpoint_prefix=arg("checkpoint_prefix"),
            criterion=torch.nn.BCELoss()
        )
        train(training_arguments)
