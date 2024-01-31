from argparse import Namespace
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from loader.data import TaskA_Dataset
from loader.spacy import SpacyFeatures
from loader.uar import UAR
from models.joint_model import JointModel, JointModelPreprocessor
from util.checkpoints import ProgressTracker

from util.device import get_device


def add_args(parser):
    def p(name): return "--joint-model-" + name
    group = parser.add_argument_group("Word2Vec Classifier")
    group.add_argument(p("train"), action="store_true", default=False)
    group.add_argument(p("predict"), type=str, default=None)

    args = [
        ("cc-model-path", str, None),
        ("cc-tokenizer-path", str, None),
        ("w2v-model-path", str, None),
        ("w2v-tokenizer-path", str, None),
        ("cc-max-len", int, None),
        ("w2v-max-len", int, None),
        ("spacy-size", int, 60),
        ("spacy-hidden-size", int, 256),
        ("spacy-del-feats", str, None),
        ("spacy-train-feats", str, None),
        ("spacy-test-feats", str, None),
        ("spacy-dev-feats", str, None),
        ("ppl-train", str, None),
        ("ppl-dev", str, None),
        ("ppl-test", str, None),
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

def predict(model: JointModel, test_loader, out_file):
    model.eval()
    predictions = []
    with torch.no_grad():
        for cc, w2v, spacy, text_ids in tqdm.tqdm(test_loader, desc="Predicting"):
            cc = cc.to(get_device())
            w2v = w2v.to(get_device())
            spacy = spacy.to(get_device())
            out = model(cc, w2v, spacy)
            pred = torch.round(out)
            for i in range(pred.shape[0]):
                p = int(pred[i].item())
                predictions.append({"id": text_ids[i], "label": p})
    pd.DataFrame(predictions).to_json(out_file, orient="records", lines=True)

def evaluate(model, dev_loader, f1_only=True):
    model.eval()
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for cc, w2v, spacy, labels in dev_loader:
            cc = cc.to(get_device())
            w2v = w2v.to(get_device())
            spacy = spacy.to(get_device())
            out = model(cc, w2v, spacy)
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
            for cc, w2v, spacy, labels in args.train_loader:
                args.optimizer.zero_grad()
                cc = cc.to(get_device())
                w2v = w2v.to(get_device())
                spacy = spacy.to(get_device())
                labels = labels.to(get_device())
                out = args.model(cc, w2v, spacy)
                # print(out)
                # print(labels)
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
    
    spacy = SpacyFeatures(
        train_path=arg("spacy-train-feats"),
        ppl_path_train=arg("ppl-train"),
        dev_path=arg("spacy-dev-feats"),
        ppl_path_dev=arg("ppl-dev"),
        test_path=arg("spacy-test-feats"),
        ppl_path_test=arg("ppl-test"),
        del_feats=arg("spacy-del-feats"),
    )
    spacy.scale()

    preprocessor = JointModelPreprocessor(
        cc_model_path=arg("cc_model_path"),
        cc_tokenizer_path=arg("cc_tokenizer_path"),
        cc_max_len=arg("cc-max-len"),
        w2v_model_path=arg("w2v_model_path"),
        w2v_tokenizer_path=arg("w2v_tokenizer_path"),
        w2v_max_len=arg("w2v-max-len"),
        spacy=spacy
    )

    if arg("load-model"):
        model, checkpoint = JointModel.from_pretrained(arg("load-model"))
        optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model = JointModel(
            cc_size=preprocessor.cc_classifier.hidden_size,
            w2v_size=preprocessor.w2v_classifier.hidden_size,
            spacy_size=arg("spacy_size"),
            spacy_hidden_size=arg("spacy_hidden_size"),
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
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor, spacy_split="train")
        )
        dev_loader = DataLoader(
            ds_dev,
            batch_size=arg("batch_size"),
            shuffle=False,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor, spacy_split="dev")
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
    
    if arg("predict"):
        dev_loader = DataLoader(
            TaskA_Dataset(split="dev"),
            batch_size=arg("batch_size"),
            shuffle=False,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
        )
        test_loader = DataLoader(
            TaskA_Dataset(split="test"),
            batch_size=arg("batch_size"),
            shuffle=False,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor, is_test=True, spacy_split="test")
        )
        _, report = evaluate(model, dev_loader, f1_only=False)
        print(report)
        predict(model, test_loader, arg("predict"))
