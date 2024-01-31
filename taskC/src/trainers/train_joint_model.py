from argparse import Namespace
from dataclasses import dataclass
import pandas as pd

import torch
from tqdm import tqdm

from loader.data import TaskC_Data
from models.joint_model import JointModel, JointModelPreprocessor
from torch.utils.data import DataLoader
from util.checkpoints import ProgressTracker

from util.device import get_device


def add_args(parser):
    group = parser.add_argument_group("joint-model")
    def p(cmd): return f"--joint-model-{cmd}"
    args = [
        ("char-tokenizer", str, None),
        ("char-model", str, None),
        ("char-max-len", int, 10000),
        ("w2v-tokenizer", str, None),
        ("w2v-model", str, None),
        ("w2v-max-len", int, 10000),

        ("load-model", str, None),
        ("save-every-extended", int, 1000),
        ("save-every-pure", int, 1000),
        ("checkpoint-prefix", str, "joint_model"),
        ("epochs-extended", int, 10),
        ("epochs-pure", int, 10),
        ("lr", float, 1e-3),
        ("batch-size", int, 8),
        ("hidden-size", int, 2),
        ("dropout", float, 0.2),
        ("predict", str, None),
    ]

    for arg in args:
        group.add_argument(p(arg[0]), type=arg[1], default=arg[2])

    group.add_argument(p("train"), action="store_true", default=False)

def predict(model, test_loader, out_file):
    model.eval()
    predictions = []
    
    def _true_label(labels):
        l = labels.cpu().tolist()
        _i = None
        if 1 in l:
            _i = l.index(1)
        else:
            _i = len(l) - 1
        return _i
    
    for x_char, x_w2v, _, ids in tqdm(test_loader, desc="Predicting"):
        with torch.no_grad():
            out = model(x_char, x_w2v)
            preds = torch.argmax(out, dim=-1)
            for i in range(len(preds)):
                p = _true_label(preds[i])
                predictions.append({"id": ids[i], "label": p})
    
    pd.DataFrame(predictions).to_json(out_file, orient="records", lines=True)

def evaluate(_model, dev_loader):
    _model.eval()
    if hasattr(_model, "module"):
        model = _model.module
    else:
        model = _model
    model.eval()
    distances = []

    def _true_label(labels):
        l = labels.cpu().tolist()
        _i = None
        if 1 in l:
            _i = l.index(1)
        else:
            _i = len(l) - 1
        return _i

    for x_char, x_w2v, labels, _ in dev_loader:
        with torch.no_grad():
            out = model(x_char, x_w2v)
            preds = torch.argmax(out, dim=-1)
            for i in range(len(preds)):
                true_label = _true_label(labels[i])
                true_prediction = _true_label(preds[i])
                distances.append(abs(true_label - true_prediction))

    return sum(distances) / len(distances)


@dataclass
class TrainingArguments:
    model: JointModel = None
    epochs_extended: int = None
    epochs_pure: int = None
    optimizer: torch.optim.Optimizer = None
    save_every_extended: int = None
    save_every_pure: int = None
    train_loader_pure: DataLoader = None
    train_loader_extended: DataLoader = None
    dev_loader: DataLoader = None
    checkpoint_prefix: str = "word2vec_labeller"
    criterion: torch.nn.Module = None

def perform_training_step(args: TrainingArguments, batch):
    x_char, x_w2v, labels, attentions = batch
    labels = labels.to(get_device())
    args.optimizer.zero_grad()

    loss = torch.tensor(0.0, device=get_device())
    out = args.model(x_char, x_w2v)

    for j in range(len(x_char)):
        a = attentions[j].cpu().tolist()
        attention_bound = a.index(0) if 0 in a else len(a)
        local_out = out[j][:attention_bound]
        local_labels = labels[j][:attention_bound]
        loss += args.criterion(local_out, local_labels)

    loss.backward()
    args.optimizer.step()
    return loss.item()

def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate, last_epoch_only=True, save_latest=False)
    args.model.train()
    losses = []
    for epoch in range(args.epochs_extended):
        with tqdm(total=len(args.train_loader_extended)) as pbar:
            pbar.set_description(f"Extended epoch {epoch}")
            for batch in args.train_loader_extended:
                loss = perform_training_step(args, batch)
                i += 1
                pbar.update(1)
                losses.append(loss)
                if i % args.save_every_extended == 0 and i != 0:
                    l = sum(losses) / len(losses)
                    losses = []
                    best, latest = pt.for_steps(args.model, args.dev_loader)
                    pbar.set_postfix(best=best, latest=latest, loss=l)
                    args.model.train()
        pt.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
        args.model.train()

    for epoch in range(args.epochs_pure):
        with tqdm(total=len(args.train_loader_pure)) as pbar:
            pbar.set_description(f"Pure epoch {epoch}")
            for batch in args.train_loader_pure:
                loss = perform_training_step(args, batch)
                i += 1
                pbar.update(1)
                losses.append(loss)
                if i % args.save_every_pure == 0 and i != 0:
                    l = sum(losses) / len(losses)
                    losses = []
                    best, latest = pt.for_steps(
                        args.model, args.dev_loader)
                    pbar.set_postfix(best=best, latest=latest, loss=l)
                    args.model.train()
        _e = epoch + args.epochs_extended
        pt.for_epoch(args.model, args.optimizer, _e, args.dev_loader)
        args.model.train()

def entry(args: Namespace):
    def arg(name):
        return getattr(args, "joint_model_" + name.replace("-", "_"))

    preprocessor = JointModelPreprocessor(
        char_model=arg("char-model"),
        char_tokenizer=arg("char-tokenizer"),
        char_max_len=arg("char-max-len"),
        word2vec_model=arg("w2v-model"),
        word2vec_tokenizer=arg("w2v-tokenizer"),
        word2vec_max_len=arg("w2v-max-len"),
    )

    train_ds_ext = TaskC_Data(split="train")
    train_ds_ext.import_task_A()
    train_ds_pure = TaskC_Data(split="train")
    dev_ds = TaskC_Data(split="dev")

    train_dl_ext = DataLoader(
        train_ds_ext,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )
    train_dl_pure = DataLoader(
        train_ds_pure,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=arg("batch-size"),
        shuffle=False,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )
    
    if arg("load-model"):
        model, checkpoint = JointModel.from_pretrained(arg("load-model"))
        print("Loaded model from", arg("load-model"))
        optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state dict")
    else:
        model = JointModel(
            char_size=preprocessor.char_model.hidden_size * 2,
            w2v_size=preprocessor.word2vec_model.hidden_size * 2,
            hidden_size=arg("hidden-size"),
            dropout=arg("dropout"),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))
    model.to_device()
    print(model)

    training_arguments = TrainingArguments(
        model=model,
        epochs_extended=arg("epochs-extended"),
        epochs_pure=arg("epochs-pure"),
        optimizer=optimizer,
        save_every_extended=arg("save-every-extended"),
        save_every_pure=arg("save-every-pure"),
        train_loader_pure=train_dl_pure,
        train_loader_extended=train_dl_ext,
        dev_loader=dev_dl,
        checkpoint_prefix=arg("checkpoint-prefix"),
        criterion=torch.nn.NLLLoss(),
    )
    
    if arg("train"):
        train(training_arguments)
    
    if arg("predict"):
        MAD = evaluate(model, dev_dl)
        print("MAD:", MAD)
        test_ds = TaskC_Data(split="test")
        test_dl = DataLoader(
            test_ds,
            batch_size=arg("batch-size"),
            shuffle=False,
            collate_fn=JointModelPreprocessor.collate_fn(preprocessor, is_test=True)
        )
        predict(model, test_dl, arg("predict"))