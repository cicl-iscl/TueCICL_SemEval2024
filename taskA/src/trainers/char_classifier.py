from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
from tqdm import tqdm
import torch.nn as nn
from loader.data import TaskA_Dataset, collate_fn

from models.char_classifier import CharClassifier, CharClassifierTokenizer
from util.checkpoints import ProgressTracker
from util.core import abspath
from util.device import get_device
from torch.utils.data import DataLoader
import pandas as pd


def add_args(parser: ArgumentParser):
    group = parser.add_argument_group("CharClassifier")
    group.add_argument("--char-class-do-train", action="store_true", default=False)
    group.add_argument("--char-class-load-model", type=str, default=None)
    group.add_argument("--char-class-emb-size", type=int, default=8)
    group.add_argument("--char-class-hidden-size", type=int, default=128)
    group.add_argument("--char-class-num-layers", type=int, default=1)
    group.add_argument("--char-class-lr", type=float, default=0.001)
    group.add_argument("--char-class-clip", type=float, default=None)
    group.add_argument("--char-class-start-epoch", type=int, default=1,
                       help="For naming checkpoints")
    group.add_argument("--char-class-n-epochs", type=int, default=5)
    group.add_argument("--char-class-save-every", type=int, default=100)
    group.add_argument("--char-class-checkpoint-prefix", type=str,
                       default="char-class")
    group.add_argument("--char-class-batch-size", type=int, default=8)
    group.add_argument("--char-class-resume-path", type=str, default=None)
    group.add_argument("--char-class-tokenizer-type", type=str,
                       default="uncondensed", choices=["condensed", "uncondensed"])
    group.add_argument("--char-class-tokenizer-max-len",
                       type=int, default=15_000)
    group.add_argument("--char-class-dropout", type=float, default=0.0)
    group.add_argument("--char-class-predict", type=str, default=None)


def evaluate(model, dev_dataloader, f1_only=True):
    y_pred = []
    y_gold = []
    model.eval()
    with torch.no_grad():
        for input_ids, _, labels in dev_dataloader:
            out, _ = model(input_ids)
            pred = out.argmax(dim=1)
            for i in range(pred.shape[0]):
                y_pred.append(pred[i].item())
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_gold, y_pred, average="macro", zero_division=0.0)

    if f1_only:
        return f1

    return f1, r

def predict(model, test_loader, out_file):
    model.eval()
    predictions = []
    with torch.no_grad():
        for input_ids, _, text_ids in tqdm(test_loader, desc="Predicting"):
            out, _ = model(input_ids)
            pred = out.argmax(dim=1)
            for i in range(pred.shape[0]):
                predictions.append({"id": text_ids[i], "label": pred[i].item()})
                
                
    pd.DataFrame(predictions).to_json(out_file, orient="records", lines=True)
                


@dataclass
class CharClassifierTrainingArguments:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    n_epochs: int = 5
    save_every: int = 100
    device: str = get_device()
    clip: float = None
    checkpoint_prefix: str = "charLM"
    start_epoch: int = 1
    window_size: int = 5000


def train_char_classifier(args: CharClassifierTrainingArguments):
    args.model.train()
    i = 0

    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for input_ids, attentions, labels in args.train_loader:
                args.optimizer.zero_grad()
                out, _ = args.model(input_ids)
                loss = args.criterion(out, labels)
                loss.backward()
                args.optimizer.step()
                pbar.update(1)
                i += 1

                if i % args.save_every == 0 and i > 0:
                    best, latest = pt.for_steps(args.model, args.dev_loader)
                    args.model.train()
                    pbar.set_postfix(best=best, latest=latest)

        pt.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
        args.model.train()


def entry(args):
    tokenizer_path = abspath(
        __file__, f"../../data/vocab/charlm_vocab_{args.char_class_tokenizer_type}.pkl")

    tokenizer = CharClassifierTokenizer.from_pretrained(tokenizer_path)

    if args.char_class_load_model is not None:
        model = CharClassifier.from_pretrained(args.char_class_load_model)
        print(model)
    else:
        model = CharClassifier(
            vocab_size=len(tokenizer.vocab),
            emb_size=args.char_class_emb_size,
            hidden_size=args.char_class_hidden_size,
            num_layers=args.char_class_num_layers,
            dropout=args.char_class_dropout
        )

    model.to(get_device())
    model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.char_class_lr)
    criterion = torch.nn.NLLLoss()
    start_epoch = args.char_class_start_epoch
    
    if args.char_class_do_train and args.char_class_load_model:
        save_data = torch.load(args.char_class_load_model)
        if "optimizer" in save_data:
            print("Loading optimizer state from checkpoint")
            optimizer.load_state_dict(save_data["optimizer"])

    dev_dataloader = DataLoader(
        TaskA_Dataset(split="dev"),
        batch_size=args.char_class_batch_size,
        shuffle=True,
        collate_fn=collate_fn(
            tokenizer, max_len=args.char_class_tokenizer_max_len),
        drop_last=True
    )

    if args.char_class_do_train:
        train_dataloader = DataLoader(
            TaskA_Dataset(split="train"),
            batch_size=args.char_class_batch_size,
            shuffle=True,
            collate_fn=collate_fn(
                tokenizer, max_len=args.char_class_tokenizer_max_len),
            drop_last=True
        )

        training_args = CharClassifierTrainingArguments(
            checkpoint_prefix=args.char_class_checkpoint_prefix,
            train_loader=train_dataloader,
            dev_loader=dev_dataloader,
            model=model,
            optimizer=optimizer,
            device=get_device(),
            n_epochs=args.char_class_n_epochs,
            start_epoch=start_epoch,
            save_every=args.char_class_save_every,
            criterion=criterion
        )

        train_char_classifier(training_args)
    
    if args.char_class_predict is not None:
        test_dataloader = DataLoader(
            TaskA_Dataset(split="test"),
            batch_size=args.char_class_batch_size,
            shuffle=False,
            collate_fn=collate_fn(
                tokenizer, max_len=args.char_class_tokenizer_max_len, is_test=True),
            drop_last=False
        )
        
        predict(model, test_dataloader, args.char_class_predict)
