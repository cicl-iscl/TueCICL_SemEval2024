from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from tqdm import tqdm

from loader.data import TaskA_Dataset, collate_fn

from util.core import abspath

from models.charLM import CharLM, CharLMTokenizer

from util.checkpoints import ProgressTracker

from util.device import get_device


def add_args(parser: ArgumentParser):
    group = parser.add_argument_group("CharLM")
    group.add_argument("--charlm-load-model", type=str, default=None)
    group.add_argument("--charlm-do-train", type=bool, default=False)
    group.add_argument("--charlm-emb-size", type=int, default=8)
    group.add_argument("--charlm-hidden-size", type=int, default=128)
    group.add_argument("--charlm-num-layers", type=int, default=1)
    group.add_argument("--charlm-aggregate-fn", type=str,
                       default="mean", choices=["mean", "max", "last"])
    group.add_argument("--charlm-lr", type=float, default=0.001)
    group.add_argument("--charlm-clip", type=float, default=None)
    group.add_argument("--charlm-start-epoch", type=int, default=1)
    group.add_argument("--charlm-n-epochs", type=int, default=5)
    group.add_argument("--charlm-save-every", type=int, default=100)
    group.add_argument("--charlm-checkpoint-prefix", type=str,
                       default="charlm")
    group.add_argument("--charlm-batch-size", type=int, default=8)
    group.add_argument("--charlm-tokenizer-type", type=str, default="uncondensed", choices=[
        "condensed", "uncondensed"])
    group.add_argument("--charlm-tokenizer-max-len", type=int,
                       default=15_000, help="Applied to dev set only")
    group.add_argument("--charlm-dropout", type=float, default=0.0)
    group.add_argument("--charlm-window-size", type=int, default=5000)


def evaluate(model, dev_dataloader, f1_only=True):
    model.eval()
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for input_ids, attentions, labels in dev_dataloader:
            _, out, _ = model(input_ids, attentions)
            pred = out.argmax(dim=-1)
            for i in range(pred.shape[0]):
                y_pred.append(pred[i].item())
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_gold, y_pred, average="macro", zero_division=0.0)

    if f1_only:
        return f1

    return f1, r


@dataclass
class CharLMTrainingArguments:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    n_epochs: int = 5
    save_every: int = 100
    device: str = get_device()
    clip: float = None
    checkpoint_prefix: str = "charLM"
    start_epoch: int = 1
    window_size: int = 5000

def _get_windows(input_ids, attentions, window_size=5000):
    l = input_ids.shape[1]
    p = 0
    while p < l:
        yield input_ids[:, p:p + window_size], attentions[:, p:p + window_size]
        p += window_size


def _process_windows(args: CharLMTrainingArguments, windows, labels, classification_criterion, lm_criterion):
    lstm_hidden = None
    for (input_ids, attentions) in windows:

        # ------------------
        # Filter no attention
        # ------------------

        # ------------------
        # Run model
        # ------------------

        args.optimizer.zero_grad()

        lm_out, classifier_out, lstm_hidden = args.model(
            input_ids, attentions, lstm_hidden)
        lstm_hidden = tuple([h.detach() for h in lstm_hidden])

        loss = torch.tensor(0, dtype=torch.float32, device=get_device())

        # ------------------
        # LM loss
        # ------------------
        for i in range(input_ids.shape[0]):
            if labels[i].item() == 0:
                # only train LM on human texts
                until = attentions[i].cpu().tolist().index(
                    0) if 0 in attentions[i] else input_ids.shape[1]
                y_pred = lm_out[i, :until]
                y_gold = input_ids[i, :until]
                y_pred = y_pred[:-1]
                y_gold = y_gold[1:]
                if y_pred.shape[0] == 0:
                    continue
                loss_update = lm_criterion(y_pred, y_gold)
                loss += loss_update

                if torch.isnan(loss):
                    print("NAN loss")

        # ------------------
        # Classifier loss
        # ------------------

        with_attention = []
        for i in range(attentions.shape[0]):
            if 1 in attentions[i]:
                with_attention.append(i)

        if len(with_attention) != 0:
            loss_update = classification_criterion(
                classifier_out[with_attention], labels[with_attention])
            loss += loss_update

        # ------------------
        # Backprop
        # ------------------

        loss.backward()
        args.optimizer.step()

        # with torch.no_grad():
        #     # prepare for next iteration
        #     _, _, lstm_hidden = args.model(input_ids, attentions, lstm_hidden)
        #     lstm_hidden = tuple([h.detach() for h in lstm_hidden])


def train_charlm(args: CharLMTrainingArguments):
    classification_criterion = nn.NLLLoss()
    lm_criterion = nn.NLLLoss()

    counter = 0
    tracker = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        args.model.train()
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for input_ids, attentions, labels in args.train_loader:
                windows = _get_windows(input_ids, attentions, window_size=args.window_size)
                _process_windows(args, windows, labels, classification_criterion=classification_criterion, lm_criterion=lm_criterion)
                
                counter += 1
                pbar.update(1)

                if counter % args.save_every == 0 and counter > 0:
                    torch.cuda.empty_cache()
                    best, latest = tracker.for_steps(
                        args.model, args.dev_loader)
                    args.model.train()
                    pbar.set_postfix({"best": best, "latest": latest})
        
        torch.cuda.empty_cache()
        tracker.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
        args.model.train()


def entry(args):
    tokenizer_path = abspath(
        __file__, f"../../data/charlm_vocab_{args.charlm_tokenizer_type}.pkl")
    tokenizer = CharLMTokenizer.from_pretrained(tokenizer_path)

    start_epoch = args.charlm_start_epoch

    if args.charlm_load_model is not None:
        model_path = abspath(
            __file__, f"../../checkpoints/{args.charlm_load_model}")
        model = CharLM.from_pretrained(model_path)
        save_data = torch.load(model_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.charlm_lr)
        if args.charlm_do_train and "optimizer" in save_data:
            print("Loading optimizer state from checkpoint")
            optimizer.load_state_dict(save_data["optimizer"])
    else:
        model = CharLM(
            vocab_size=len(tokenizer.vocab),
            aggregate_fn=args.charlm_aggregate_fn,
            emb_size=args.charlm_emb_size,
            hidden_size=args.charlm_hidden_size,
            num_layers=args.charlm_num_layers,
            dropout=args.charlm_dropout
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.charlm_lr)
    model.to(get_device())
    print(model)
    model = nn.DataParallel(model)

    dev_dataloader = DataLoader(
        TaskA_Dataset(split="dev"),
        batch_size=args.charlm_batch_size,
        shuffle=True,
        collate_fn=collate_fn(
            tokenizer, max_len=args.charlm_tokenizer_max_len),
    )

    if args.charlm_do_train:
        train_dataloader = DataLoader(
            TaskA_Dataset(split="train"),
            batch_size=args.charlm_batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer, max_len=None),
            drop_last=True
        )

        training_args = CharLMTrainingArguments(
            checkpoint_prefix=args.charlm_checkpoint_prefix,
            train_loader=train_dataloader,
            dev_loader=dev_dataloader,
            model=model,
            optimizer=optimizer,
            device=get_device(),
            n_epochs=args.charlm_n_epochs,
            start_epoch=start_epoch,
            save_every=args.charlm_save_every,
            window_size=args.charlm_window_size
        )

        train_charlm(training_args)
