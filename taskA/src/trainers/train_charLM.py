from dataclasses import dataclass
from pandas import isna
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from tqdm import tqdm

from loader.data import TaskA_Dataset, collate_fn

from util.core import abspath

from models.charLM import CharLM, CharLMTokenizer

from util.checkpoints import ProgressTracker

from util.device import get_device


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

        lm_out, classifier_out, _ = args.model(
            input_ids, attentions, lstm_hidden)

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

        with torch.no_grad():
            # prepare for next iteration
            _, _, lstm_hidden = args.model(input_ids, attentions, lstm_hidden)
            lstm_hidden = tuple([h.detach() for h in lstm_hidden])


def train_charlm(args: CharLMTrainingArguments):
    classification_criterion = nn.NLLLoss()
    lm_criterion = nn.NLLLoss()

    i = 0
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        args.model.train()
        tracker = ProgressTracker(args.checkpoint_prefix)
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for input_ids, attentions, labels in args.train_loader:
                windows = _get_windows(
                    input_ids, attentions, window_size=args.window_size)
                _process_windows(args, windows, labels,
                                 classification_criterion, lm_criterion)
                i += 1
                pbar.update(1)

                if i % args.save_every == 0 and i > 0:
                    best, latest = tracker.for_steps(
                        args.model, args.dev_loader)
                    pbar.set_postfix({"best": best, "latest": latest})
        tracker.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)


def entry(args):
    tokenizer = CharLMTokenizer.from_pretrained(
        abspath(__file__, "../../data/charlm_vocab_uncondensed.pkl"))
    model = CharLM(
        vocab_size=len(tokenizer.vocab),
        aggregate_fn="mean",
        emb_size=8,
        hidden_size=128,
        num_layers=1
    )
    model.to(get_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_dataloader = DataLoader(
        TaskA_Dataset(split="train"),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn(tokenizer)
    )

    dev_dataloader = DataLoader(
        TaskA_Dataset(split="dev"),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, max_len=15_000)
    )

    training_args = CharLMTrainingArguments(
        checkpoint_prefix="test",
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        model=model,
        optimizer=optimizer,
        device=get_device(),
        n_epochs=5,
        start_epoch=1,
        save_every=2000
    )

    train_charlm(training_args)
