from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

import torch
from tqdm import tqdm

from ..util.validation import evaluate

from ..util.checkpoints import ProgressTracker

from ..loader.data import TaskC_Data, collate_fn_charlevel, collate_fn_wordlevel

from ..util.device import get_device
from ..models.char_labeller import CharLevelLabeller, CharTokenizer


@dataclass
class CharLevelLabellerArgs:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    n_epochs: int = 5
    save_every: int = 100
    device: str = get_device()
    clip: float = 1.0
    checkpoint_prefix: str = "charlabeller"
    start_epoch: int = 1


def train_char_level_labeller(args: CharLevelLabellerArgs):
    tracker = ProgressTracker(args.checkpoint_prefix)
    i = 0
    for epoch in range(args.start_epoch, args.n_epochs+1):
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for batch in args.train_loader:
                args.optimizer.zero_grad()
                input_ids, _, _, labels, _ = batch
                out = args.model(input_ids)
                out = out.reshape(-1, 2)
                labels = labels.reshape(-1)
                loss = args.criterion(out, labels)
                loss.backward()
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        args.model.parameters(), args.clip)
                args.optimizer.step()

                pbar.update(1)
                i += 1

                if i % args.save_every == 0 and i != 0:
                    best, latest = tracker.for_steps(
                        args.model, args.dev_loader)
                    pbar.set_postfix(
                        {"loss": loss.item(), "best": best, "latest": latest})
            tracker.for_epoch(args.model, args.optimizer,
                              epoch, args.dev_loader)


def entry():
    tokenizer = CharTokenizer.from_pretrained(
        "data/charlm_vocab_uncondensed.pkl")
    model = CharLevelLabeller(
        embedding_size=8,
        hidden_size=256,
        num_layers=1,
        output_size=2,
        vocab_size=len(tokenizer.idx2word),
    )
    model.to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.NLLLoss()
    batch_size = 8
    ds = TaskC_Data(split="train")
    # enrich training data
    ds.import_task_A()
    train_loader = DataLoader(
        ds, shuffle=True, batch_size=batch_size, collate_fn=collate_fn_charlevel(tokenizer))
    dev_loader = DataLoader(TaskC_Data(
        split="dev"), batch_size=batch_size, collate_fn=collate_fn_charlevel(tokenizer))

    args = CharLevelLabellerArgs(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        dev_loader=dev_loader,
        clip=None,
        device=get_device(),
        n_epochs=10,
        save_every=2000,
        checkpoint_prefix="charlabeller",
        start_epoch=1
    )

    train_char_level_labeller(args)
