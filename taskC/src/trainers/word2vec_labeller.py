from dataclasses import dataclass
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from ..loader.data import TaskC_Data, collate_fn_wordlevel

from ..util.checkpoints import ProgressTracker

from ..util.device import get_device
from ..models.word2vec_labeller import Word2Vec_Labeller, Word2Vec_Tokenizer


@dataclass
class Word2VecLabellerArgs:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    n_epochs: int = 5
    save_every: int = 50
    device: str = get_device()
    clip: float = 1.0
    checkpoint_prefix = "word2vec_labeller"
    tokenizer: Word2Vec_Tokenizer = None


def train_word2vec_labeller(args: Word2VecLabellerArgs):
    tracker = ProgressTracker(args.checkpoint_prefix)
    for epoch in range(1, args.n_epochs+1):
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for i, (input_ids, labels) in enumerate(args.train_loader):
                args.optimizer.zero_grad()
                out = args.model(input_ids)
                out = out.reshape(-1, 2)
                labels = labels.reshape(-1)
                loss = args.criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    args.model.parameters(), args.clip)
                args.optimizer.step()

                pbar.update(1)

                if i % args.save_every == 0 and i != 0:
                    best, latest = tracker.for_steps(
                        args.model, args.dev_loader)
                    pbar.set_postfix(
                        {"loss": loss.item(), "best": best, "latest": latest})
            tracker.for_epoch(args.model, args.optimizer,
                              epoch, args.dev_loader)


def entry():
    tokenizer = Word2Vec_Tokenizer()
    model = Word2Vec_Labeller(
        hidden_size=128, num_layers=1, pretrained_embeddings=tokenizer.embeddings)
    model.to(get_device())
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.NLLLoss()
    train_loader = DataLoader(TaskC_Data(
        split="train"), batch_size=2, collate_fn=collate_fn_wordlevel(tokenizer))
    dev_loader = DataLoader(TaskC_Data(
        split="dev"), batch_size=2, collate_fn=collate_fn_wordlevel(tokenizer))

    args = Word2VecLabellerArgs(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        dev_loader=dev_loader,
        tokenizer=tokenizer,
        save_every=300
    )

    train_word2vec_labeller(args)
