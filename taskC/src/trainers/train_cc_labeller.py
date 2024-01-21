from argparse import Namespace
from dataclasses import dataclass


import torch
from tqdm import tqdm
from loader.data import TaskC_Data
from models.cc_labeller import CCModel, CCTokenizer
from torch.utils.data import DataLoader

from util.checkpoints import ProgressTracker
from util.device import get_device


def add_args(parser):
    group = parser.add_argument_group("cc_labeller")
    def p(cmd): return f"--cc-labeller-{cmd}"
    group.add_argument(p("tokenizer-path"), type=str, default=None)
    group.add_argument(p("tokenizer-max-len"), type=int, default=10000)
    group.add_argument(p("batch-size"), type=int, default=8)
    group.add_argument(p("train"), action="store_true", default=False)
    group.add_argument(p("epochs"), type=int, default=10)
    group.add_argument(p("lr"), type=float, default=1e-3)
    group.add_argument(p("cc-path"), type=str, default=None)
    group.add_argument(p("save-every"), type=int, default=1000)
    group.add_argument(p("checkpoint-prefix"), type=str, default="cc_labeller")


def evaluate(model: CCModel, dev_loader):
    model.eval()
    distances = []

    def _true_label(labels, words):
        l = labels.cpu().tolist()
        _i = None
        if 1 in l:
            _i = l.index(1)
        else:
            _i = len(l) - 1
        return words[_i].item()

    for ids, labels, words, _ in dev_loader:
        ids = ids.to(get_device())
        with torch.no_grad():
            out = model(ids)
            preds = torch.argmax(out, dim=-1)
            for i in range(len(preds)):
                true_label = _true_label(labels[i], words[i])
                true_prediction = _true_label(preds[i], words[i])
                distances.append(abs(true_label - true_prediction))

    return sum(distances) / len(distances)


@dataclass
class TrainingArguments:
    model: CCModel = None
    tokenizer: CCTokenizer = None
    epochs: int = None
    lr: float = None
    optimizer: torch.optim.Optimizer = None
    save_every: int = None
    train_loader: DataLoader = None
    dev_loader: DataLoader = None
    checkpoint_prefix: str = "cc_labeller"
    criterion: torch.nn.Module = None


def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)
    args.model.train()

    for epoch in range(args.epochs):
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for ids, labels, _, attentions in args.train_loader:
                ids = ids.to(get_device())
                labels = labels.to(get_device())
                args.optimizer.zero_grad()
                
                loss = torch.tensor(0.0, device=get_device())
                out = args.model(ids)
                
                for j in range(len(ids)):
                    a = attentions[j].cpu().tolist()
                    attention_bound = a.index(0) if 0 in a else len(a)
                    local_out = out[j][:attention_bound]
                    local_labels = labels[j][:attention_bound]
                    loss += args.criterion(local_out, local_labels)
                    
                loss.backward()
                args.optimizer.step()
                pbar.update(1)
                i += 1

                if i % args.save_every == 0 and i != 0:
                    best, metric = pt.for_steps(args.model, args.dev_loader)
                    args.model.train()
                    pbar.set_postfix({
                        "loss": loss.item(),
                        "latest": metric,
                        "best": best
                    })
        pt.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
        args.model.train()


def entry(args: Namespace):
    def arg(cmd):
        p = f"cc_labeller_{cmd.replace('-', '_')}"
        return args.__getattribute__(p)

    tokenizer = CCTokenizer.from_pretrained(
        arg("tokenizer-path"), max_len=arg("tokenizer-max-len"))

    train_ds = TaskC_Data(split="train")
    dev_ds = TaskC_Data(split="dev")
    train_dl = DataLoader(
        train_ds,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=CCTokenizer.collate_fn(tokenizer)
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=arg("batch-size"),
        shuffle=False,
        collate_fn=CCTokenizer.collate_fn(tokenizer)
    )

    model = CCModel.from_cc_checkpoint(arg("cc-path"))
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))

    training_arguments = TrainingArguments(
        model=model,
        tokenizer=tokenizer,
        epochs=arg("epochs"),
        lr=arg("lr"),
        optimizer=optimizer,
        save_every=arg("save-every"),
        train_loader=train_dl,
        dev_loader=dev_dl,
        checkpoint_prefix=arg("checkpoint-prefix"),
        criterion=torch.nn.NLLLoss()
    )

    train(training_arguments)
