from argparse import Namespace
from dataclasses import dataclass
from threading import local

import torch
from tqdm import tqdm
from loader.data import TaskC_Data
from util.checkpoints import ProgressTracker

from util.device import get_device
from models.word2vec_bilstm import Word2VecBiLSTM, Word2VecTokenizer
from torch.utils.data import DataLoader


def add_args(parser):
    group = parser.add_argument_group("char-bilstm")
    def p(cmd): return f"--word2vec-bilstm-{cmd}"
    group.add_argument(p("tokenizer-path"), type=str, default=None)
    group.add_argument(p("tokenizer-vocab"), type=str, default=None)
    group.add_argument(p("tokenizer-weights"), type=str, default=None)
    group.add_argument(p("save-vocab"), type=str, default=None)
    group.add_argument(p("save-weights"), type=str, default=None)
    group.add_argument(p("emb-size"), type=int, default=500)
    group.add_argument(p("tokenizer-max-len"), type=int, default=10000)
    group.add_argument(p("batch-size"), type=int, default=8)
    group.add_argument(p("train"), action="store_true", default=False)
    group.add_argument(p("epochs-extended"), type=int, default=10)
    group.add_argument(p("epochs-pure"), type=int, default=10)
    group.add_argument(p("lr"), type=float, default=1e-3)
    group.add_argument(p("save-every-extended"), type=int, default=1000)
    group.add_argument(p("save-every-pure"), type=int, default=1000)
    group.add_argument(p("hidden-size"), type=int, default=2)
    group.add_argument(p("num-layers"), type=int, default=2)
    group.add_argument(p("dropout"), type=float, default=0.2)
    group.add_argument(p("checkpoint-prefix"), type=str, default="char_bilstm")
    group.add_argument(p("load-model"), type=str, default=None)
    group.add_argument(p("use-parallel"), action="store_true", default=False)


def evaluate(_model, dev_loader):
    _model.eval()
    if hasattr(_model, "module"):
        model = _model.module
    else:
        model = _model
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
    model: Word2VecBiLSTM = None
    tokenizer: Word2VecBiLSTM = None
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


def perform_training_step(args, batch):
    ids, labels, _, attentions = batch
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
    return loss.item()


def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)
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
                if i % args.save_every_pure == 0 and i != 0:
                    l = sum(losses) / len(losses)
                    losses = []
                    best, latest = pt.for_steps(
                        args.model, args.dev_loader, loss=l)
                    pbar.set_postfix(best=best, latest=latest)
                    args.model.train()
        _e = epoch + args.epochs_extended
        pt.for_epoch(args.model, args.optimizer, _e, args.dev_loader)
        args.model.train()


def entry(args: Namespace):
    def arg(cmd):
        p = f"word2vec_bilstm_{cmd.replace('-', '_')}"
        return args.__getattribute__(p)

    weights = None
    if arg("tokenizer-vocab"):
        tokenizer = Word2VecTokenizer.from_pretrained(arg("tokenizer-vocab"))
        if arg("load-model") is None and arg("tokenizer-weights"):
            weights = torch.load(arg("tokenizer-weights"))
        elif arg("load-model") is None and arg("tokenizer-weights") is None:
            raise ValueError("No pretrained weights provided")
    elif arg("tokenizer-path"):
        tokenizer, weights = Word2VecTokenizer.from_txt(arg("tokenizer-path"), emb_size=arg("emb-size"))
        if arg("save-vocab"):
            tokenizer.save(arg("save-vocab"))
        if arg("save-weights"):
            torch.save(weights, arg("save-weights"))
    else:
        raise ValueError(
            "Either tokenizer-path or tokenizer-vocab must be provided")

    if arg("load-model"):
        model = Word2VecBiLSTM.from_pretrained(arg("load-model"))
    else:
        if weights is None:
            raise ValueError("No pretrained weights provided")
        model = Word2VecBiLSTM(
            pretrained_embeddings=weights,
            hidden_size=arg("hidden-size"),
            num_layers=arg("num-layers"),
            dropout=arg("dropout"),
        )
    print(model)
    if arg("use-parallel"):
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg("lr"))

    train_ds_ext = TaskC_Data(split="train")
    train_ds_ext.import_task_A()
    train_ds_pure = TaskC_Data(split="train")
    dev_ds = TaskC_Data(split="dev")

    train_dl_ext = DataLoader(
        train_ds_ext,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=Word2VecTokenizer.collate_fn(tokenizer),
        drop_last=True
    )
    train_dl_pure = DataLoader(
        train_ds_pure,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=Word2VecTokenizer.collate_fn(tokenizer),
        drop_last=True
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=arg("batch-size"),
        shuffle=False,
        collate_fn=Word2VecTokenizer.collate_fn(tokenizer, check_label_mismatch=True),
    )

    training_arguments = TrainingArguments(
        model=model,
        tokenizer=tokenizer,
        epochs_extended=arg("epochs-extended"),
        epochs_pure=arg("epochs-pure"),
        optimizer=optimizer,
        save_every_extended=arg("save-every-extended"),
        save_every_pure=arg("save-every-pure"),
        train_loader_extended=train_dl_ext,
        train_loader_pure=train_dl_pure,
        dev_loader=dev_dl,
        checkpoint_prefix=arg("checkpoint-prefix"),
        criterion=torch.nn.NLLLoss()
    )

    train(training_arguments)
