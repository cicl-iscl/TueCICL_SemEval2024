from argparse import Namespace
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loader.data import TaskC_Data

from models.char_bilstm import CharBiLSTMTokenizer, ChariBiLSTM
from util.checkpoints import ProgressTracker
from util.device import get_device


def add_args(parser):
    group = parser.add_argument_group("char-bilstm")
    def p(cmd): return f"--char-bilstm-{cmd}"
    group.add_argument(p("tokenizer-path"), type=str, default=None)
    group.add_argument(p("tokenizer-max-len"), type=int, default=10000)
    group.add_argument(p("batch-size"), type=int, default=8)
    group.add_argument(p("train"), action="store_true", default=False)
    group.add_argument(p("epochs-extended"), type=int, default=10)
    group.add_argument(p("epochs-pure"), type=int, default=10)
    group.add_argument(p("lr"), type=float, default=1e-3)
    group.add_argument(p("save-every"), type=int, default=1000)
    group.add_argument(p("hidden-size"), type=int, default=2)
    group.add_argument(p("emb-size"), type=int, default=8)
    group.add_argument(p("num-layers"), type=int, default=2)
    group.add_argument(p("dropout"), type=float, default=0.2)
    group.add_argument(p("checkpoint-prefix"), type=str, default="char_bilstm")


def evaluate(model: ChariBiLSTM, dev_loader):
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
    model: ChariBiLSTM = None
    tokenizer: CharBiLSTMTokenizer = None
    epochs_extended: int = None
    epochs_pure: int = None
    optimizer: torch.optim.Optimizer = None
    save_every: int = None
    train_loader_pure: DataLoader = None
    train_loader_extended: DataLoader = None
    dev_loader: DataLoader = None
    checkpoint_prefix: str = "cc_labeller"
    criterion: torch.nn.Module = None


def perform_training_step(args, batch):
    ids, labels, _, attentions = batch
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


def train(args: TrainingArguments):
    i = 0
    pt = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)
    args.model.train()
    for epoch in range(args.epochs_extended):
        with tqdm(total=len(args.train_loader_extended)) as pbar:
            pbar.set_description(f"Extended epoch {epoch}")
            for batch in args.train_loader_extended:
                perform_training_step(args, batch)
                i += 1
                pbar.update(1)
                if i % args.save_every == 0 and i != 0:
                    pt.for_steps(args.model, args.dev_loader)
                    args.model.train()
        pt.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)

    for epoch in range(args.epochs_pure):
        with tqdm(total=len(args.train_loader_pure)) as pbar:
            pbar.set_description(f"Pure epoch {epoch}")
            for batch in args.train_loader_pure:
                perform_training_step(args, batch)
                i += 1
                pbar.update(1)
                if i % args.save_every == 0 and i != 0:
                    pt.for_steps(args.model, args.dev_loader)
                    args.model.train()
        _e = epoch + args.epochs_extended
        pt.for_epoch(args.model, args.optimizer, _e, args.dev_loader)


def entry(args: Namespace):
    def arg(cmd):
        p = f"char_bilstm_{cmd.replace('-', '_')}"
        return args.__getattribute__(p)

    tokenizer = CharBiLSTMTokenizer.from_pretrained(
        arg("tokenizer-path"), max_len=arg("tokenizer-max-len"))

    train_ds_ext = TaskC_Data(split="train")
    train_ds_ext.import_task_A()
    train_ds_pure = TaskC_Data(split="train")
    dev_ds = TaskC_Data(split="dev")

    train_dl_ext = DataLoader(
        train_ds_ext,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=CharBiLSTMTokenizer.collate_fn(tokenizer)
    )
    train_dl_pure = DataLoader(
        train_ds_pure,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=CharBiLSTMTokenizer.collate_fn(tokenizer)
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=arg("batch-size"),
        shuffle=False,
        collate_fn=CharBiLSTMTokenizer.collate_fn(tokenizer)
    )

    model = ChariBiLSTM(
        emb_size=arg("emb-size"),
        hidden_size=arg("hidden-size"),
        num_layers=arg("num-layers"),
        dropout=arg("dropout"),
        vocab_size=len(tokenizer.idx2word)
    )
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))

    training_arguments = TrainingArguments(
        model=model,
        tokenizer=tokenizer,
        epochs_extended=arg("epochs-extended"),
        epochs_pure=arg("epochs-pure"),
        optimizer=optimizer,
        save_every=arg("save-every"),
        train_loader_extended=train_dl_ext,
        train_loader_pure=train_dl_pure,
        dev_loader=dev_dl,
        checkpoint_prefix=arg("checkpoint-prefix"),
        criterion=torch.nn.NLLLoss()
    )

    train(training_arguments)
