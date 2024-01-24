from argparse import Namespace
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support


import torch
from tqdm import tqdm
from models.word2vec import Word2VecClassifier, Word2VecTokenizer

from util.checkpoints import ProgressTracker
from loader.data import TaskA_Dataset, collate_fn
from util.device import get_device


def add_args(parser):
    def p(name): return "--word2vec-classifier-" + name
    group = parser.add_argument_group("Word2Vec Classifier")
    group.add_argument(p("hidden-size"), type=int, default=128)
    group.add_argument(p("num-layers"), type=int, default=1)
    group.add_argument(p("dropout"), type=float, default=0.0)
    group.add_argument(p("lr"), type=float, default=1e-3)
    group.add_argument(p("batch-size"), type=int, default=6)
    group.add_argument(p("n-epochs"), type=int, default=10)
    group.add_argument(p("start-epoch"), type=int, default=1)
    group.add_argument(p("save-every"), type=int, default=1)
    group.add_argument(p("do-train"), default=False, action="store_true")
    group.add_argument(p("checkpoint-prefix"),
                       type=str, default="word2vec-classifier")
    group.add_argument(p("max-len"), type=int, default=15_000)
    group.add_argument(p("load-model"), type=str, default=None)

    group.add_argument(p("emb-size"), type=int, default=500)
    group.add_argument(p("tokenizer-extend"),
                       default=False, action="store_true")
    group.add_argument(p("tokenizer-txt-path"), type=str, default=None)
    group.add_argument(p("tokenizer-save-path-weights"),
                       type=str, default=None)
    group.add_argument(p("tokenizer-save-path-vocab"), type=str, default=None)
    group.add_argument(p("tokenizer-pkl-path-vocab"), type=str, default=None)
    group.add_argument(p("tokenizer-pkl-path-weights"), type=str, default=None)


def evaluate(model: Word2VecClassifier, dev_loader: torch.utils.data.DataLoader, f1_only=True):
    y_pred = []
    y_gold = []
    model.eval()
    with torch.no_grad():
        for input_ids, attentions, labels in dev_loader:
            out = model(input_ids)
            pred = torch.argmax(out, dim=1)
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
class TrainingArgumets:
    model: Word2VecClassifier
    optimizer: torch.optim.Optimizer
    tokenizer: Word2VecTokenizer
    train_loader: torch.utils.data.DataLoader
    dev_loader: torch.utils.data.DataLoader
    start_epoch: int = 1
    n_epochs: int = 5
    save_every: int = 100
    clip: float = None
    checkpoint_prefix: str = "word2vec-classifier"
    criterion: torch.nn.Module = None


def train_classifier(args: TrainingArgumets):
    cp = ProgressTracker(args.checkpoint_prefix, evaluate_fn=evaluate)
    i = 0
    args.model.train()

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        with tqdm(total=len(args.train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for input_ids, attentions, labels in args.train_loader:
                labels: torch.Tensor = labels.to(get_device())
                args.optimizer.zero_grad()
                try:
                    out = args.model(input_ids)
                except:
                    print(input_ids)
                    raise
                loss = args.criterion(out, labels)

                loss.backward()
                args.optimizer.step()

                pbar.update(1)
                i += 1

                if i % args.save_every == 0 and i != 0:
                    best, latest = cp.for_steps(args.model, args.dev_loader)
                    args.model.train()
                    pbar.set_postfix({
                        "best": best,
                        "latest": latest
                    })
            cp.for_epoch(args.model, args.optimizer, epoch, args.dev_loader)
            args.model.train()


def entry(args: Namespace):
    def arg(cmd):
        p = f"word2vec_classifier_{cmd.replace('-', '_')}"
        return args.__getattribute__(p)

    weights = None
    if arg("tokenizer-pkl-path-vocab"):
        tokenizer = Word2VecTokenizer.from_pretrained(
            arg("tokenizer-pkl-path-vocab"),
            max_len=arg("max-len")
        )
        if arg("load-model") is None and arg("tokenizer-pkl-path-weights"):
            weights = torch.load(arg("tokenizer-pkl-path-weights"))
        elif arg("load-model") is not None and arg("tokenizer-pkl-path-weights") is None:
            raise ValueError("Cannot load model without weights")
    elif arg("tokenizer-txt-path"):
        tokenizer, weights = Word2VecTokenizer.from_txt(
            arg("tokenizer-txt-path"),
            arg("emb-size"),
            max_len=arg("max-len")
        )
        if arg("tokenizer-save-path-vocab"):
            tokenizer.save(arg("tokenizer-save-path-vocab"))
        if arg("tokenizer-save-path-weights"):
            torch.save(weights, arg("tokenizer-save-path-weights"))
    else:
        raise ValueError("No tokenizer specified")

    if arg("load-model"):
        model, checkpoint = Word2VecClassifier.from_pretrained(
            arg("load-model"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state dict")
    else:
        if weights is None:
            raise ValueError("No weights specified")
        model = Word2VecClassifier(
            pretrained_embeddings=weights,
            dropout=arg("dropout"),
            emb_size=arg("emb-size"),
            hidden_size=arg("hidden-size"),
            num_layers=arg("num-layers")
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=arg("lr"))

    print(model)

    if arg("do-train"):
        ds = TaskA_Dataset(split="train")
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.word2vec_classifier_batch_size,
            shuffle=True,
            collate_fn=collate_fn(
                Word2VecTokenizer.collate_fn(tokenizer))
        )
        dev_loader = torch.utils.data.DataLoader(
            TaskA_Dataset(split="dev"),
            batch_size=args.word2vec_classifier_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn(
                Word2VecTokenizer.collate_fn(tokenizer, check_label_mismatch=True))
        )
        criterion = torch.nn.NLLLoss()
        arguments = TrainingArgumets(
            checkpoint_prefix=args.word2vec_classifier_checkpoint_prefix,
            clip=None,
            dev_loader=dev_loader,
            model=model,
            n_epochs=args.word2vec_classifier_n_epochs,
            start_epoch=args.word2vec_classifier_start_epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            save_every=args.word2vec_classifier_save_every,
            train_loader=train_loader,
            criterion=criterion
        )
        train_classifier(arguments)
