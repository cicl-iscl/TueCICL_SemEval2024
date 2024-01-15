from argparse import Namespace
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support


import torch
from models.word2vec import Word2VecClassifier, Word2VecTokenizer

from util.checkpoints import ProgressTracker
from loader.data import TaskA_Dataset
from util.device import get_device


def add_args(parser):
    def p(name): return "--word2vec-classifier-" + name
    group = parser.add_argument_group("Word2Vec Classifier")
    group.add_argument(p("hidden-size"), type=int, default=128)
    group.add_argument(p("num-layers"), type=int, default=1)
    group.add_argument(p("aggregate-fn"), type=str,
                       default="last", choices=["mean", "max", "last"])
    group.add_argument(p("dropout"), type=float, default=0.0)
    group.add_argument(p("lr"), type=float, default=1e-3)
    group.add_argument(p("batch-size"), type=int, default=6)
    group.add_argument(p("n-epochs"), type=int, default=10)
    group.add_argument(p("start-epoch"), type=int, default=1)
    group.add_argument(p("save-every"), type=int, default=1)
    group.add_argument(p("do-train"), type=bool, default=True)
    group.add_argument(p("checkpoint-prefix"),
                       type=str, default="word2vec-classifier")
    group.add_argument(p("max-len"), type=int, default=15_000)
    group.add_argument(p("load-model"), type=str, default=None)

    group.add_argument(p("emb-size"), type=int, default=None, required=True)
    group.add_argument(p("tokenizer-extend"), type=bool, default=True)
    group.add_argument(p("tokenizer-txt-path"), type=str, default=None)
    group.add_argument(p("tokenizer-save-path"), type=str, default=None)
    group.add_argument(p("tokenizer-pkl-path"), type=str, default=None)


def evaluate(model: Word2VecClassifier, dev_loader: torch.utils.data.DataLoader, f1_only=True):
    y_pred = []
    y_gold = []
    _model = model.module if model.module else model
    with torch.no_grad():
        for input_ids, attentions, labels in dev_loader:
            pred = _model.predict(input_ids, attentions)
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
    device: str = get_device()
    clip: float = None
    checkpoint_prefix: str = "word2vec-classifier"


def train_classifier(args: TrainingArgumets):
    cp = ProgressTracker(args.checkpoint_prefix)


def entry(args: Namespace):

    ds = TaskA_Dataset(split="train")
    if args.word2vec_classifier_tokenizer_pkl_path:
        tokenizer = Word2VecTokenizer.from_pkl(
            args.word2vec_classifier_tokenizer_pkl_path)
    else:
        tokenizer = Word2VecTokenizer.from_txt(
            args.word2vec_classifier_tokenizer_txt_path, auto_save=False, emb_size=args.word2vec_classifier_emb_size)

        if args.word2vec_classifier_tokenizer_extend:
            tokenizer.extend((i[0] for i in ds))

        if args.word2vec_classifier_tokenizer_save_path:
            tokenizer.save(args.word2vec_classifier_tokenizer_save_path)

    if args.word2vec_classifier_load_model:
        model = Word2VecClassifier.from_pretrained(
            args.word2vec_classifier_load_model)
    else:
        model = Word2VecClassifier(
            aggregate_fn=args.word2vec_classifier_aggregate_fn,
            dropout=args.word2vec_classifier_dropout,
            hidden_size=args.word2vec_classifier_hidden_size,
            num_layers=args.word2vec_classifier_num_layers,
            pretrained_embeddings=torch.tensor(tokenizer.weights)
        )

    if args.word2vec_classifier_do_train:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.word2vec_classifier_lr)
        train_loader = torch.utils.data.DataLoader(
            ds, batch_size=args.word2vec_classifier_batch_size, shuffle=True, drop_last=True)
        dev_loader = torch.utils.data.DataLoader(
            TaskA_Dataset(split="dev"), batch_size=args.word2vec_classifier_batch_size, shuffle=True, drop_last=True)
        arguments = TrainingArgumets(
            checkpoint_prefix=args.word2vec_classifier_checkpoint_prefix,
            clip=None,
            dev_loader=dev_loader,
            device=get_device(),
            model=model,
            n_epochs=args.word2vec_classifier_n_epochs,
            start_epoch=args.word2vec_classifier_start_epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            save_every=args.word2vec_classifier_save_every,
            train_loader=train_loader
        )
        train_classifier(arguments)
