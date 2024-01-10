import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import typing as T
import string
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pickle
from sklearn.metrics import classification_report, precision_recall_fscore_support
import os

torch.manual_seed(42)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(device)

def get_data():
    train = pd.read_json("./data/subtaskA_train_monolingual.jsonl", lines=True)
    dev = pd.read_json("./data/subtaskA_dev_monolingual.jsonl", lines=True)
    return train, dev

train, dev = get_data()

WHITESPACE = "<WS>"
PUNCTUATION = "<PUNCT>"
DIGIT = "<DIGIT>"
UNK = "<UNK>"
SENT_TERMINATE = "<SENT_TERMINATE>"

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

CONDENSED_VOCAB = False


def map_char(char: str, condense=True):
    if not condense:
        return char

    sentence_ending = [".", "!", "?"]
    if char.isspace():
        return WHITESPACE
    if char in sentence_ending:
        return SENT_TERMINATE
    if char in string.punctuation:
        return PUNCTUATION
    if char in string.digits:
        return DIGIT
    if char not in string.printable:
        return UNK
    return char


def build_vocab(train_set: pd.DataFrame, condense=True):
    vocab = set([UNK])
    for _, series in train_set.iterrows():
        text: str = series["text"]
        tokens: T.List[str] = [*text.lower().strip()]
        tokens = [map_char(token, condense=condense) for token in tokens]
        for token in tokens:
            vocab.add(token)
    vocab = list(vocab)

    vocab.append(BOS)
    vocab.append(EOS)
    vocab.append(PAD)

    word2idx = {
        word: idx for idx, word in enumerate(vocab)
    }
    idx2word = {
        idx: word for idx, word in enumerate(vocab)
    }
    return word2idx, idx2word, vocab


def get_vocab(condense=True):
    fp = "./data/charlm_vocab.pkl" if condense else "./data/charlm_vocab_uncondensed.pkl"
    try:
        with open(fp, "rb") as f:
            return pickle.load(f)
    except:
        train, _ = get_data()
        res = build_vocab(train, condense=condense)
        with open(fp, "wb") as f:
            pickle.dump(res, f)
        return res


word2idx, idx2word, vocab = get_vocab(condense=CONDENSED_VOCAB)

print(len(vocab))


def get_text_tokens(text: str, condense=False):
    tokens: T.List[str] = [*text.lower().strip()]
    tokens = [map_char(token, condense=condense) for token in tokens]
    return tokens


def tokenize(texts: T.List[str], max_len=None, add_special_tokens=True, condense=False):
    tokenized_texts = [get_text_tokens(t, condense=condense) for t in texts]

    longest_len = max([len(t) for t in tokenized_texts])
    if max_len and max_len < longest_len:
        longest_len = max_len
    tokenized_texts = [t[:longest_len] for t in tokenized_texts]

    tokens, attentions = [], []
    for tokenized_text in tokenized_texts:

        pad_amount = longest_len - len(tokenized_text)
        if add_special_tokens:
            tokenized_text = [BOS] + tokenized_text + [EOS]

        tokenized_text += [PAD] * (pad_amount)
        tokens.append([word2idx.get(token, word2idx[UNK]) for token in tokenized_text])
        attentions.append([1 if token != PAD else 0 for token in tokenized_text])
    return torch.tensor(tokens, device=device), torch.tensor(attentions, device=device)


def decode(tokens: T.List[T.List[int]]):
    return [[idx2word[token] for token in tokenized_text] for tokenized_text in tokens]


class TaskA_Dataset(Dataset):
    def __init__(self, split="train") -> None:
        if split == "train":
            self.data = pd.read_json("./data/subtaskA_train_monolingual.jsonl", lines=True)
        else:
            self.data = pd.read_json("./data/subtaskA_dev_monolingual.jsonl", lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text, label, _id = item["text"], item["label"], item["id"]
        return text, label, _id



class CharLM(nn.Module):
    def __init__(self, vocab_size=None, emb_size=8, hidden_size=1024, num_layers=1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            hidden_size=hidden_size,
            input_size=emb_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lstm2lm = nn.Linear(hidden_size, vocab_size)
        self.lstm2class = nn.Linear(hidden_size, 2)

    def _get_means(self, tensors, attentions):
        # Function computes embedding averages, but taking out the PAD tokens, as they should not contribute to MEAN.
        # Written kinda complicated to avoid copying tensors as much as possible, a previous solution was extremely slow because of
        # such copy and slice operations
        batch_size, seq_len, hidden_size = tensors.shape
        filter = attentions.reshape((batch_size, seq_len, 1)).expand((batch_size, seq_len, hidden_size))
        filtered = torch.where(filter > 0, tensors, 0)
        l = attentions.sum(dim=1).reshape(-1, 1)
        s = tensors.sum(dim=1)
        return s / l

    def forward(self, input_ids, attention):
        embedded = self.emb(input_ids)
        out, _ = self.lstm(embedded)
        lm_out = self.lstm2lm(out)
        if isnan(lm_out[0]):
            print("gotcha!")
        lm_out = F.log_softmax(lm_out, dim=-1)
        means_for_classification = self._get_means(out, attention)
        classification_out = self.lstm2class(means_for_classification)
        classification_out = F.log_softmax(classification_out, dim=-1)
        return lm_out, classification_out


def collate_fn(data):
    labels = [i[1] for i in data]
    texts = [i[0] for i in data]
    ids = [i[2] for i in data]
    max_len = 10_000
    input_ids, attentions = tokenize(texts, max_len=max_len, condense=CONDENSED_VOCAB)
    return input_ids, attentions, torch.tensor(labels, device=device), torch.tensor(ids, device=device)


def evaluate(model, dataset):
    dev_dataloader = DataLoader(dataset, shuffle=False, batch_size=4, collate_fn=collate_fn)
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for input_ids, attentions, labels, _ in dev_dataloader:
            _, out = model(input_ids, attentions)
            for i in range(out.shape[0]):
                pred = torch.argmax(out[i]).item()
                y_pred.append(pred)
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(y_gold, y_pred, average="macro", zero_division=0.0)
    return f1, r

def make_checkpoint(model, optimizer, epoch, prefix="classifier", report=None, progress=None):
    dirpath = f"checkpoints/{prefix}"
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "report": report,
        "progress": progress
    }
    torch.save(checkpoint, f"{dirpath}/epoch_{epoch}.pt")


def save_best(model, step=None, progress=None, prefix=None):
    dirpath = f"checkpoints/{prefix}"
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass
    fpath = f"{dirpath}/latest.pt"
    fpath_best = f"{dirpath}/best.pt"
    f1, report = evaluate(model, TaskA_Dataset(split="dev"))

    if "best" not in progress:
        progress["best"] = 0

    is_best = f1 > progress["best"]

    cp = {
        "model": model.state_dict(),
        "progress": progress,
        "step": step,
        "report": report
    }

    if is_best:
        progress["best"] = f1
        torch.save(cp, fpath_best)
    torch.save(cp, fpath)
    return progress["best"], f1


from math import isnan

last_batch = None


def train(
        model=None,
        optimizer=None,
        dataloader=None,
        n_epochs=5,
        start_epoch=1,
        checkpoint_prefix=None,
        save_every=1000,
        progress={},
        clip=5
):
    model.train()
    lm_criterion = nn.NLLLoss(reduction="mean")
    cl_criterion = nn.NLLLoss(reduction="sum")

    step = 0

    for epoch in range(start_epoch, n_epochs + 1):
        with tqdm(total=len(dataloader)) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            losses = []
            for batch in dataloader:
                input_ids, attentions, labels, text_ids = batch
                optimizer.zero_grad()

                lm_out, classifier_out = model(input_ids, attentions)
                loss = torch.tensor(0, dtype=torch.float32, device=device)

                # ------------------
                # LM loss
                # ------------------

                for i in range(input_ids.shape[0]):
                    if labels[i].item() == 0:
                        # only train LM on human texts
                        until = attentions[i].cpu().tolist().index(0) if 0 in attentions[i] else -1
                        y_pred = lm_out[i, :until]
                        y_gold = input_ids[i, :until]
                        y_pred = y_pred[:-1]
                        y_gold = y_gold[1:]
                        loss_update = lm_criterion(y_pred, y_gold)
                        if isnan(loss_update.item()):
                            print("NAN loss detected, do not backprop")
                            print(text_ids)
                            print(i)
                            return
                        loss += loss_update

                # ------------------
                # Classifier loss
                # ------------------
                loss_update = cl_criterion(classifier_out, labels)
                loss += loss_update

                # ------------------
                # Backprop
                # ------------------

                losses.append(loss.item())
                loss.backward()

                optimizer.step()
                pbar.update(1)
                last_batch = batch

                step += 1
                if step != 0 and step % save_every == 0:
                    best_f1, latest_f1 = save_best(model, step=step, progress=progress, prefix=checkpoint_prefix)
                    pbar.set_postfix({"f1": best_f1, "latest": latest_f1})

            print("LOSS", sum(losses) / len(losses))
            _, report = evaluate(model, TaskA_Dataset(split="dev"))
            print(report)
            make_checkpoint(model, optimizer, epoch, prefix=checkpoint_prefix, report=report)


model = CharLM(
    vocab_size=len(vocab),
    hidden_size=512,
    num_layers=2
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=0.005)
start_epoch = 1
checkpoint_prefix="charlm_256_1_fullvocab"
progress = {}

cp_file = None
CP = f"checkpoints/{cp_file}" if cp_file else None

if CP:
    checkpoint_data = torch.load(CP)
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    start_epoch = checkpoint_data["epoch"] + 1
    progress = checkpoint_data.get("progress", {})
    print("-------------------------")
    print("CHECKPOINT MODEL EVAL")
    print("-------------------------")
    if "report" in checkpoint_data:
        print(checkpoint_data["report"])
    else:
        evaluate(model, TaskA_Dataset(split="dev"))
    print()


ds = TaskA_Dataset(split="train")

loader = DataLoader(
    ds,
    shuffle=True,
    batch_size=4,
    collate_fn=collate_fn
)

losses = train(
    model=model,
    optimizer=optimizer,
    dataloader=loader,
    n_epochs=10,
    start_epoch=start_epoch,
    checkpoint_prefix=checkpoint_prefix,
    save_every=4000,
    progress=progress,
)