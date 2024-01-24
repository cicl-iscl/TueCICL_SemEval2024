import pickle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.device import get_device
import re


class Word2VecClassifier(nn.Module):
    def __init__(
        self,
        pretrained_embeddings=None,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        vocab_size=None,
        emb_size=None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if pretrained_embeddings is not None:
            self.vocab_size = pretrained_embeddings.shape[0]
            self.emb_size = pretrained_embeddings.shape[1]
            self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)
        elif vocab_size is not None and emb_size is not None:
            self.vocab_size = vocab_size
            self.emb_size = emb_size
            self.emb = nn.Embedding(vocab_size, emb_size)
        else:
            raise ValueError(
                "Either pretrained_embeddings or vocab_size and emb_size must be provided")

        self.lstm = nn.LSTM(
            self.emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm2class = nn.Linear(hidden_size, 2)
        
    def to_device(self):
        self.emb.cpu()
        self.lstm.to(get_device())
        self.lstm2class.to(get_device())

    def forward(self, x):
        x: torch.Tensor = self.emb(x)
        x = x.to(get_device())
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        pred_class = self.lstm2class(x[:, -1, :])
        pred_class = F.log_softmax(pred_class, dim=-1)

        return pred_class

    def predict(self, x):
        pred_class, _ = self.forward(x)
        pred_class = pred_class.argmax(dim=1)
        return pred_class

    def save(self, path, extra={}):
        torch.save({
            "model": self.state_dict(),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "vocab_size": self.vocab_size,
            "emb_size": self.emb_size,
            **extra
        }, path)

    def __str__(self) -> str:
        return f"Word2VecClassifier(vocab_size={self.vocab_size}, emb_size={self.emb_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout})"

    @classmethod
    def from_pretrained(cls, path):
        data = torch.load(path, map_location=torch.device("cpu"))
        model = cls(
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            dropout=data["dropout"],
            vocab_size=data["vocab_size"],
            emb_size=data["emb_size"],
        )
        model.load_state_dict(data["model"])
        model.to_device()
        return model, data


class Word2VecTokenizer:
    WHITESPACE = "<WS>"
    NUMBER = "<NUM>"
    PUNCT = "<P>"
    UNK = "<UNK>"
    PAD = "<PAD>"

    def __init__(self, word2idx, idx2word, max_len=5000) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_len = max_len

    def _split(self, texts):
        spl = [
            text.lower().split(" ")
            for text in texts
        ]
        adj = [
            [" " if x == "" else x for x in text]
            for text in spl
        ]
        return adj

    def _get_subwords(self, word):
        # separate punctuation marks
        word = re.sub(r"([^\w\s])", r" \1 ", word)
        subwords = word.split()
        subwords = [" " if x == "" else x for x in subwords]
        if len(subwords) == 0:
            # input ' ' turns into empty list, correct that
            subwords = [" "]
        return subwords

    def _is_punct(self, word: str):
        return re.match(r"([^\w\s])", word)

    def _is_number(self, word: str):
        return re.match(r"^[0-9]+$", word)

    def _get_id(self, word: str):
        if word.isspace():
            return self.word2idx[self.WHITESPACE]
        elif self._is_punct(word):
            return self.word2idx[self.PUNCT]
        elif self._is_number(word):
            return self.word2idx[self.NUMBER]
        return self.word2idx.get(word, self.word2idx[self.UNK])

    def tokenize(self, texts):
        _ids = []
        _attentions = []

        texts = self._split(texts)

        for text_idx, text in enumerate(texts):
            text_ids = []
            for word_idx, word in enumerate(text):
                subwords = self._get_subwords(word)
                for subword in subwords:
                    text_ids.append(self._get_id(subword))
            _ids.append(text_ids)

        longest = max([len(x) for x in _ids])
        if longest > self.max_len:
            longest = self.max_len

        for i in range(len(_ids)):
            _ids[i] = _ids[i][:longest]
            l = len(_ids[i])
            _attentions.append([1] * l)
            if l < longest:
                _ids[i] += [self.word2idx[self.PAD]] * (longest - l)
                _attentions[i] += [0] * (longest - l)
        return (
            torch.tensor(_ids),
            torch.tensor(_attentions)
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.word2idx, f)

    @classmethod
    def from_txt(cls, path, emb_size, max_len=5000):
        word2idx = {}
        idx2word = {}
        weights = []
        print("Loading word2vec from", path)
        with open(path, "r") as f:
            next(f)
            i = 0
            for line in f:
                if not line:
                    continue
                spl = line.split()
                vec = [float(x) for x in spl[-emb_size:]]
                word = " ".join(spl[:-emb_size])

                if len(word.split()) > 1 or "ENTITY/" in word:
                    continue

                word2idx[word] = i
                idx2word[i] = word
                weights.append(vec)
                i += 1

        for token in [cls.WHITESPACE, cls.NUMBER, cls.UNK, cls.PAD, cls.PUNCT]:
            word2idx[token] = len(word2idx)
            idx2word[len(idx2word)] = token
            weights.append([0.0] * emb_size)

        c = cls(word2idx, idx2word, max_len=max_len)
        return c, torch.tensor(weights)

    @classmethod
    def from_pretrained(cls, path, max_len=5000):
        with open(path, "rb") as f:
            word2idx = pickle.load(f)
        idx2word = {v: k for k, v in word2idx.items()}
        return cls(word2idx, idx2word, max_len=max_len)

    @staticmethod
    def collate_fn(tokenizer):
        def collate_batch(batch):
            texts = [x[0] for x in batch]
            labels = [x[1] for x in batch]
            labels = torch.tensor(labels)
            input_ids, attentions = tokenizer.tokenize(
                texts)
            return input_ids, labels, attentions

        return collate_batch
