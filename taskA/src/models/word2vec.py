from curses.ascii import isspace
import pickle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.device import get_device


class Word2VecClassifier(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        hidden_size=128,
        num_layers=1,
        aggregate_fn="mean",
        dropout=0.0,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.aggregate_fn = aggregate_fn
        self.dropout = dropout
        self.emb_size = pretrained_embeddings.shape[1]
        self.vocab_size = pretrained_embeddings.shape[0]

        self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.lstm = nn.LSTM(
            self.emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm2class = nn.Linear(hidden_size, 2)
        self.lstm2lm = nn.Linear(hidden_size, self.vocab_size)

        self.emb.cpu()
        self.lstm.to(get_device())
        self.lstm2class.to(get_device())
        self.lstm2lm.to(get_device())

    def forward(self, x):
        x = self.emb(x)
        x.to(get_device())
        x, _ = self.lstm(x)
        pred_class = self.lstm2class(x[:, -1, :])
        pred_class = F.log_softmax(pred_class, dim=-1)
        pred_lm = self.lstm2lm(x)
        pred_lm = F.log_softmax(pred_lm, dim=-1)

        return pred_class, pred_lm

    def predict(self, x):
        pred_class, _ = self.forward(x)
        pred_class = pred_class.argmax(dim=1)
        return pred_class


class Word2VecTokenizer:
    UNK = "<UNK>"
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    WHITESPACE = "<WS>"
    PUNCTUATION = "<PUNCT>"

    def __init__(self, idx2word, word2idx, vocab, weights) -> None:
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.weights = weights

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.idx2word, self.word2idx, self.weights), f)

    def _get_ids(self, tokens):
        _ids = []
        def idof(token): return self.word2idx.get(
            token, self.word2idx[self.UNK])
        for token in tokens:
            if string.isspace(token):
                _ids.append(idof(self.WHITESPACE))
            elif token in string.punctuation:
                _ids.append(idof(self.PUNCTUATION))
            else:
                _ids.append(idof(token))

    def tokenize(self, texts, add_special_tokens=True, max_len=None):
        tokens = [text.lower().split(" ") for text in texts]
        longest = max([len(x) for x in tokens])
        if max_len is not None:
            longest = min(longest, max_len)
        tokens = [x[:longest] for x in tokens]
        if add_special_tokens:
            tokens = [[self.BOS] + x + [self.EOS] for x in tokens]
            longest = longest + 2
        attentions = []
        for i, text in enumerate(tokens):
            attentions[i] = [1] * len(text)
            if len(text) < longest:
                attentions[i] = attentions[i] + [0] * (longest - len(text))
                tokens[i] += [self.PAD] * (longest - len(text))
        unk_id = self.word2idx[self.UNK]
        ids = [[self.word2idx.get(x, unk_id) for x in text]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attentions, dtype=torch.long)

    def extend(self, texts):
        for text in texts:
            spl = text.lower().split()
            for token in spl:
                if token not in self.word2idx:
                    self.weights.append([0.0] * self.emb_size)
                    self.word2idx[token] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = token

    @classmethod
    def from_txt(cls, path, auto_save=True, emb_size=500):
        word2idx = {}
        idx2word = {}
        weights = []
        print("Loading word2vec from", path)
        with open(path, "r") as f:
            next(f)  #  skip header
            for i, line in enumerate(f):
                if not line:
                    continue
                try:
                    spl = line.split()
                    vec = [float(x) for x in spl[-emb_size:]]
                    word = " ".join(spl[:-emb_size])
                    weights.append(vec)
                    word2idx[word] = i
                    idx2word[i] = word
                except Exception as e:
                    print(e)
                    print(line)

        for token in [cls.UNK, cls.PAD, cls.BOS, cls.EOS, cls.WHITESPACE, cls.PUNCTUATION]:
            word2idx[token] = len(word2idx)
            idx2word[len(idx2word)] = token
            weights.append([0.0] * emb_size)

        o = cls(idx2word, word2idx, weights)
        if auto_save:
            o.save(path.replace(".txt", ".pkl"))
        return o

    @classmethod
    def from_pkl(cls, path):
        with open(path, "rb") as f:
            idx2word, word2idx, weights = pickle.load(f)
        return cls(idx2word, word2idx, weights)
