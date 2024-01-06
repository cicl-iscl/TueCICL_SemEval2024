import pickle
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.device import get_device


class CharLM(nn.Module):
    def __init__(self, vocab_size=None, emb_size=8, hidden_size=1024, num_layers=1, aggregate_fn="mean") -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.aggregate_fn = aggregate_fn

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            hidden_size=self.hidden_size,
            input_size=self.emb_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.lstm2lm = nn.Linear(hidden_size, vocab_size)
        self.lstm2class = nn.Linear(hidden_size, 2)

    def _get_means(self, tensors, attentions):
        # Function computes embedding averages, but taking out the PAD tokens, as they should not contribute to MEAN.
        # Written kinda complicated to avoid copying tensors as much as possible, a previous solution was extremely slow because of
        # such copy and slice operations
        batch_size, seq_len, hidden_size = tensors.shape
        filter = attentions.reshape((batch_size, seq_len, 1)).expand(
            (batch_size, seq_len, hidden_size))
        filtered = torch.where(filter > 0, tensors, 0)

        l = attentions.sum(dim=1).reshape(-1, 1)
        s = filtered.sum(dim=1)

        for i in range(batch_size):
            if l[i] == 0:
                l[i] = 1

        return s / l

    def _get_max(tensors):
        return torch.max(tensors, dim=1)

    def aggregate(self, tensors, attentions):
        if self.aggregate_fn == "mean":
            return self._get_means(tensors, attentions)
        elif self.aggregate == "max":
            return self._get_max(tensors)
        else:
            raise NotImplementedError()

    def forward(self, input_ids, attention, lstm_hidden=None):
        if lstm_hidden is None:
            lstm_hidden = self.init_lstm_hidden(input_ids.shape[0])

        embedded = self.emb(input_ids)
        out, lstm_hidden = self.lstm(embedded, lstm_hidden)

        lm_out = self.lstm2lm(out)
        lm_out = F.log_softmax(lm_out, dim=-1)
        means_for_classification = self._get_means(out, attention)
        classification_out = self.lstm2class(means_for_classification)
        classification_out = F.log_softmax(classification_out, dim=-1)
        return lm_out, classification_out, lstm_hidden

    def init_lstm_hidden(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size),
                        device=get_device())
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size),
                        device=get_device())
        return (h, c)

    def save(self, path, extra={}):
        save_data = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "emb_size": self.emb_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "aggregate_fn": self.aggregate_fn,
            **extra
        }
        torch.save(save_data, path)

    @classmethod
    def from_pretrained(cls, path):
        save_data = torch.load(path)
        model = cls(
            vocab_size=save_data["vocab_size"],
            emb_size=save_data["emb_size"],
            hidden_size=save_data["hidden_size"],
            num_layers=save_data["num_layers"],
            aggregate_fn=save_data["aggregate_fn"]
        )
        model.load_state_dict(save_data["state_dict"])
        return model


class CharLMTokenizer:
    def __init__(self, idx2word, word2idx, vocab) -> None:
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.vocab = vocab

        self.WHITESPACE = "<WS>"
        self.PUNCTUATION = "<PUNCT>"
        self.DIGIT = "<DIGIT>"
        self.UNK = "<UNK>"
        self.SENT_TERMINATE = "<SENT_TERMINATE>"

        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"

    def map_char(self, char):
        return char

    def tokenize(self, texts: List[str], max_len: int = None):
        tokenized_texts = [[c for c in text.lower()] for text in texts]
        longest = max([len(text) for text in tokenized_texts])
        if max_len is not None and longest > max_len:
            longest = max_len
        tokenized_texts = [text[:longest] for text in tokenized_texts]

        attentions = []
        input_ids = []

        for text in tokenized_texts:
            # +1 because of BOS
            _ids = [self.word2idx[self.BOS]]
            _attentions = [1]
            for i in range(longest + 1):
                if i < len(text):
                    _ids.append(self.word2idx.get(
                        self.map_char(text[i]), self.word2idx[self.UNK]))
                    _attentions.append(1)
                elif i == len(text):
                    _ids.append(self.word2idx[self.EOS])
                    _attentions.append(1)
                else:
                    _ids.append(self.word2idx[self.PAD])
                    _attentions.append(0)
            input_ids.append(_ids)
            attentions.append(_attentions)

        return torch.tensor(input_ids, device=get_device()), torch.tensor(attentions, device=get_device())

    def save(self, path):
        pass

    def build(self, texts):
        pass

    @classmethod
    def from_pretrained(cls, path):
        with open(path, "rb") as f:
            word2idx, idx2word, vocab = pickle.load(f)
        return cls(idx2word, word2idx, vocab)
