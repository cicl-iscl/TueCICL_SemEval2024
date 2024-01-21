import torch
import torch.nn as nn
import torch.nn.functional as F
from util.device import get_device
from typing import List
import pickle


class ChariBiLSTM(nn.Module):
    def __init__(self, vocab_size=None, emb_size=8, hidden_size=1024, num_layers=1, dropout=0.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            hidden_size=self.hidden_size,
            input_size=self.emb_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm2out = nn.Linear(hidden_size * 2, 2)

        self.emb.to(get_device())
        self.lstm.to(get_device())
        self.lstm2out.to(get_device())

    def forward(self, input_ids):
        embedded = self.emb(input_ids)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out

    def predict(self, input_ids):
        out, _ = self(input_ids)
        out = self.lstm2out(out)
        return out.argmax(dim=1)

    def save(self, path, extra={}):
        save_data = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "emb_size": self.emb_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
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
            dropout=save_data["dropout"],
        )
        model.load_state_dict(save_data["state_dict"])
        return model

    def __str__(self):
        return f"ChariBiLSTM<vocab={self.vocab_size}, emb={self.emb_size}, h={self.hidden_size}, l={self.num_layers}>"


class CharBiLSTMTokenizer:
    def __init__(self, idx2word, word2idx, max_len=10000) -> None:
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.max_len = max_len

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

    def pad(self, ids, l):
        if len(ids) >= l:
            return ids[:l]
        return ids + [self.word2idx[self.PAD]] * (l - len(ids))

    def pad_labels(self, labels, l):
        if len(labels) >= l:
            return labels[:l]
        return labels + [0] * (l - len(labels))

    def pad_words(self, words, l):
        if len(words) >= l:
            return words[:l]
        return words + [0] * (l - len(words))

    def _get_attention(self, padded_ids):
        attention = []
        for idx, _id in enumerate(padded_ids):
            if _id == self.word2idx[self.PAD]:
                attention.append(0)
            else:
                attention.append(1)
        return attention

    def tokenize(self, texts: List[str], labels: List[int] = None):
        _ids = []
        _labels = []
        _words = []
        for text_idx, text in enumerate(texts):
            _id = []
            _label = []
            _word = []
            boundary = labels[text_idx]
            spl = text.split(" ")
            for idx, word in enumerate(spl):
                chars = [c for c in word]
                if len(chars) == 0:
                    chars = [" "]
                for char in chars:
                    _id.append(self.word2idx.get(
                        self.map_char(char), self.word2idx[self.UNK]))
                    _word.append(idx)
                    if boundary == -1 or idx < boundary:
                        # -1 -> imported from task A, fully
                        # human text
                        _label.append(0)
                    else:
                        _label.append(1)

            _ids.append(_id)
            _labels.append(_label)
            _words.append(_word)

        longest = max([len(_id) for _id in _ids])
        if self.max_len is not None:
            longest = min(longest, self.max_len)

        _attentions = []
        for i in range(len(_ids)):
            _ids[i] = self.pad(_ids[i], longest)
            _labels[i] = self.pad_labels(_labels[i], longest)
            _words[i] = self.pad_words(_words[i], longest)
            _attentions.append(self._get_attention(_ids[i]))

        return torch.tensor(_ids), torch.tensor(_labels), torch.tensor(_words), torch.tensor(_attentions)

    @staticmethod
    def collate_fn(tokenizer):
        def collate_batch(batch):
            texts = [x[0] for x in batch]
            _labels = [x[1] for x in batch]
            input_ids, labels, words, attentions = tokenizer.tokenize(
                texts, _labels)

            return input_ids, labels, words, attentions

        return collate_batch

    @classmethod
    def from_pretrained(cls, path, max_len=10000):
        with open(path, "rb") as f:
            word2idx, idx2word, _ = pickle.load(f)
        return cls(idx2word, word2idx, max_len=max_len)
