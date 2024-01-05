import pickle
import string
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch

from ..util.device import get_device


class CharLevelLabeller(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        bidirectional: bool = True
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            hidden_size=self.hidden_size,
            input_size=self.embedding_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        h = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.lstm2pred = nn.Linear(h, self.output_size)

    def forward(self, input_ids):
        embedded = self.emb(input_ids)
        out, _ = self.lstm(embedded)
        pred = self.lstm2pred(out)
        pred = F.log_softmax(pred, dim=-1)
        return pred

    def predict(self, input_ids, words):
        pred = self.forward(input_ids)
        pred = pred.argmax(dim=-1)
        result = []
        for i in range(pred.shape[0]):
            l = pred[i].cpu().tolist()
            # -1 = fully human text
            char = l.index(1) if 1 in l else -1
            # predicted word is the word where the char is located
            # 'words' tracks the word index of each char
            word = words[i].tolist()[char] if char != -1 else -1
            result.append(word)
        return result

    def save(self, path, extra={}):
        save_data = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            **extra
        }
        torch.save(save_data, path)

    @classmethod
    def from_pretrained(cls, path):
        save_data = torch.load(path)
        model = cls(
            vocab_size=save_data["vocab_size"],
            embedding_size=save_data["embedding_size"],
            vocab_size=save_data["vocab_size"],
            hidden_size=save_data["hidden_size"],
            output_size=save_data["output_size"],
            num_layers=save_data["num_layers"],
            bidirectional=save_data["bidirectional"],
        )
        model.load_state_dict(save_data["state_dict"])
        return model


class CharTokenizer:
    def __init__(self, word2idx, idx2word, vocab) -> None:
        self.WHITESPACE = "<WS>"
        self.PUNCTUATION = "<PUNCT>"
        self.DIGIT = "<DIGIT>"
        self.UNK = "<UNK>"
        self.SENT_TERMINATE = "<SENT_TERMINATE>"

        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"

        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab = vocab

    def tokenize(self, texts: List[str], max_len=20_000) -> Tuple[torch.Tensor, torch.Tensor]:
        attentions = []
        input_ids = []
        words = []
        for text in texts:
            _i = []
            _w = []
            w = 0
            chars = [c for c in text.strip().lower()]
            for char in chars:
                if char == " ":
                    w += 1
                _i.append(self.word2idx.get(char, self.word2idx[self.UNK]))
                _w.append(w)
            input_ids.append(_i)
            words.append(_w)
        longest = max([len(text) for text in texts])
        if longest > max_len:
            longest = max_len
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i][:longest]
            words[i] = words[i][:longest]
            input_ids[i] = input_ids[i] + \
                [self.word2idx[self.PAD]] * (longest - len(input_ids[i]))
            attentions.append(
                [1 if _id != self.word2idx[self.PAD] else 0 for _id in input_ids[i]])
            pad_word = max(words[i]) + 1
            words[i] = words[i] + [pad_word] * (longest - len(words[i]))
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=get_device())
        attentions = torch.tensor(
            attentions, dtype=torch.long, device=get_device())
        words = torch.tensor(words, dtype=torch.long, device=get_device())
        return input_ids, attentions, words

    def build(self, texts: List[str] = []):
        vocab = set([self.BOS, self.EOS, self.PAD, self.WHITESPACE,
                     self.PUNCTUATION, self.DIGIT, self.UNK, self.SENT_TERMINATE])
        for text in texts:
            vocab.update([c for c in text.strip().lower()])
        vocab = sorted(vocab)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab = vocab

    def __str__(self) -> str:
        return f"CharTokenizer[ size={len(self.vocab)} ]"

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.word2idx, self.idx2word, self.vocab), f)

    @classmethod
    def from_pretrained(cls, path: str):
        with open(path, "rb") as f:
            word2idx, idx2word, vocab = pickle.load(f)
            return cls(word2idx, idx2word, vocab)
