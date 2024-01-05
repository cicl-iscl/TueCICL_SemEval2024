import pickle
import re
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.device import get_device

device = get_device()


class Word2Vec_Labeller(nn.Module):
    def __init__(
        self,
        pretrained_embeddings=None,
        hidden_size=1024,
        num_layers=2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.emb.cpu()
        self.emb.weight.requires_grad = False
        self.input_size = self.emb.weight.shape[1]

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm.to(device)
        # multiply by 2 because of bidirectional
        self.classifier_head = nn.Linear(self.hidden_size * 2, 2)
        self.classifier_head.to(device)

    def forward(self, input_ids):
        inputs: torch.Tensor = self.emb(input_ids)
        inputs = inputs.to(device)
        outputs, _ = self.lstm(inputs)
        logits = self.classifier_head(outputs)
        predicted = F.log_softmax(logits, dim=-1)
        return predicted

    def predict(self, input_ids) -> List[int]:
        predicted = self.forward(input_ids)
        predicted = torch.argmax(predicted, dim=-1)
        p = predicted.cpu().numpy().tolist()
        r = []
        for item in p:
            try:
                r.append(item.index(1))
            except ValueError:
                r.append(len(item) // 2)
        return r

    def save(self, path, extra={}):
        cp = {
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            **extra
        }
        torch.save(cp, path)

    @classmethod
    def from_pretrained(cls, path):
        cp = torch.load(path)
        model = cls(
            hidden_size=cp["hidden_size"],
            num_layers=cp["num_layers"]
        )
        model.load_state_dict(cp["state_dict"])
        return model


class Word2Vec_Tokenizer:
    def __init__(self) -> None:
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.WHITESPACE = "<WS>"

        with open("wiki2vec/dump_t.pkl", "rb") as f:
            self.word2idx, self.idx2word, self.embeddings = pickle.load(f)

        self.word2idx[self.WHITESPACE] = len(self.word2idx)
        self.idx2word[self.word2idx[self.WHITESPACE]] = self.WHITESPACE
        dim = self.embeddings.shape[1]
        ws_emb = torch.randn(1, dim)
        self.embeddings = torch.cat([self.embeddings, ws_emb], dim=0)

    def _clean(self, word):
        # all non alphanumeric
        replace = re.compile(r"[^a-zA-Z0-9\-]")
        word = replace.sub("", word)
        return word.lower().strip()

    def _is_whitespace(self, word):
        pat = re.compile(r"^\s*$")
        return pat.match(word) is not None

    def tokenize(self, texts: List[str], max_len: int = 1000) -> torch.Tensor:
        tokens = [text.split(" ") for text in texts]
        tokens = [[self._clean(word) for word in text] for text in tokens]
        input_ids = []
        for text in tokens:
            encoded = []
            for token in text:
                if self._is_whitespace(token):
                    encoded.append(self.word2idx[self.WHITESPACE])
                else:
                    encoded.append(self.word2idx.get(
                        token, self.word2idx[self.UNK]))
            input_ids.append(encoded)

        longest = max([len(text) for text in input_ids])
        if longest > max_len:
            longest = max_len
        input_ids = [text[:longest] for text in input_ids]
        input_ids = [text + [self.word2idx[self.PAD]] *
                     (longest - len(text)) for text in input_ids]

        return torch.tensor(input_ids)
