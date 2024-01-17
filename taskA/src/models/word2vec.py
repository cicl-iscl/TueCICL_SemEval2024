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
            raise ValueError("Either pretrained_embeddings or vocab_size and emb_size must be provided")    
        
        self.lstm = nn.LSTM(
            self.emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm2class = nn.Linear(hidden_size, 2)

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
        return f"Word2VecClassifier(emb_size={self.emb_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout})"

    @classmethod
    def from_pretrained(cls, path):
        data = torch.load(path)
        model = cls(
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            dropout=data["dropout"],
            vocab_size=data["vocab_size"],
            emb_size=data["emb_size"],
        )
        model.load_state_dict(data["model"])
        return model


class Word2VecTokenizer:
    UNK = "<UNK>"
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    WHITESPACE = "<WS>"
    PUNCTUATION = "<PUNCT>"

    def __init__(self, idx2word, word2idx, weights=None) -> None:
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.weights = weights

    def save(self, weights_path=None, vocab_path=None):
        with open(weights_path, "wb") as f:
            pickle.dump(self.weights, f)
        with open(vocab_path, "wb") as f:
            pickle.dump((self.idx2word, self.word2idx), f)
            
    def _is_space(self, token):
        pat = re.compile(r"\s+")
        return pat.match(token) is not None

    def _get_ids(self, tokens):
        _ids = []
        def idof(token): return self.word2idx.get(
            token, self.word2idx[self.UNK])
        for token in tokens:
            if self._is_space(token):
                _ids.append(idof(self.WHITESPACE))
            elif token in string.punctuation:
                _ids.append(idof(self.PUNCTUATION))
            else:
                _ids.append(idof(token))
        return _ids

    def tokenize(self, texts, add_special_tokens=True, max_len=None, device=get_device()):
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
            attentions.append([1] * len(text))
            if len(text) < longest:
                attentions[i] = attentions[i] + [0] * (longest - len(text))
                tokens[i] += [self.PAD] * (longest - len(text))
        
        ids = [self._get_ids(x) for x in tokens]
        return (
            torch.tensor(ids, dtype=torch.long, device=device), 
            torch.tensor(attentions, dtype=torch.long, device=device)
        )

    def extend(self, texts):
        for text in texts:
            spl = text.lower().split()
            for token in spl:
                if token not in self.word2idx:
                    self.weights.append([0.0] * self.emb_size)
                    self.word2idx[token] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = token

    @classmethod
    def from_txt(cls, path, emb_size=500):
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
        
        weights = torch.tensor(weights, dtype=torch.float32)

        o = cls(idx2word, word2idx, weights)
        
        return o

    @classmethod
    def from_pkl(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            idx2word, word2idx = data
        return cls(idx2word, word2idx)
