import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from util.device import get_device


class Word2VecBiLSTM(nn.Module):
    def __init__(
        self,
        pretrained_embeddings=None,
        vocab_size=None,
        emb_size=None,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if pretrained_embeddings is not None:
            self.vocab_size = pretrained_embeddings.shape[0]
            self.emb_size = pretrained_embeddings.shape[1]
            self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.vocab_size = vocab_size
            self.emb_size = emb_size
            self.emb = nn.Embedding(self.vocab_size, self.emb_size)

        self.lstm = nn.LSTM(
            hidden_size=self.hidden_size,
            input_size=self.emb_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.lstm2out = nn.Linear(hidden_size * 2, 2)

        self.emb.cpu()
        self.lstm.to(get_device())
        self.lstm2out.to(get_device())

    def __str__(self):
        return f"Word2VecBiLSTM(vocab_size={self.vocab_size}, emb_size={self.emb_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"

    def forward(self, X):
        embedded = self.emb(X)
        inputs = embedded.to(get_device())
        lstm_out, _ = self.lstm(inputs)
        out = self.lstm2out(lstm_out)
        out = F.log_softmax(out, dim=-1)
        return out

    def save(self, path, extra={}):
        cp = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "emb_size": self.emb_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            **extra
        }
        torch.save(cp, path)

    def from_pretrained(cls, path):
        cp = torch.load(path)
        model = cls(
            vocab_size=cp["vocab_size"],
            emb_size=cp["emb_size"],
            hidden_size=cp["hidden_size"],
            num_layers=cp["num_layers"],
            dropout=cp["dropout"],
        )
        model.load_state_dict(cp["state_dict"])
        return model, cp


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
        if len (subwords) == 0:
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

    def tokenize(self, texts, label_bounds):
        _ids = []
        _words = []
        _attentions = []
        _labels = []

        texts = self._split(texts)

        for text_idx, text in enumerate(texts):
            label_bound = label_bounds[text_idx]
            text_ids = []
            text_words = []
            text_labels = []
            for word_idx, word in enumerate(text):
                subwords = self._get_subwords(word)
                for subword in subwords:
                    text_ids.append(self._get_id(subword))
                    text_words.append(word_idx)
                    if label_bound == -1 or word_idx < label_bound:
                        # -1 -> imported from task A, fully
                        # human text
                        text_labels.append(0)
                    else:
                        text_labels.append(1)
            _ids.append(text_ids)
            _words.append(text_words)
            _labels.append(text_labels)

        longest = max([len(x) for x in _ids])
        if longest > self.max_len:
            longest = self.max_len

        for i in range(len(_ids)):
            _ids[i] = _ids[i][:longest]
            _words[i] = _words[i][:longest]
            _labels[i] = _labels[i][:longest]
            l = len(_ids[i])
            _attentions.append([1] * l)
            if l < longest:
                _ids[i] += [self.word2idx[self.PAD]] * (longest - l)
                _words[i] += [-1] * (longest - l)
                _labels[i] += [0] * (longest - l)
                _attentions[i] += [0] * (longest - l)
        return (
            torch.tensor(_ids),
            torch.tensor(_labels),
            torch.tensor(_words),
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
    def collate_fn(tokenizer, check_label_mismatch=False):
        def collate_batch(batch):
            texts = [x[0] for x in batch]
            true_labels = [x[1] for x in batch]
            input_ids, labels, words, attentions = tokenizer.tokenize(
                texts, true_labels)

            if check_label_mismatch:
                for i in range(len(texts)):
                    mapped_label = labels[i].tolist().index(1)
                    mapped_label = words[i].tolist()[mapped_label]
                    try:
                        assert true_labels[i] == mapped_label
                    except:
                        print("Detected label mismatch", mapped_label, true_labels[i])
                        print(labels[i].tolist())
            return input_ids, labels, words, attentions

        return collate_batch
