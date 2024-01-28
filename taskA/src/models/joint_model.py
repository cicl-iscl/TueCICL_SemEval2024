import torch
import torch.nn as nn
import torch.nn.functional as F
from loader.uar import UAR
from models.char_classifier import CharClassifier, CharClassifierTokenizer
from models.word2vec import Word2VecClassifier, Word2VecTokenizer
from util.device import get_device


class JointModel(nn.Module):
    def __init__(
        self,
        cc_size=None,
        w2v_size=None,
        uar_size=None,
        hidden_size=128,
        dropout=None
    ) -> None:
        super().__init__()
        self.cc_size = cc_size
        self.w2v_size = w2v_size
        self.uar_size = uar_size
        self.input_size = cc_size + w2v_size + uar_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.cc_weight = nn.Parameter(torch.tensor(
            1.0, requires_grad=True, dtype=torch.float32))
        self.w2v_weight = nn.Parameter(torch.tensor(
            1.0, requires_grad=True, dtype=torch.float32))
        self.uar_weight = nn.Parameter(torch.tensor(
            1.0, requires_grad=True, dtype=torch.float32))

        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def to_device(self):
        self.mlp.to(get_device())

    def forward(self, X_cc, X_w2v, X_uar):
        cc_out = X_cc * self.cc_weight
        w2v_out = X_w2v * self.w2v_weight
        uar_out = X_uar * self.uar_weight
        X = torch.cat([cc_out, w2v_out, uar_out], dim=1)
        return self.mlp(X)

    def save(self, path, extra={}):
        torch.save({
            "state_dict": self.state_dict(),
            "cc_size": self.cc_size,
            "w2v_size": self.w2v_size,
            "uar_size": self.uar_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            **extra
        }, path)

    def __str__(self):
        return f"JointModel(input_size={self.input_size}, hidden_size={self.hidden_size}, dropout={self.dropout})"

    @classmethod
    def from_pretrained(cls, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            dropout=checkpoint["dropout"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to_device()
        return model, checkpoint


class JointModelPreprocessor:
    def __init__(
        self,
        cc_model_path=None,
        cc_tokenizer_path=None,
        w2v_model_path=None,
        w2v_tokenizer_path=None,
        cc_max_len=5000,
        w2v_max_len=1000,
        uar=None
    ):
        self.uar: UAR = uar

        self.cc_max_len = cc_max_len
        self.w2v_max_len = w2v_max_len
        self.cc_tokenizer = CharClassifierTokenizer.from_pretrained(
            cc_tokenizer_path)
        self.cc_classifier = CharClassifier.from_pretrained(cc_model_path)
        self.cc_classifier.to_device()

        self.w2v_tokenizer = Word2VecTokenizer.from_pretrained(
            w2v_tokenizer_path, max_len=w2v_max_len)
        self.w2v_classifier, _ = Word2VecClassifier.from_pretrained(
            w2v_model_path)
        self.w2v_classifier.to_device()

        self.input_size = self.cc_classifier.hidden_size + \
            self.w2v_classifier.hidden_size

        print("\n------------------")
        print("Preprocessor models:")
        print(self.cc_classifier)
        print(self.w2v_classifier)
        print(self.uar)
        print("------------------\n")

    def prepare(self, texts):
        with torch.no_grad():
            cc_X, _ = self.cc_tokenizer.tokenize(
                texts, max_len=self.cc_max_len)
            w2v_X, w2v_attention = self.w2v_tokenizer.tokenize(texts)
            _, cc_out = self.cc_classifier(cc_X)
            cc_out = cc_out[:, -1, :]
            _, _, w2v_out = self.w2v_classifier(
                w2v_X, w2v_attention, return_last_hidden=True)
            uar_out = self.uar.get(texts)
            return cc_out, w2v_out, uar_out

    @staticmethod
    def collate_fn(tokenizer):
        def collate(batch):
            text = [i[0] for i in batch]
            label = [i[1] for i in batch]
            cc_out, w2v_out, uar_out = tokenizer.prepare(text)
            labels_tensor = torch.tensor(
                label, dtype=torch.float32)
            return cc_out, w2v_out, uar_out, labels_tensor
        return collate
