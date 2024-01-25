import torch
import torch.nn as nn
import torch.nn.functional as F
from models.char_classifier import CharClassifier, CharClassifierTokenizer
from models.word2vec import Word2VecClassifier, Word2VecTokenizer
from util.device import get_device


class JointModel(nn.Module):
    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        dropout=None
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

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

    def forward(self, X):
        return self.mlp(X)

    def save(self, path, extra={}):
        torch.save({
            "state_dict": self.state_dict(),
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
        w2v_max_len=1000
    ):
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
        print("------------------\n")

    def prepare(self, texts):
        with torch.no_grad():
            cc_X, _ = self.cc_tokenizer.tokenize(
                texts, max_len=self.cc_max_len)
            w2v_X, _ = self.w2v_tokenizer.tokenize(texts)
            cc_out = self.cc_classifier(cc_X)[1][:, -1, :]
            w2v_out = self.w2v_classifier(w2v_X)[1][:, -1, :]
            joint_out = torch.cat([cc_out, w2v_out], dim=1)
            return joint_out

    @staticmethod
    def collate_fn(tokenizer):
        def collate(batch):
            text = [i[0] for i in batch]
            label = [i[1] for i in batch]
            input_ids = tokenizer.prepare(text)
            labels_tensor = torch.tensor(
                label, dtype=torch.float32)
            return input_ids, labels_tensor
        return collate
