import torch
import torch.nn as nn
from loader.spacy import SpacyFeatures
import torch.nn.functional as F

from util.device import get_device


class SpacyFeaturesMLP(nn.Module):
    def __init__(
        self,
        n_input_features,
        hidden_size=128,
        dropout=0.0,
        n_output_features=1
    ):
        super(SpacyFeaturesMLP, self).__init__()
        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(n_input_features, self.hidden_size),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, n_output_features)
        )

    def forward(self, x):
        out = self.mlp(x)
        y_pred = F.sigmoid(out)
        return y_pred

    def to_device(self):
        self.mlp.to(get_device())

    def save(self, path, extra={}):
        cp = {
            "state_dict": self.state_dict(),
            "n_input_features": self.n_input_features,
            "n_output_features": self.n_output_features,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            **extra
        }
        torch.save(cp, path)

    @classmethod
    def from_pretrained(cls, path):
        cp = torch.load(path, map_location=torch.device('cpu'))
        model = cls(
            cp["n_input_features"],
            hidden_size=cp["hidden_size"],
            dropout=cp["dropout"],
            n_output_features=cp["n_output_features"],
        )
        model.load_state_dict(cp["state_dict"])
        model.to_device()
        return model, cp

    def __str__(self):
        return f"SpacyFeaturesMLP({self.n_input_features}, {self.n_output_features}, {self.hidden_size}, {self.dropout})"


def collate_fn(batch):
    texts, labels, ids, spacy_feats = zip(*batch)
    texts = list(texts)
    labels = torch.tensor(labels, dtype=torch.float32)
    ids = list(ids)
    spacy_feats = torch.stack(spacy_feats)
    return spacy_feats, labels, ids
