import torch
import torch.nn as nn

from util.device import get_device
from sklearn.metrics import classification_report, precision_recall_fscore_support
from loader.data import TaskA_Dataset
from torch.utils.data import DataLoader


class SpacyFeaturesMLP(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super(SpacyFeaturesMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_output_features)
        )

    def forward(self, x):
        y_pred = torch.sigmoid(self.mlp(x))
        return y_pred

    def to_device(self):
        self.mlp.to(get_device())

    @classmethod
    def from_pretrained(cls, path):
        print(path)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model = cls(66, 1)
        model.load_state_dict(state_dict)
        model.to_device()
        return model


def collate_fn(batch):
    texts, labels, ids, spacy_feats = zip(*batch)
    texts = list(texts)
    labels = torch.tensor(labels, dtype=torch.long)
    ids = list(ids)
    spacy_feats = torch.tensor(spacy_feats, dtype=torch.float32)
    return spacy_feats, labels, ids


def evaluate(model: SpacyFeaturesMLP, dev_loader, f1_only=True):
    model.eval()
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for spacy_feats, labels, _ in dev_loader:
            spacy_feats = spacy_feats.to(get_device())
            out = model(spacy_feats)
            pred = out.argmax(dim=1)
            for i in range(pred.shape[0]):
                y_pred.append(pred[i].item())
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_gold, y_pred, average="macro", zero_division=0.0)

    if f1_only:
        return f1

    return f1, r


def demo():
    ds = TaskA_Dataset(split="train", return_spacy=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model_path = "../data/pretrained/spacy_feats_mlp.pt"
    model = SpacyFeaturesMLP.from_pretrained(model_path)
    _, report = evaluate(model, dl, f1_only=False)
    print(report)
