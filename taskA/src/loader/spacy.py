import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class SpacyFeatures(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        path_train = "~/cicl/taskA/data/subtaskA_spacy_feats.json"
        path_dev = "~/cicl/taskA/data/subtaskA_test_spacy_feats.json"
        data_path = path_train if split == "train" else path_dev
        self.data = pd.read_json(data_path)
        self.data.drop(["passed_quality_check", "oov_ratio",
                       "n_characters", "n_sentences"], inplace=True, axis=1)
        self.data.dropna(inplace=True)
        self.feats = np.array(self.data.drop(
            ["label", "id", "text"], axis=1).values)

    def get_scaling_parameters(self):
        means = np.mean(self.feats, axis=0)
        sd = np.std(self.feats, axis=0)
        return means, sd

    def scale(self, means: np.ndarray = None, sd: np.ndarray = None):
        means = np.tile(means, (np.shape(self.feats)[0], 1))
        sd = np.tile(sd, (np.shape(self.feats)[0], 1))
        self.feats = (self.feats - means) / sd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label, id = item["label"], item["id"]
        return self.feats[index], label, id
