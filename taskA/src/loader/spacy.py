import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class SpacyFeatures:
    def __init__(self, train_path, dev_path) -> None:
        self.train_ids, self.train_vectors = self.__load_data(
            train_path)
        self.dev_ids, self.dev_vectors = self.__load_data(
            dev_path)

    def __load_data(self, path):
        _ids, vectors = {}, []
        with open(path, 'r') as f:
            i = 0
            for line in f:
                try:
                    content = json.loads(line)
                    vec = content["vector"]
                    vec = [0.0 if not x else x for x in vec]
                    _id = content["id"]
                    _ids[_id] = i
                    vectors.append(
                        torch.tensor(vec, dtype=torch.float32)
                    )
                    i += 1
                except:
                    print(line)
        return _ids, torch.stack(vectors)

    def scale(self):
        mean = self.train_vectors.mean(dim=0)
        std = self.train_vectors.std(dim=0)
        self.train_vectors = (self.train_vectors - mean) / std
        self.dev_vectors = (self.dev_vectors - mean) / std

    def get(self, text_id, split="train"):
        if split == "train":
            return self.train_vectors[self.train_ids[text_id]]
        else:
            return self.dev_vectors[self.dev_ids[text_id]]
