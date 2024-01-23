import pandas as pd
import torch
from torch.utils.data import Dataset
from util.core import abspath

from util.device import get_device
from loader.spacy import SpacyFeatures


class TaskA_Dataset(Dataset):
    def __init__(self, split="train", return_spacy=False) -> None:
        self.split = split
        if split == "train":
            p = abspath(
                __file__, "../../data/subtaskA_train_monolingual.jsonl")
            self.data = pd.read_json(
                p, lines=True)
        else:
            p = abspath(__file__, "../../data/subtaskA_dev_monolingual.jsonl")
            self.data = pd.read_json(
                p, lines=True)

        self.return_spacy = return_spacy
        if return_spacy:
            if self.split == "train":
                self.spacy_ds = SpacyFeatures(split=split)
                self.spacy_ds.scale(*self.spacy_ds.get_scaling_parameters())
            else:
                self.spacy_ds = SpacyFeatures(split=split)
                spacy_ds_train = SpacyFeatures(split="train")
                self.spacy_ds.scale(*spacy_ds_train.get_scaling_parameters())
            print(len(self.spacy_ds), len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text, label, _id = item["text"], item["label"], item["id"]
        if self.return_spacy:
            spacy_feats, _, _ = self.spacy_ds[index]
            return text, label, _id, spacy_feats
        return text, label, _id


def collate_fn(tokenizer, max_len=None, device=get_device()):
    def collate(batch):
        texts = [text for text, _, _ in batch]
        labels = [label for _, label, _ in batch]
        input_ids, attentions = tokenizer.tokenize(
            texts, max_len=max_len, device=device)
        labels_tensor = torch.tensor(
            labels, dtype=torch.long, device=device)
        return input_ids, attentions, labels_tensor
    return collate
