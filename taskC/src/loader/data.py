import json
import torch
from torch.utils.data import Dataset

from util.device import get_device
import pandas as pd


class TaskC_Data(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        f = "../data/subtaskC_train.jsonl" if self.split == "train" else "../data/subtaskC_dev.jsonl"
        data = []

        with open(f, "r", encoding="utf-8") as f:
            for line in f:
                parsed = json.loads(line)
                data.append(parsed)

        return data

    def import_task_A(self):
        path = "../data/subtaskA_train_monolingual.jsonl"
        with open(path, "r") as f:
            for line in f:
                if not line:
                    continue
                parsed = json.loads(line)
                obj = {
                    "text": parsed["text"],
                    "label": 0 if parsed["label"] == 1 else -1,
                    "id": parsed["id"]
                }
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["text"], item["label"]


def collate_fn_wordlevel(tokenizer):
    def collate(batch):
        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        input_ids = tokenizer.tokenize(texts)
        labels_tensor = torch.zeros(
            (len(batch), input_ids.shape[1]), dtype=torch.long, device=get_device())
        for i, label in enumerate(labels):
            if label == -1:
                # text is fully human
                continue
            for j in range(label, input_ids.shape[1]):
                labels_tensor[i, j] = 1
        return input_ids, labels_tensor
    return collate


def collate_fn_charlevel(tokenizer, max_len=20_000):
    def collate(batch):
        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        input_ids, attentions, words = tokenizer.tokenize(
            texts, max_len=max_len)

        labels_tensor = torch.zeros(
            (len(batch), input_ids.shape[1]), dtype=torch.long, device=get_device())
        true_labels = torch.zeros(
            (len(batch)), dtype=torch.long, device=get_device())
        for i, label in enumerate(labels):
            true_labels[i] = label
            if label == -1:
                # fully human text
                continue
            label_start = words[i].tolist().index(label)
            for j in range(label_start, input_ids.shape[1]):
                labels_tensor[i, j] = 1
        return input_ids, attentions, words, labels_tensor, true_labels
    return collate
