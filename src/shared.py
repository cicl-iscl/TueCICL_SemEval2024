import json
from typing import Dict, List
from .shared_types import *
import numpy as np


def load_data(split="train") -> List[TaskThreeEntry]:
    p_dev = "/Users/aron/codebase/uniwork/cicl/src/data/subtaskC_dev.jsonl"
    p_train = "/Users/aron/codebase/uniwork/cicl/src/data/subtaskC_train.jsonl"
    p = p_dev if split == "dev" else p_train
    data = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize(data: List[TaskThreeEntry]) -> List[TaskThreeEntry]:
    for entry in data:
        entry["tokens"] = entry["text"].strip().lower().split()
    return data


def get_vocab(data: List[TaskThreeEntry]) -> Dict[str, int]:
    vocab = {}
    for entry in data:
        for token in entry["tokens"]:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def onehot_encode(tokens: List[str], vocab: Dict[str, int]) -> List[TaskThreeEntry]:
    oh = []
    for token in tokens:
        encoded = np.zeros(len(vocab) + 1)
        unk_idx = len(vocab)
        if token in vocab:
            encoded[vocab[token]] = 1
        else:
            encoded[unk_idx] = 1
        oh.append(encoded)
    return np.array(oh)
