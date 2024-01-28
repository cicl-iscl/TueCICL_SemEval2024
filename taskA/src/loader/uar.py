from typing import List
import torch
from transformers import AutoModel, AutoTokenizer


class UAR:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "rrivera1849/LUAR-CRUD", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            "rrivera1849/LUAR-CRUD", trust_remote_code=True)

    def _get(self, text: str):
        text = [
            text.split()
        ]
        batch_size = 1
        episode_length = len(text[0])
        text = [j for i in text for j in i]
        tokenized_text = self.tokenizer(
            text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(
            batch_size, episode_length, -1)
        tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(
            batch_size, episode_length, -1)
        with torch.no_grad():
            out = self.model(**tokenized_text)
            return out[0]

    def get(self, texts: List[str]):
        tensors = [self._get(text) for text in texts]
        return torch.stack(tensors, dim=0)
