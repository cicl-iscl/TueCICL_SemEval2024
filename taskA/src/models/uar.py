from typing import Any
from transformers import AutoModel, AutoTokenizer


class UAR:
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD")
        self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")

    def __call__(self, texts) -> Any:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.logits
