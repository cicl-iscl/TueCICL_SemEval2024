from typing import TypedDict
import numpy as np


class TaskThreeEntry(TypedDict):
    id: str
    text: str
    label: int
    tokens: list[str]
    onehot: np.ndarray
