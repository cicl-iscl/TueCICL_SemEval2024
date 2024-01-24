import torch
import torch.nn as nn
import torch.nn.functional as F
from models.char_classifier import CharClassifier, CharClassifierTokenizer
from models.word2vec import Word2VecClassifier, Word2VecTokenizer


class JointModel(nn.Module):
    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        dropout=None
    ) -> None:
        super().__init__()


class JointModelPreprocessor:
    def __init__(
        self,
        cc_model_path=None,
        cc_tokenizer_path=None,
        w2v_model_path=None,
        w2v_tokenizer_path=None,
    ):
        self.cc_classifier = CharClassifier.from_pretrained(cc_model_path)
        self.cc_tokenizer = CharClassifierTokenizer.from_pretrained(
            cc_tokenizer_path)
        self.w2v_classifier = Word2VecClassifier.from_pretrained(
            w2v_model_path)
        self.w2v_tokenizer = Word2VecTokenizer.from_pkl(
            w2v_tokenizer_path)
