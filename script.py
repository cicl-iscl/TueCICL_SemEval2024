from argparse import ArgumentParser
import numpy as np

import torch
from src.trainers.word2vec_labeller import entry as train_word2vec_labeller
from src.trainers.char_labeller import entry as train_char_labeller

torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["train_word2vec", "train_char"])
    args = parser.parse_args()

    if args.command == "train_word2vec":
        train_word2vec_labeller()
    if args.command == "train_char":
        train_char_labeller()
