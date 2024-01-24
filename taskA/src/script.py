from argparse import ArgumentParser
import numpy as np

import torch
from trainers.train_charLM import (
    entry as train_charlm, add_args as add_charlm_args)
from trainers.char_classifier import (
    entry as train_char_classifier, add_args as add_char_classifier_args)

from trainers.train_word2vec_classifier import (
    entry as train_word2vec_classifier, add_args as add_word2vec_classifier_args)

from trainers.train_lang_mlp import (
    entry as train_lang_mlp, add_args as add_lang_mlp_args)


torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    parser = ArgumentParser()
    choices = ["train_charlm", "train_char_classifier",
               "train_word2vec_classifier", "train_lang_mlp"]
    parser.add_argument("command", help="Command to execute",
                        choices=choices)
    add_charlm_args(parser)
    add_char_classifier_args(parser)
    add_word2vec_classifier_args(parser)
    add_lang_mlp_args(parser)
    args = parser.parse_args()

    if args.command == "train_charlm":
        train_charlm(args)
    elif args.command == "train_char_classifier":
        train_char_classifier(args)
    elif args.command == "train_word2vec_classifier":
        train_word2vec_classifier(args)
    elif args.command == "train_lang_mlp":
        train_lang_mlp(args)
