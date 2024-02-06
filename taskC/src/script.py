from argparse import ArgumentParser
from util.device import get_device
import torch
import numpy as np
import random

from trainers.train_cc_labeller import (
    entry as train_cc_labeller, add_args as add_args_cc_labeller)
from trainers.train_char_bilstm import (
    entry as train_char_bilstm, add_args as add_args_char_bilstm)
from trainers.train_word2vec_bilstm import (
    entry as train_word2vec_bilstm, add_args as add_args_word2vec_bilstm)

from trainers.train_joint_model import (
    entry as train_joint_model, add_args as add_args_joint_model)

from tl_scripts.evaluate_predictions import(
    entry as evaluate_predictions, add_args as add_args_evaluate_predictions)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



if __name__ == "__main__":
    device = get_device()
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["finetune_cc", "train_char_bilstm", "train_word2vec_bilstm", "train_joint_model", "evaluate_predictions"])
    add_args_cc_labeller(parser)
    add_args_char_bilstm(parser)
    add_args_word2vec_bilstm(parser)
    add_args_joint_model(parser)
    add_args_evaluate_predictions(parser)

    args = parser.parse_args()

    if args.command == "finetune_cc":
        train_cc_labeller(args)
    elif args.command == "train_char_bilstm":
        train_char_bilstm(args)
    elif args.command == "train_word2vec_bilstm":
        train_word2vec_bilstm(args)
    elif args.command == "train_joint_model":
        train_joint_model(args)
    elif args.command == "evaluate_predictions":
        evaluate_predictions(args)
