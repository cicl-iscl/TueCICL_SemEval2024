from argparse import ArgumentParser

from trainers.train_cc_labeller import (
    entry as train_cc_labeller, add_args as add_args_cc_labeller)
from trainers.train_char_bilstm import (
    entry as train_char_bilstm, add_args as add_args_char_bilstm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["finetune_cc", "train_char_bilstm"])
    add_args_cc_labeller(parser)
    add_args_char_bilstm(parser)

    args = parser.parse_args()

    if args.command == "finetune_cc":
        train_cc_labeller(args)
    elif args.command == "train_char_bilstm":
        train_char_bilstm(args)
