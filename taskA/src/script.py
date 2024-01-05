from argparse import ArgumentParser
import numpy as np

import torch
from trainers.train_charLM import entry as train_charlm

torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["train_charlm"])
    args = parser.parse_args()

    if args.command == "train_charlm":
        train_charlm(args)
