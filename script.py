from argparse import ArgumentParser
from src.trainers.word2vec_labeller import entry as train_word2vec_labeller
from src.trainers.char_labeller import entry as train_char_labeller


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["train_word2vec", "train_char"])
    args = parser.parse_args()

    if args.command == "train_word2vec":
        train_word2vec_labeller()
    if args.command == "train_char":
        train_char_labeller()
