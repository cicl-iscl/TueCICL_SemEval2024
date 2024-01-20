from argparse import ArgumentParser
from trainers.train_cc_labeller import entry as train_cc_labeller, add_args as add_args_cc_labeller

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", help="Command to execute",
                        choices=["finetune_cc"])
    add_args_cc_labeller(parser)

    args = parser.parse_args()

    if args.command == "finetune_cc":
        train_cc_labeller(args)
