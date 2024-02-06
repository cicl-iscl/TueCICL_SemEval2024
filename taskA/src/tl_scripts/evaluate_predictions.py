from argparse import Namespace, ArgumentParser
import json

from sklearn.metrics import classification_report


def add_args(parser: ArgumentParser):
    def p(name):
        return '--evaluate-predictions-' + name
    group = parser.add_argument_group('Prediction evaluation')
    group.add_argument(p("pred-file"), type=str, default=None)
    group.add_argument(p("gold-file"), type=str, default=None)


def entry(args: Namespace):
    def arg(name):
        return getattr(args, 'evaluate_predictions_' + name.replace('-', '_'))

    data = {}

    with open(arg("pred-file"), "r") as f:
        for line in f:
            if line:
                content = json.loads(line)
                data[content['id']] = {
                    "pred": content['label']
                }

    with open(arg("gold-file"), "r") as f:
        for line in f:
            if line:
                content = json.loads(line)
                data[content['id']]["gold"] = content['label']

    preds = []
    golds = []

    for k, v in data.items():
        preds.append(v["pred"])
        golds.append(v["gold"])

    print(classification_report(golds, preds))
