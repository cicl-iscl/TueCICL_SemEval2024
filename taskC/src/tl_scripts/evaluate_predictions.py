from argparse import Namespace, ArgumentParser
import json

from sklearn.metrics import classification_report


def add_args(parser: ArgumentParser):
    def p(name):
        return '--evaluate-predictions-' + name
    group = parser.add_argument_group('Prediction evaluation')
    group.add_argument(p("pred-file"), type=str, default=None, action="append")
    group.add_argument(p("gold-file"), type=str, default=None)


def entry(args: Namespace):
    def arg(name):
        return getattr(args, 'evaluate_predictions_' + name.replace('-', '_'))

    
    gold = {}
    
    with open(arg("gold-file"), "r") as f:
        for line in f:
            if line:
                content = json.loads(line)
                gold[content['id']] = content['label']
                
    for pred_file in arg("pred-file"):
        data = {
            _id: {
                "gold": gold[_id]
            } for _id in gold
        }
        
    
        with open(pred_file, "r") as f:
            for line in f:
                if line:
                    content = json.loads(line)
                    data[content['id']]['pred'] = content['label']

    

        distances = []

        for k, v in data.items():
            a = v['gold']
            b = v['pred']
            distances.append(abs(a - b))
        
        print("----------------")
        print(pred_file)
        print("MAE:", sum(distances) / len(distances))
        print("----------------")
            

        
