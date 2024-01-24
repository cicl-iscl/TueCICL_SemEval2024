import spacy
import textdescriptives as td
import pandas as pd
import torch
from tqdm import tqdm

class DS(torch.utils.data.Dataset):
    def __init__(self, dev=False):
        if dev:
            self.data = pd.read_json('~/cicl/taskA/data/subtaskA_dev_monolingual.jsonl', lines=True)
        else:
            self.data = pd.read_json('~/cicl/taskA/data/subtaskA_train_monolingual.jsonl', lines=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text, id = item["text"], item["id"]
        return id, text
    


def entry(args):
    ds = DS(dev=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('textdescriptives/all')
    result = []
    for id, text in tqdm(dl):
        docs = nlp.pipe(text)
        feats_df = td.extract_df(docs)
        feats_df["id"] = id
        for i in range(len(feats_df)):
            vec = feats_df.iloc[i].to_dict()
            _id = vec.pop('id')
            vec.pop('text')
            vec.pop('passed_quality_check')
            vec.pop('oov_ratio')
            vec.pop('n_characters')
            
            vec = [vec[k] for k in vec]
            
            r = {
                "vector": vec,
                "id": _id
            }
            result.append(r)
    out_path = '~/cicl/taskA/data/subtaskA_dev_spacy_feats.jsonl'
    pd.DataFrame(result).to_json(out_path, lines=True, orient='records')

# docs = nlp.pipe(train_df.text)
# feats_df = td.extract_df(docs)
# feats_df["log_ppl"] = train_df["log_ppl"]
# feats_df["id"] = train_df["id"]
# feats_df["label"] = train_df["label"]
# feats_df.to_json('~/cicl/taskA/data/subtaskA_train_spacy_feats.json')
# x = 0
