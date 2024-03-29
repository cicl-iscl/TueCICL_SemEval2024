import spacy
import textdescriptives as td
import pandas as pd
import torch
from tqdm import tqdm


class DS(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        if mode == "train":
            self.data = pd.read_json(
                '~/cicl/taskA/data/task_files/subtaskA_dev_monolingual.jsonl', lines=True)
        elif mode == "dev":
            self.data = pd.read_json(
                '~/cicl/taskA/data/task_files/subtaskA_train_monolingual.jsonl', lines=True)
        elif mode == "test":
            self.data = pd.read_json(
                '~/cicl/taskA/data/task_files/subtaskA_test_monolingual.jsonl', lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text, id = item["text"], item["id"]
        return id, text


def entry(args):
    mode = "test"
    ds = DS(mode=mode)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    nlp = spacy.load('en_core_web_sm')
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

            sorted_keys = sorted(vec.keys())

            vec = [vec[k] for k in sorted_keys]

            r = {
                "vector": vec,
                "id": _id
            }
            result.append(r)
    if mode == "train":
        out_path = '~/cicl/taskA/data/spacy/spacy_feats_sm_train.jsonl'
    elif mode == "dev":
        out_path = '~/cicl/taskA/data/spacy/spacy_feats_sm_dev.jsonl'
    elif mode == "test":
        out_path = '~/cicl/taskA/data/spacy/spacy_feats_sm_test.jsonl'
    pd.DataFrame(result).to_json(out_path, lines=True, orient='records')

# docs = nlp.pipe(train_df.text)
# feats_df = td.extract_df(docs)
# feats_df["log_ppl"] = train_df["log_ppl"]
# feats_df["id"] = train_df["id"]
# feats_df["label"] = train_df["label"]
# feats_df.to_json('~/cicl/taskA/data/subtaskA_train_spacy_feats.json')
# x = 0
