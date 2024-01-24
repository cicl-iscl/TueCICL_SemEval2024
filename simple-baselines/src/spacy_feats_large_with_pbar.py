import pandas as pd
import spacy
import lftk
from load_data import get_data
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

train = pd.read_json("../../data/subtaskA_train_ppl.json")
feats_df = pd.DataFrame()
for i in tqdm(range(1198)):
    docs = [nlp(doc) for doc in train.text.iloc[i:(i+1)*10]]
    LFTK = lftk.Extractor(docs = docs)
    feat_names = lftk.search_features(return_format="list_key")
    extracted_features_train = LFTK.extract(features=feat_names)
    batch = pd.DataFrame.from_dict(extracted_features_train)
    pd.concat([feats_df, batch], ignore_index=True, axis=0)
feats_df["log_ppl"] = train["log_ppl"]
feats_df["id"] = train["id"]
feats_df["label"] = train["label"]
feats_df.to_json("data/subtaskA_train_spacy_feats_large.json")
