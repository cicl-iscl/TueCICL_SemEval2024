import pandas as pd
import spacy
import lftk
from load_data import get_data
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

train = pd.read_json("../../data/subtaskA_train_ppl.json")

doc = nlp(train.text.iloc[1])
LFTK = lftk.Extractor(docs=doc)
feat_names = lftk.search_features(return_format="list_key")
feats_df = pd.DataFrame(columns=feat_names)
for i in tqdm(range(119757)):
    doc = nlp(train.text.iloc[i])
    LFTK = lftk.Extractor(docs = doc)
    extracted_features_train = LFTK.extract(features=feat_names)
    batch = pd.Series(extracted_features_train)
    feats_df.loc[i] = batch
feats_df["log_ppl"] = train["log_ppl"]
feats_df["id"] = train["id"]
feats_df["label"] = train["label"]
feats_df.to_json("data/subtaskA_train_spacy_feats_large.json")
