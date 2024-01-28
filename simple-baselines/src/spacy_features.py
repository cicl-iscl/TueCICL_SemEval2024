import spacy
import textdescriptives as td
import pandas as pd

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('textdescriptives/all')
train_df = pd.read_json('../../data/subtaskA_train_spacy_feats_large9.json')
docs = nlp.pipe(train_df.iloc[1:4].text)
feats_df = td.extract_df(docs)
feats_df["log_ppl"] = train_df["log_ppl"]
feats_df["id"] = train_df["id"]
feats_df["label"] = train_df["label"]
feats_df.to_json('../../data/subtaskA_spacy_feats_two_sample.json', orient="records")
x = 0