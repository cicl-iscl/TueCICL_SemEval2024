import spacy
import pandas as pd
import seaborn as sns
from load_data import get_data
import matplotlib.pyplot as plt
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")
train_df, test_df = get_data("../../data/subtaskA_train_monolingual.jsonl", "../../data/subtaskA_dev_monolingual.jsonl", 0)

red_df = pd.concat([train_df.iloc[:1000, :], train_df.iloc[-1000:, :]]).reset_index(drop=True)
red_test_df = pd.concat([test_df.iloc[:1000, :], test_df.iloc[-1000:, :]]).reset_index(drop=True)
train_df_ttr = red_df
ttr = []
for i in tqdm(range(len(red_df))):
    doc = nlp(train_df.text.iloc[i])
    types_dict = {}
    tokens = 0
    types = 0
    for t in doc:
        tokens += 1
        if t.text not in types_dict:
            types_dict[t.text] = 1
            types += 1
        # use lemmata instead of raw tokens
        # if t.lemma_ not in types_dict:
        #     types_dict[t.lemma_] = 1
        #     types += 1
    ttr.append(types/tokens)

train_df_ttr["ttr"] = pd.Series(ttr)
# train_df_ttr = train_df_ttr.pivot(values="label")
train_df_ttr.to_json("../../data/train_ttr.json")
sns.boxplot(data=train_df_ttr, x="label", y="ttr")
plt.show()


