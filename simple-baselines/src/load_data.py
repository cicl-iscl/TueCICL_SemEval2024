import pandas as pd
def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)

    # train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, test_df

train_df = pd.read_json("../../data/subtaskA_test_spacy_feats_large1.json")
new_df = pd.read_json("../../data/subtaskA_test_spacy_feats_large2.json")
print(train_df.columns.tolist() == new_df.columns.tolist())
# for i in range(2, 11):
#     if i == 5:
#         new_df_1 = pd.read_json(f"../../data/subtaskA_train_spacy_feats_large51.json")
#         new_df_2 = pd.read_json(f"../../data/subtaskA_train_spacy_feats_large52.json")
#         new_df = pd.concat([new_df_1, new_df_2])
#     else:
#         new_df = pd.read_json(f"../../data/subtaskA_train_spacy_feats_large{i}.json")
#     print(train_df.columns.tolist() == new_df.columns.tolist())
#     train_df = pd.concat([train_df, new_df])

# train_df.to_json("../../data/subtaskA_train_spacy_feats_large_all_2.json")



