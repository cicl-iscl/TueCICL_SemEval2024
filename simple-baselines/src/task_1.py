import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, test_df


def preprocess(df):
    df.text.str.replace('\n+', '', regex=True)
    return df

# some test sentences
# X = ['This is the first document.', 'This document is the second document.','And this is the third one.','Is this the first document?']

# load data
train_df, test_df = get_data("../../data/subtaskA_train_monolingual.jsonl", "../../data/subtaskA_dev_monolingual.jsonl", 0)
train_df = preprocess(train_df)
test_df = preprocess(test_df)

# shuffle data because it's sorted
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# create vectorizer to transform documents in sparse tf-idf vector representations
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(4, 6), sublinear_tf=True, max_df=0.7, min_df=50, use_idf=True)
X_train = vectorizer.fit_transform(train_df.text)
X_test = vectorizer.transform(test_df.text)
features = vectorizer.get_feature_names_out()

# create and train classifier
clf = RidgeClassifier(solver="sparse_cg", tol=1e-10)
clf.fit(X_train, train_df.label)

# make predictions on test data
pred = clf.predict(X_test)

# number of machine generated docs
count = pred[pred == 1]
print(len(count))

f1_1 = f1_score(test_df.label, pred)
print(f"f_1_1 score: {f1_1}")





# vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,3), sublinear_tf=True, max_df=0.5, min_df=5)
# X_train = vectorizer.fit_transform(train_df.text)
# X_test = vectorizer.transform(test_df.text)
# clf = RidgeClassifier(solver="sparse_cg", tol=1e-10)
# clf.fit(X_train, train_df.label)
# # pred = clf.predict(X_test)
# # print(pred)
# pred_2 = clf.decision_function(X_test)
# # print(pred)
# # count = pred[pred == 1]
# # print(len(count))
# f1_2 = f1_score(test_df.label, pred_2)
# print(f"f_1_2 score: {f1_2}")
# pred = pred_1 + pred_2
# print(pred)
# print(np.round(pred))
# f1 = f1_score(test_df.label, pred)
# print(f"f_1 score: {f1}")
# # print(f1)




