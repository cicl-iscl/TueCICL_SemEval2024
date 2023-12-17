import pandas as pd
import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from nltk.util import ngrams

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, test_df


def doc2vec(docs, dictionary, allow_update):
    # doc_tokenized = [gensim.utils.simple_preprocess(doc) for doc in docs]
    doc_tokenized = tokenize(docs, 5)
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=allow_update) for doc in doc_tokenized]
    tfidf = gensim.models.TfidfModel(BoW_corpus, smartirs='lfc')
    vecs = tfidf[BoW_corpus]
    vecs = conv_sparse_vecs(vecs, dictionary)
    return vecs


def conv_sparse_vecs(vecs, dictionary):
    # Convert TransformedCorpus to a sparse CSR matrix
    num_docs = len(vecs)
    num_features = len(dictionary)
    indptr = [0]
    indices = []
    data = []

    for doc in vecs:
        for word_id, value in doc:
            indices.append(word_id)
            data.append(value)
        indptr.append(len(indices))

    X_sparse = csr_matrix((data, indices, indptr), shape=(num_docs, num_features))
    return X_sparse

def tokenize(train_df, n):
    for doc in train_df:
        n_grams = []
        for i in range(len(doc)):
            n_grams.append(doc[i:i+n])
        yield n_grams

train_df, test_df = get_data("../../data/subtaskA_train_monolingual.jsonl", "../../data/subtaskA_dev_monolingual.jsonl", 0)

dictionary = gensim.corpora.Dictionary()

# vectorize training docs
X_train = doc2vec(train_df.text, dictionary, True)

# vectorize test docs
X_test = doc2vec(test_df.text, dictionary, False)


clf = RidgeClassifier(solver="sparse_cg", tol=1e-10)
clf.fit(X_train, train_df.label)
pred = clf.predict(X_test)
f1_1 = f1_score(test_df.label, pred)
print(f"f_1_1 score: {f1_1}")
train_df["ngrams"] = None
for i in range(len(train_df)):
    chars = list(ngrams(list(train_df.loc[i,"text"]), 5))
    new_chars = []
    for tup in chars:
        s = ""
        for char in tup:
            s += char
        new_chars.append(s)
    train_df.at[i,"ngrams"] = chars
print(train_df.head(10))