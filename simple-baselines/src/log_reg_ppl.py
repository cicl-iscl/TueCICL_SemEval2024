from sklearn.linear_model import LogisticRegression
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from torch.nn.functional import log_softmax
import torch
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import f1_score

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, test_df


def get_ppl_data(dataset, model, tokenizer, device):

    class TaskADataset(torch.utils.data.Dataset):
        """
        Wrapper for the IMDB dataset which returns the tokenized text
        and truncates / pads to a maximum length of 64 tokens.
        This is done following the paper referenced above where the input review
        snippets were maximally 64 tokens and then the review had to be completed
        with a positive sentiment.
        """

        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer
            # following the paper referenced above, input texts are <= 64 tokens
            self.max_len = 1022
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def __getitem__(self, idx):
            # get the text from the dataset
            text = self.dataset.iloc[idx]['text']
            # tokenize the text
            # and manually prepend BOS token (GPT-2 tokenizer doesn't do it somehow)
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            # return the tokens and the attention mask
            return {
                'input_ids': tokens.input_ids.squeeze().to(self.device),
                'attention_mask': tokens.attention_mask.squeeze().to(self.device)
            }

        def __len__(self):
            return len(self.dataset)

    def perplexity(logits, batch):
        logprobs = log_softmax(logits, dim=-1)
        logprobs = np.array(logprobs)
        ppl_lst = []
        for i in range(len(logprobs)):
            input_ids = batch["input_ids"][i].cpu()
            attention_mask = batch["attention_mask"][i].cpu()
            attention_mask = np.array(attention_mask)
            l = len(attention_mask[attention_mask == 1])
            total_prob = 0
            for j in range(l -1):
                curr_prob = logprobs[i, j, input_ids[j+1]]
                total_prob += curr_prob
            ppl_lst.append(np.exp( - (1 / l) * total_prob))
        return ppl_lst



    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    ppl_list = []
    dataset = TaskADataset(dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    for batch in tqdm(dataloader):
        out = model(**batch)
        logits = out.logits.detach().cpu()
        logits = logits.squeeze()
        ppl = perplexity(logits, batch)
        ppl_list += ppl
    return ppl_list


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_df, test_df = get_data("../../data/subtaskA_train_monolingual.jsonl",
                             "../../data/subtaskA_dev_monolingual.jsonl", 0)

red_df = pd.concat([train_df.iloc[:100, :], train_df.iloc[-100:, :]])
red_test_df = pd.concat([test_df.iloc[:100, :], test_df.iloc[-100:, :]])

red_df = red_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ppl_train = get_ppl_data(red_df, model, tokenizer, device)

clf = LogisticRegression()
clf.fit(ppl_train, red_df.label)

ppl_test = get_ppl_data(test_df, model, tokenizer, device)
pred = clf.predict(ppl_test)

count = pred[pred == 1]
print(len(count))

f1_1 = f1_score(test_df.label, pred)
print(f"f_1_1 score: {f1_1}")