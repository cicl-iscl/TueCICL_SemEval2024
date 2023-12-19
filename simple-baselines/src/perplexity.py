from transformers import GPT2Tokenizer, AutoModelForCausalLM
from torch.nn.functional import log_softmax
import torch
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import pandas as pd

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, test_df

def encode(string):
    ids = []
    spl = string.split()
    for i, tok in enumerate(spl):
        if i == 1022:
            break
        encoded = tokenizer.encode(tok)
        ids.extend(encoded)
    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    return ids


def perplexity(logits, l, input_ids):
    logprobs = log_softmax(logits, dim=1)
    logprobs = np.array(logprobs)
    total_prob = 0
    for i in range(len(input_ids)):
        curr_prob = logprobs[i, input_ids[i]]
        total_prob += curr_prob
    ppl = - (1 / l) * total_prob
    return np.exp(ppl)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
train_df, test_df = get_data("../../data/subtaskA_train_monolingual.jsonl", "../../data/subtaskA_dev_monolingual.jsonl", 0)

red_df = pd.concat([train_df.iloc[:100,:], train_df.iloc[-100:,:]])
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.to(device)
human_ppl_list = []
machine_ppl_list = []
ppl_list = []
for i in range(len(red_df)):
    input_ids = encode(train_df.text.iloc[i])
    if len(input_ids) > 1024:
        input_ids = input_ids[:1022] + [tokenizer.eos_token_id]
    input_text = torch.tensor(input_ids).to(device)
    out = model(torch.tensor(input_text).unsqueeze(0))
    logits = out.logits.detach().cpu()
    logits = logits.squeeze()
    l = len(input_ids)
    ppl = perplexity(logits, l, input_ids)
    ppl_list.append(ppl)
    # if train_df.iloc[i]["label"] == 1:
    #     machine_ppl_list.append(ppl)
    # else:
    #     human_ppl_list.append(ppl)

plt.plot(ppl_list)
plt.show()
machine = np.mean(ppl_list[:100])
human = np.mean(ppl_list[100:])
# machine = np.mean(np.array(machine_ppl_list))
# human = np.mean(np.array(human_ppl_list))
print(f"machine: {machine}, human: {human}")
# with open("data/ppl_list.json", "w", encoding="utf-8") as f:
#     json.dump(ppl_list, f)