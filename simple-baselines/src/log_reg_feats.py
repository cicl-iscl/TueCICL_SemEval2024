import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler


class Model(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, n_output_features)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class TaskA_Dataset(torch.utils.data.Dataset):
    def __init__(self, split="train") -> None:
        if split == "train":
            self.data = pd.read_json("../../data/subtaskA_spacy_feats.json")
            self.data.drop(["passed_quality_check", "oov_ratio"], inplace=True, axis=1)
            print(self.data.isnull().values.any())
            self.data.dropna(inplace=True)
            print(self.data.isnull().values.any())
            self.feats = self.data.drop(["label", "id", "text"], axis=1).values
            sc = StandardScaler()
            X_train = sc.fit_transform(self.feats)
            X_train = torch.from_numpy(X_train.astype(np.float32))
            self.feats = X_train
        else:
            self.data = pd.read_json("../../data/subtaskA_dev_monolingual.jsonl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label, id = item["label"], item["id"]
        return self.feats[index], label, id

def collate_fn(data):
    feats = [data[i][0] for i in range(len(data))]
    label = [data[i][1] for i in range(len(data))]
    id = [data[i][2] for i in range(len(data))]
    return torch.tensor(np.array(feats, dtype=np.float32)), torch.tensor(np.array(label, dtype=np.float32)), torch.tensor(np.array(id, dtype=np.float32))


log_regr = Model(68, 1)
# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.001)
# defining Cross-Entropy loss
criterion = torch.nn.BCELoss()

train_df = pd.read_json("../../data/subtaskA_spacy_feats.json")

# test_df = pd.read_json("../../data")
trainset = TaskA_Dataset("train")
# load train and test data samples into dataloader
batach_size = 32
train_loader = DataLoader(dataset=trainset, batch_size=batach_size, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(dataset=test_df, batch_size=batach_size, shuffle=False)

epochs = 50
Loss = []
acc = []
for epoch in range(epochs):
    for i, (feats, labels, ids) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = log_regr(feats).squeeze()
        x = 0
        loss = criterion(outputs, labels)
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    Loss.append(loss.item())
    correct = 0
    # for images, labels in test_loader:
    #     outputs = log_regr(images.view(-1, 28 * 28))
    #     _, predicted = torch.max(outputs.data, 1)
    #     correct += (predicted == labels).sum()
    # accuracy = 100 * (correct.item()) / len(test_df)
    # acc.append(accuracy)
    # print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))