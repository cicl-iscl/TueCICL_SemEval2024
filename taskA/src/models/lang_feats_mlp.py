import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super(Model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_output_features)
        )

    def forward(self, x):
        y_pred = torch.sigmoid(self.mlp(x))
        return y_pred




class TaskA_Dataset(torch.utils.data.Dataset):
    def __init__(self, split="train") -> None:
        if split == "train":
            self.data = pd.read_json("../../data/subtaskA_spacy_feats.json")
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.data.drop(["passed_quality_check", "oov_ratio", "n_characters", "n_sentences"], inplace=True, axis=1)
            self.data.dropna(inplace=True)
            self.feats = np.array(self.data.drop(["label", "id", "text"], axis=1).values)
            X_train = self.feature_scale(self.feats, True)
            X_train = torch.from_numpy(X_train.astype(np.float64))
            self.feats = X_train
        else:
            self.data = pd.read_json("../../data/subtaskA_test_spacy_feats.json")
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.data.drop(["passed_quality_check", "oov_ratio", "n_characters", "n_sentences"], inplace=True, axis=1)
            self.data.dropna(inplace=True)
            self.feats = np.array(self.data.drop(["label", "id", "text"], axis=1).values)
            X_test = self.feature_scale(self.feats, False)
            X_test = torch.from_numpy(X_test.astype(np.float64))
            self.feats = X_test
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label, id = item["label"], item["id"]
        return self.feats[index], label, id

    def feature_scale(self, x, train):
        if train:
            means = np.mean(x, axis=0)
            sd = np.std(x, axis=0)
            self.means = means
            self.sd = sd
        else:
            means = train_set.means
            sd = train_set.sd
        means = np.tile(means, (np.shape(x)[0], 1))
        sd = np.tile(sd, (np.shape(x)[0], 1))
        return (x - means) / sd


log_regr = Model(66, 1)
log_regr.double()
# defining the optimizer
optimizer = torch.optim.Adam(log_regr.parameters(), lr=0.0005)
# defining Cross-Entropy loss
criterion = torch.nn.BCELoss()

train_set = TaskA_Dataset("train")
test_set = TaskA_Dataset("test")
X_train = train_set.feats
Y_train = torch.from_numpy(train_set.data.label.values.astype(np.float64))
X_test = test_set.feats

# test = torch.sum(X_test, dim=0)
# train = x = torch.sum(X_train, dim=0)

Y_test = torch.from_numpy(test_set.data.label.values.astype(np.float64))

y_train = Y_train.view(Y_train.shape[0], 1)
y_test = Y_test.view(Y_test.shape[0], 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_regr = log_regr.to(device)
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

epochs = 2000
Loss = []
acc = []
for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    outputs = log_regr(X_train).squeeze()
    x = 0
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    Loss.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}\n')
        print(f'number of ones: {len(outputs[outputs >= 0.5])}')
        with torch.no_grad():
            y_predicted = log_regr(X_test).squeeze()
            y_predicted_cls = y_predicted.round()
            print(f'number of ones: {len(y_predicted[y_predicted == 1])}')
            f1_1 = f1_score(Y_test.cpu(), y_predicted_cls.cpu())
            print(f"f_1_1 score: {f1_1}")