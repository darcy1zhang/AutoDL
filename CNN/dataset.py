import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, para, x, train_or_test):
        self.data = np.load(para)
        self.x = x
        self.features_train = np.load("./data/features_train_norm.npy")
        self.features_test = np.load("./data/features_test_norm.npy")
        self.train_or_test = train_or_test

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        if self.train_or_test == 0:
            X_train = self.features_train[idx, :]
        else:
            X_train = self.features_test[idx, :]

        if self.x == 0:
            Y_train = self.data[idx, 1004]
        else:
            Y_train = self.data[idx, 1005]
        Y_train = np.array([Y_train])

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train