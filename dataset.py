import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, para, s_or_d, train_or_test, unrelated_feature_number):
        self.data = np.load(para)
        self.unrelated_feature_number = unrelated_feature_number
        train_data = np.load("../data/features_train.npy")

        # normalize
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        self.data = (self.data-mean)/std

        self.s_or_d = s_or_d
        self.raw_data_train = np.load("../data/simu_20000_0.1_90_140_train.npy")
        self.raw_data_test = np.load("../data/simu_10000_0.1_141_178_test.npy")
        self.train_or_test = train_or_test
        
        self.unrelated_feature = self.data[:,17:(17+unrelated_feature_number)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.s_or_d == "s":
            X_train = self.data[idx, :3]
        else:
            X_train = self.data[idx, 3:5]
            
        if self.unrelated_feature_number != 0:
            X_train = np.hstack((X_train,self.unrelated_feature[idx,:]))

        if self.train_or_test == "train":
            label_data = self.raw_data_train
        else:
            label_data = self.raw_data_test

        if self.s_or_d == "s":
            Y_train = label_data[idx, 1004]
        else:
            Y_train = label_data[idx, 1005]

        Y_train = np.array([Y_train])
        Y_train = Y_train.reshape((1,1))

        # 转为torch格式
        X_train = np.array([X_train])
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        # Y_train = Y_train.view(Y_train.size(0), 1, 1)

        return X_train, Y_train