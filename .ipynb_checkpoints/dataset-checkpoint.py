import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, para, s_or_d, train_or_test, unrelated_feature_number):
        self.data = np.load(para)
        self.unrelated_feature_number = unrelated_feature_number
        train_data = np.load("../data/features_rand_train.npy")

        # normalize
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        self.data = (self.data-mean)/std

        self.s_or_d = s_or_d
        self.raw_data_train = np.load("../data/simu_20000_0.1_90_140_train.npy")
        self.raw_data_test = np.load("../data/simu_10000_0.1_141_178_test.npy")
        self.train_or_test = train_or_test
        
        self.unrelated_feature = self.data[:,17:(17+unrelated_feature_number)]
        
        
        if self.train_or_test == "train":
            self.label_data = self.raw_data_train
        else:
            self.label_data = self.raw_data_test
        
        self.data = torch.from_numpy(self.data)
        self.label_data = torch.from_numpy(self.label_data)
        self.data = self.data.type(torch.FloatTensor)
        self.label_data = self.label_data.type(torch.FloatTensor)
        

        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.s_or_d == "s":
            X_train = self.data[idx, :3]
        else:
            X_train = self.data[idx, 3:5]
            
        if self.unrelated_feature_number != 0:
            X_train = np.hstack((X_train,self.unrelated_feature[idx,:]))

        if self.s_or_d == "s":
            Y_train = self.label_data[idx, 1004]
        else:
            Y_train = self.label_data[idx, 1005]

        Y_train = Y_train.reshape((1,1))
        X_train = X_train.reshape((1,-1))

        return X_train, Y_train