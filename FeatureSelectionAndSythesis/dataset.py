import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, data):
        self.data = np.load(para).astype(np.float64)
        self.unrelated_feature_number = unrelated_feature_number
        train_data = np.load("../data/features_rand_train.npy").astype(np.float64)


        self.s_or_d = s_or_d
        self.raw_data_train = np.load("../data/simu_20000_0.1_90_140_train.npy").astype(np.float64)
        self.raw_data_test = np.load("../data/simu_10000_0.1_141_178_test.npy").astype(np.float64)
        self.train_or_test = train_or_test
        
        self.unrelated_feature = self.data[:,17:(17+unrelated_feature_number)]
        
        
        if self.train_or_test == "train":
            self.label_data = self.raw_data_train
        else:
            self.label_data = self.raw_data_test
        
        self.data = torch.from_numpy(self.data)
        self.label_data = torch.from_numpy(self.label_data)
        # self.data = self.data.type(torch.DoubleTensor)
        # self.label_data = self.label_data.type(torch.DoubleTensor)
        
        if self.s_or_d == "s":
            self.data = self.data[:, :3]
        else:
            self.data = self.data[:, 3:5]
            
        if self.s_or_d == "s":
            self.label_data = self.label_data[:, 1004]
        else:
            self.label_data = self.label_data[:, 1005]

        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        X_train = self.data[idx, :]
            
        if self.unrelated_feature_number != 0:
            X_train = np.hstack((X_train,self.unrelated_feature[idx,:]))

        Y_train = self.label_data[idx]

        Y_train = Y_train.reshape((1,1))
        X_train = X_train.reshape((1,-1))

        return X_train, Y_train