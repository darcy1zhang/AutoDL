import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, para, s_or_d, train_or_test, unrelated_feature_number):
        self.data = np.load(para)
        self.unrelated_feature_number = unrelated_feature_number
        train_data = np.load("../data/features_rand_train_90_130.npy")

        # normalize
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        self.data = (self.data-mean)/std

        self.s_or_d = s_or_d
        self.raw_data_train = np.load("../data/simu_20000_0.1_90_140_train.npy")
        self.raw_data_test = np.load("../data/simu_10000_0.1_141_178_test.npy")
        self.train_or_test = train_or_test
        
        self.train_idx = np.load("../data/features_rand_train_idx.npy")
        self.validate_idx = np.load("../data/features_rand_validate_idx.npy")
        
        self.unrelated_feature = self.data[:,17:(17+unrelated_feature_number)]
        
        if self.train_or_test == 0:
            self.label_data = self.raw_data_train[self.train_idx]
        elif self.train_or_test == 1:
            self.label_data = self.raw_data_train[self.validate_idx]
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
            



        # Y_train = np.array([Y_train])
        Y_train = Y_train.reshape((1,1))
        X_train = X_train.reshape((1,-1))

        # 转为torch格式
        # X_train = np.array([X_train])
        # X_train = torch.from_numpy(X_train)
        # Y_train = torch.from_numpy(Y_train)
        # X_train = X_train.type(torch.FloatTensor)
        # Y_train = Y_train.type(torch.FloatTensor)
        
        # Y_train = Y_train.view(Y_train.size(0), 1, 1)

        return X_train, Y_train
    
if __name__ == "__main__":
    train_dataset = Dataset("./data/features_rand_train_90_115.npy", 0, 0, 0)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)

    val_dataset = Dataset("./data/features_rand_validate_116_140.npy", 0, 1, 0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True,drop_last=True)

    test_dataset = Dataset("./data/features_rand_test.npy", 0, 2, 0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
