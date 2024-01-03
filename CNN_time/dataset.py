import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, para, s_or_d, train_or_test):
        tmp = np.load("../data/simu_20000_0.1_90_140_train.npy")[:,:1000]
        min_value = np.min(tmp)
        max_value = np.max(tmp)
        
        self.data = np.load(para)
        self.s_or_d = s_or_d
        self.train_or_test = train_or_test
        self.X = self.data[:, :1000]
        self.X = (self.X - min_value)/(max_value - min_value)
        
        if self.s_or_d == 0:
            self.label = self.data[:, 1004]
        else:
            self.label = self.data[:, 1005]
        self.X = torch.from_numpy(self.X).cuda()
        self.label = torch.from_numpy(self.label).cuda()
        self.X = self.X.type(torch.FloatTensor)
        self.label = self.label.type(torch.FloatTensor)

    def __len__(self):
        return self.data.shape[0]
        # return 2000

    def __getitem__(self, idx):
        Y = self.label[idx].reshape(1,1)
        X = self.X[idx].reshape(1,-1)
        X = torch.cat((torch.arange(1,1001).reshape(1,-1),X),axis=0)

        return X, Y