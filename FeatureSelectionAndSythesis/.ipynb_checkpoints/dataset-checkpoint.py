import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, data, label):
        self.data = data 
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        X = self.data[idx, :]
        Y = self.label[idx]

        X = X.reshape((1,-1))
        Y = Y.reshape((1,1))
        
        return X, Y