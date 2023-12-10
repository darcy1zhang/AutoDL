import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

class Dataset_test(Dataset):

    def __init__(self, para):
        self.data = np.load("simu_10000_0.1_141_178_test.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X_train = self.data[idx,:1000]

        Y_train = self.data[idx,1005]
        Y_train = np.array([Y_train])

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 1000)
        Y_train = Y_train.reshape(1, 1)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 300),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=6)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 60),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*12, 128),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(16, -1, 128*12) # flatten the tensor

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)


model = CNN()
model = torch.load("./pth/D_cnn_12.1621_11.8.pth" ,map_location = torch.device('cpu'))
# model.load_state_dict(torch.load("./pth/model_50_11.4820.pth"))
dataset_test = Dataset_test(Dataset)
train_loader = DataLoader(dataset_test, batch_size=16, shuffle=True)
criterion = nn.L1Loss()


loss_total = 0
batch_num = 0


for batch_idx, (data, target) in enumerate(train_loader):
    # data, target = data.cuda(), target.cuda()
    output = model(data)
    # print("_______", output, target)
    loss = criterion(output, target)
    # print(loss.mean().item())
    loss_total = loss_total + loss.item()
    batch_num = batch_num + 1

print(loss_total / batch_num)



