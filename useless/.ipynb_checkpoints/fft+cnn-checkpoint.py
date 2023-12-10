import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Dataset(Dataset):

    def __init__(self, para):
        self.data = np.load("simu_20000_0.1_90_140_train.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X1 = self.data[idx,:1000]
        X2 = self.data[idx,1002:1004]

        X1 = np.fft.fft(X1)
        X1 = np.abs(X1)
        X1 = X1[1:501]

        X_train = np.concatenate([X1, X2], axis=0)
        Y_train = self.data[idx,-2:]

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 502)
        Y_train = Y_train.reshape(1, 2)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 40),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 20),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*51, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(16, -1, 128*51) # flatten the tensor

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)

        return output


model = CNN().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

train_dataset = Dataset(Dataset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

l2_lambda = 0.01 # L2正则化系数
loss_total = 0
step = 0

for epoch in range(300):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # data = data.reshape(-1, 10, 502)
        output = model(data)

        # print(output, target)


        loss = criterion(output, target)

        # L2正则化
        l2_reg = torch.tensor(0.).to("cuda:0")
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss
        step = step + 1

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    loss_mean = loss_total / step
    loss_total = 0
    step = 0
    tmp = './pth/model_%d_%.4f.pth'%(epoch,loss_mean)
    if loss_mean < 2:
        torch.save(model, tmp)
    if (epoch == 50) or (epoch == 100) or (epoch == 150) or (epoch == 200) or (epoch == 250):
        torch.save(model, tmp)