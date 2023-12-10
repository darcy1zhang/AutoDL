import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Dataset_train(Dataset):

    def __init__(self):
        self.data = np.load("simu_20000_0.1_90_140_train.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X_train = self.data[idx, :1000]

        Y_train = self.data[idx, 1004]
        Y_train = np.array([Y_train])

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 1000)
        Y_train = Y_train.reshape(1, 1)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train

class Dataset_test(Dataset):
    def __init__(self):
        self.data_test = np.load("simu_10000_0.1_141_178_test.npy")

    def __len__(self):
        return self.data_test.shape[0]

    def __getitem__(self, idx):
        X_test = self.data_test[idx,:1000]

        Y_test = self.data_test[idx,1004]
        Y_test = np.array([Y_test])

        # 转为torch格式
        X_test = torch.from_numpy(X_test)
        Y_test = torch.from_numpy(Y_test)
        X_test = X_test.reshape(1, 1000)
        Y_test = Y_test.reshape(1, 1)
        X_test = X_test.type(torch.FloatTensor)
        Y_test = Y_test.type(torch.FloatTensor)

        return X_test, Y_test

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 5),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 8, 5),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, 5),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32 * 48, 10)
        )

        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(16, -1, 32 * 48)  # flatten the tensor

        x = self.fc1(x)
        output = self.out(x)

        return output

model = VGG().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = Dataset_train()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = Dataset_test()
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


l2_lambda = 0.02  # L2正则化系数
loss_best = 20


for epoch in range(300):
    model.train()

    loss_total = 0
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)

        # L2正则化
        l2_reg = torch.tensor(0.).to("cuda:0")
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    print("epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))

    # model.eval().to('cuda')
    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        tmp = './pth/model_%d_%.4f.pth' % (epoch, loss_total)
        if loss_mean < loss_best:
            loss_best = loss_mean
            torch.save(model, tmp)
        if (epoch == 50) or (epoch == 100) or (epoch == 150) or (epoch == 200) or (epoch == 250):
            torch.save(model, tmp)

    print("epoch:" + str(epoch) + "    test_MAE:" + str(loss_mean))
