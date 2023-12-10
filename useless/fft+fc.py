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



class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # self.dim_in = para.dim_in
        # self.dim_out = para.dim_out
        # self.dim_embeding = para.dim_embeding
        # self.dropout = para.dropout
        # self.negative_slope = para.negative_slope

        # self.BNlayer = nn.BatchNorm1d(self.dim_out)

        self.seq = nn.Sequential(
            # 先行正则化
            nn.BatchNorm1d(1),

            # 第一层
            nn.Linear(502, 256),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            # nn.Dropout(0.1),

            # 第二层
            nn.Linear(256, 256),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            # nn.Dropout(0.1),

            # 第三层
            nn.Linear(256, 128),
            nn.BatchNorm1d(1),
            # nn.ReLU(),
            nn.LeakyReLU(1e-4),
            # nn.Dropout(0.1),

            # 再来一层
            nn.Linear(128, 128),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            
            # 再来一层
            nn.Linear(128, 64),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            
            # 再来一层
            nn.Linear(64, 32),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            
            # 第四层
            nn.Linear(32, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        out_enh = self.seq(x)
        return out_enh


model = DNN().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = Dataset(Dataset)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# l2_lambda = 0.01 # L2正则化系数
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

        # # L2正则化
        # l2_reg = torch.tensor(0.).to("cuda:0")
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, 2)
        # loss += l2_lambda * l2_reg

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
    if loss_mean < 3:
        torch.save(model, tmp)
    if (epoch == 50) or (epoch == 100) or (epoch == 150) or (epoch == 200) or (epoch == 250):
        torch.save(model, tmp)