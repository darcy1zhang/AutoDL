import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import gc

class Config():
    data_path_train = 'simu_20000_0.1_90_140_train.npy'
    data_path_test = 'simu_10000_0.1_141_178_test.npy'
    batch_size = 32  # 批次大小
    feature_size = 1003  # 每个步长对应的特征数量，这里只使用2
    hidden_size = 120  # 隐层大小
    output_size = 2  # 输出2个
    num_layers = 2  # lstm的层数
    epochs = 300 # 迭代轮数
    best_loss = 10 # 记录损失
    learning_rate = 0.0001 # 学习率
    model_name = 'zdj_lstm' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径

config = Config()
# 1.加载时间序列数据

# data_train = np.load(config.data_path_train)
# data_train = np.delete(data_train, -5, axis=1)
#
#
# data_test = np.load(config.data_path_test)
# data_test = np.delete(data_test, -5, axis=1)



# merged_data = np.concatenate((data_train, data_test), axis=0)
#
# # 2.将数据进行标准化
# # 2.将数据进行标准化
#
# scaler = MinMaxScaler()
# scaler_model = MinMaxScaler()
# data_scaled = scaler_model.fit_transform(merged_data)
# scaler.fit_transform(merged_data[:,-2:])#
#
#
# # 形成训练数据，例如12345789 12345-67 23456-78
# def split_data(data):
#     dataX_ = data[:, :-2] # 保存X
#     dataY_ = data[:, -2:]  # 保存Y
#     # 将整个窗口的数据保存到X中，将未来一天保存到Y中
#
#     dataX = dataX_.reshape(-1, 1, dataX_.shape[1])
#     dataY = dataY_.reshape(-1,dataY_.shape[1])
#
#     # 获取训练集大小
#     # 划分训练集、测试集
#     x_train = dataX[: len(data_train), :]
#     y_train = dataY[: len(data_train)]
#
#     x_test = dataX[len(data_train):, :]
#     y_test = dataY[len(data_train):]
#     return [x_train, y_train, x_test, y_test]



data_train = np.load("simu_20000_0.1_90_140_train.npy")
data_test = np.load("simu_10000_0.1_141_178_test.npy")
x1 = data_train[:,:1000]
x2 = np.fft.fft(x1)
x2 = x2.real
x2 = x2[:,1:501]
x3 = data_train[:,-4:-2]
x_train = np.concatenate((x2, x3), axis=1)
y_train = data_train[:,-2:]

x1_test = data_test[:,:1000]
x2_test = np.fft.fft(x1_test)
x2_test = x2_test.real
x2_test = x2_test[:,1:501]
x3_test = data_test[:,-4:-2]
x_test = np.concatenate((x2_test, x3_test), axis=1)
y_test = data_test[:,-2:]

x_train = x_train.reshape(20000, 1, 502)
y_train = y_train.reshape(20000, 1, 2)
x_test = x_test.reshape(10000, 1, 502)
y_test = y_test.reshape(10000, 1, 2)

# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
# x_train, y_train, x_test, y_test = split_data(data_scaled)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to('cuda')
y_train_tensor = torch.from_numpy(y_train).to(torch.float32).to('cuda')
x_test_tensor = torch.from_numpy(x_test).to(torch.float32).to('cuda')
y_test_tensor = torch.from_numpy(y_test).to(torch.float32).to('cuda')#双精度1

# print(scaler.inverse_transform(y_test_tensor.detach().cpu().numpy()))
# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False)
test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)


# 7.定义LSTM网络
# class LSTM(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size  # 隐层大小
#         self.num_layers = num_layers  # lstm层数
#         # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
#         self.relu = nn.ReLU(inplace=True)
#         self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#     def forward(self, x, hidden=None):
#         batch_size = x.shape[0]  # 获取批次大小

#         # 初始化隐层状态
#         # if hidden is None:
#         #     h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
#         #     c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
#         # else:
#         #     h_0, c_0 = hidden
#         if hidden is None:
#             h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).double()
#             c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).double()#双精度2
#         else:
#             h_0, c_0 = hidden
#             h_0 = h_0.double()#双精度3
#             c_0 = c_0.double()


#         # LSTM运算
#         output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
#         output = output[:, -1, :] # batch_size, timestep, output_size
#         # 全连接层
#         output = self.fc(output)
#         # 我们只需要返回最后一个时间片的数据即可
#         return output.to('cuda')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sequential block of layer1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, 9),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, 9),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.adaptive = nn.AdaptiveMaxPool1d(4)


        # lstm and fully connected layer
        self.lstm = nn.LSTM(1024, 56)
        self.fc1 = nn.Linear(56, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self, x):
        # x = x.unsqueeze(-2)
        out = self.layer1(x)
        out = self.layer2(out)

        # adaptive maxpool
        out = self.adaptive(out)
        # print(out.shape)

        # flatten
        out = out.reshape(out.size(0), -1)
        #print(out1.shape)

        # lstm layer
        # out = out.unsqueeze(0)
        out, hid = self.lstm(out)
        # print(out.shape)

        # output layer
        out = self.fc1(out)
        out = self.fc2(out)
        # out = out.squeeze(-1)
        # print(out.shape)
        return out



model = ConvNet()  # 定义LSTM网络


# model = LSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size).double()  # 定义LSTM网络
loss_function = nn.L1Loss() # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 定义优化器

# 8.模型训练
for epoch in range(config.epochs):

    model.train().to('cuda')
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        # x_train = x_train.reshape(1, 32, 502)
        # y_train = y_train.reshape(1, 32, 2)
        x_train = x_train.to('cuda')
        y_train = y_train.to('cuda')
        optimizer.zero_grad()


        y_train_pred = model(x_train).to('cuda')


        # print(y_train_pred,y_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, config.output_size))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.9f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)
    # 模型验证
    model.eval().to('cuda')
    test_loss = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test = y_test.to('cuda')
            y_train = y_train.to('cuda')
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, config.output_size))
            print("----------" + str(test_loss.mean().item()) + "----------")

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)
    # gc.collect()
    # torch.cuda.empty_cache()

# print('Finished Training')
# # 9.绘制结果
# plot_size = 200
# plt.figure(figsize=(12, 8))
#
# y_test_pred = model(x_test_tensor[:,:,:])
# y_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
# y_yes = scaler.inverse_transform(y_test_tensor[:,:].detach().cpu().numpy())

# loss_function = nn.L1Loss()
# mae = loss_function(torch.from_numpy(y_pred).to('cuda'), torch.from_numpy(y_yes).to('cuda'))
# print("======测试集mae：" + str(mae) + "======")
# plt.figure(figsize=(12, 8))
# y_test_pred = model(x_test_tensor[0:200,:,:])
# plt.plot(scaler.inverse_transform(y_test_pred.detach().cpu().numpy())[:,1], "b")#S
# plt.plot(scaler.inverse_transform(y_test_tensor[0:200,:].detach().cpu().numpy())[:,1], "r")#S
# plt.legend()
# plt.show()
# print(scaler.inverse_transform(y_test_pred.detach().cpu().numpy())[:,0])
# [[172.  83.]
#  [173.  79.]
#  [146.  73.]
#  ...
#  [155.  88.]
#  [159.  87.]
#  [164.  69.]]