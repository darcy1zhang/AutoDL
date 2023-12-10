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

class Config():
    data_path_train = 'simu_20000_0.1_90_140_train.npy'
    data_path_test = 'simu_10000_0.1_141_178_test.npy'
    batch_size = 100  # 批次大小
    feature_size = 1003  # 每个步长对应的特征数量，这里只使用2
    hidden_size = 120  # 隐层大小
    output_size = 2  # 输出2个
    num_layers = 2  # lstm的层数
    epochs = 200 # 迭代轮数
    best_loss = 0 # 记录损失
    learning_rate = 0.0001 # 学习率
    model_name = 'lstm' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径


config = Config()
# 1.加载时间序列数据

data_train = np.load(config.data_path_train)
data_train = np.delete(data_train, -5, axis=1)


data_test = np.load(config.data_path_test)
data_test = np.delete(data_test, -5, axis=1)



merged_data = np.concatenate((data_train, data_test), axis=0)

# 2.将数据进行标准化
# 2.将数据进行标准化

scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data_scaled = scaler_model.fit_transform(merged_data)
scaler.fit_transform(merged_data[:,-2:])#只能做对两列的恢复


# 形成训练数据，例如12345789 12345-67 23456-78
def split_data(data):
    dataX_ = data[:, :-2] # 保存X
    dataY_ = data[:, -2:]  # 保存Y
    # 将整个窗口的数据保存到X中，将未来一天保存到Y中

    dataX = dataX_.reshape(-1, 1, dataX_.shape[1])
    dataY = dataY_.reshape(-1,dataY_.shape[1])

    # 获取训练集大小
    # 划分训练集、测试集
    x_train = dataX[: len(data_train), :]
    y_train = dataY[: len(data_train)]

    x_test = dataX[len(data_train):, :]
    y_test = dataY[len(data_train):]
    return [x_train, y_train, x_test, y_test]


# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
x_train, y_train, x_test, y_test = split_data(data_scaled)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float64).to('cuda')
y_train_tensor = torch.from_numpy(y_train).to(torch.float64).to('cuda')
x_test_tensor = torch.from_numpy(x_test).to(torch.float64).to('cuda')
y_test_tensor = torch.from_numpy(y_test).to(torch.float64).to('cuda')#双精度1

print(scaler.inverse_transform(y_test_tensor.detach().cpu().numpy()))
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



model_state_dict = torch.load('zdj_lstm.pth')
class RNN(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.RNN(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).double()
        else:
            h_0 = hidden.double()

        # RNN运算
        output, h_0 = self.rnn(x, h_0)
        output = output[:, -1, :] # batch_size, timestep, output_size
        # 全连接层
        output = self.fc(output)
        output = self.sigmoid(output)
        # 我们只需要返回最后一个时间片的数据即可
        return output


model = RNN(config.feature_size, config.hidden_size, config.num_layers, config.output_size).double()  # 定义RNN网络
model.load_state_dict(model_state_dict)
model.eval().to('cuda')
# 9.绘制结果
plot_size = 200
plt.figure(figsize=(12, 8))

y_test_pred = model(x_test_tensor[:,:,:])
y_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_yes = scaler.inverse_transform(y_test_tensor[:,:].detach().cpu().numpy())

loss_function = nn.L1Loss()
mae = loss_function(torch.from_numpy(y_pred).to('cuda'), torch.from_numpy(y_yes).to('cuda'))
print("======测试集mae：" + str(mae) + "======")
plt.figure(figsize=(12, 8))
y_test_pred = model(x_test_tensor[0:200,:,:])
plt.plot(scaler.inverse_transform(y_test_pred.detach().cpu().numpy())[:,1], "b")#S
plt.plot(scaler.inverse_transform(y_test_tensor[0:200,:].detach().cpu().numpy())[:,1], "r")#S
plt.legend()
plt.show()
print(scaler.inverse_transform(y_test_pred.detach().cpu().numpy())[:,0])
# [[172.  83.]
#  [173.  79.]
#  [146.  73.]
#  ...
#  [155.  88.]
#  [159.  87.]
#  [164.  69.]]