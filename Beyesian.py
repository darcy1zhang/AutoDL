import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# 定义贝叶斯线性回归模型
class BayesianLinearRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        return self.linear(x)

# 设置输入特征数和输出目标数
input_features = 13
output_features = 1

# 创建模型实例
model = BayesianLinearRegression(input_features, output_features)

# 设置损失函数和优化器
loss_function = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 加载数据（请将以下数据替换为实际数据）
X_train = np.load("./data/features_train_norm.npy")[:, :]
y_train = np.load("./data/simu_20000_0.1_90_140_train.npy")[:, 1004]
X_test = np.load("./data/features_test_norm.npy")[:, :]
y_test = np.load("./data/simu_10000_0.1_141_178_test.npy")[:, 1004]

y_train = y_train.reshape((20000, 1))
y_test = y_test.reshape((10000, 1))

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_train = X_train.type(torch.FloatTensor)
y_train = y_train.type(torch.FloatTensor)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
X_test = X_test.type(torch.FloatTensor)
y_test = y_test.type(torch.FloatTensor)


# 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    model.train()
    # 前向传播
    y_pred = model(X_train)
    
    # 计算损失
    loss = loss_function(y_pred, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# 验证模型
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不需要计算梯度
            y_test_pred = model(X_test)
            test_loss = loss_function(y_test_pred, y_test)
            print(f'Test Loss: {test_loss.item()}')
