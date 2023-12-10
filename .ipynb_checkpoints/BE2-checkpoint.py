import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义贝叶斯线性回归模型
class BayesianLinearRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        return self.linear(x)

# 设置输入特征数和输出目标数
input_features = 14
output_features = 1

# 创建模型实例
model = BayesianLinearRegression(input_features, output_features)


loss_function = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 加载数据（请将以下数据替换为实际数据）
X = np.load("./data/features_train_norm.npy")
y = np.load("./data/simu_20000_0.1_90_140_train.npy")[:, 1004]
y = y.reshape((20000, 1))
X = torch.from_numpy(X)
y = torch.from_numpy(y)
X = X.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)



# 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_function(y_pred, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
