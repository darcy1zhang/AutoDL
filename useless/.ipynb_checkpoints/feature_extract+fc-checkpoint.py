import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.fft import fft
from scipy.stats import skew, kurtosis

def extract_features_freq(signal):
    # 进行傅里叶变换
    spectrum = np.abs(fft(signal))
    # 计算频谱能量特征
    energy = np.sum(spectrum ** 2)
    # 计算频谱均值和方差特征
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    # 找到频谱中的最大值（峰值）及其位置
    max_amp = np.max(spectrum)
    max_freq = np.argmax(spectrum).astype("float64")
    # 计算频谱峰值频率的偏度和峰度特征
    spectrum = spectrum.reshape(1000)
    normalized_freq = np.arange(len(spectrum)) - max_freq
    skewness = np.float64(skew(normalized_freq))
    kurt = np.float64(kurtosis(normalized_freq))
    # 计算频谱变化率特征
    # spectrum = spectrum.reshape(-1, 1000)
    diff_spectrum = np.diff(spectrum)
    change_rate = np.mean(diff_spectrum / spectrum[:-1])
    # 计算频谱带宽特征
    bw = np.sum(spectrum >= max_amp / 2).astype("float64")
    # 将特征值整理成一个数组
    features = np.array([energy, mean, std, max_amp, max_freq, change_rate, bw])
    skewness = skewness.reshape(1, -1)
    kurt = kurt.reshape(1, -1)
    features = features.reshape(1, -1)
    features = np.concatenate((features, skewness, kurt), axis=1)
    return features.astype("float64")

def extract_features_time(signal):
    # 计算均值
    mean = np.mean(signal)

    # 计算方差
    variance = np.var(signal)

    # 计算标准差
    std = np.std(signal)

    # 计算均方根值（RMS）
    rms = np.sqrt(np.mean(signal ** 2))

    # 计算偏度
    skewness = np.mean((signal - mean) ** 3) / (std ** 3)

    # 计算峰度
    kurtosis = np.mean((signal - mean) ** 4) / (std ** 4) - 3

    # 将所有特征组合成一个ndarray
    features = np.array([mean, variance, std, rms, skewness, kurtosis])
    return features

class Dataset(Dataset):

    def __init__(self, para):
        self.data = np.load("simu_20000_0.1_90_140_train.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X1 = self.data[idx,:1000]
        X2 = self.data[idx,1002:1004]

        X1 = X1.astype('float64')
        X2 = X2.astype('float64')

        X2 = X2.reshape(1, -1)

        # X1 = np.fft.fft(X1)
        # X1 = np.abs(X1)
        # X1 = X1[1:501]

        feature_t = extract_features_time(X1)
        feature_t = feature_t.reshape(1, -1)
        feature_f = extract_features_freq(X1)
        feature_f = feature_f.reshape(1, -1)


        # X_train = np.concatenate([X1, X2], axis=0)
        X_train = np.concatenate((feature_t, feature_f, X2), axis=1)

        Y_train = self.data[idx,-2:]

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 17)
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
            nn.Linear(17, 10),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(1e-4),
            nn.Dropout(0.1),

            # # 第二层
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(1),
            # nn.LeakyReLU(1e-4),
            # nn.Dropout(0.1),
            #
            # # 第三层
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(1),
            # # nn.ReLU(),
            # nn.LeakyReLU(1e-4),
            # nn.Dropout(0.1),

            # 第四层
            nn.Linear(10, 2)
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