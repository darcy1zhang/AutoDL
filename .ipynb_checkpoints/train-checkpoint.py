import torch.optim as optim
from dataset import *
from model import *
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
# import tsfel

# X_train = tsfel.time_series_features_extractor(cfg_file, tmp, fs=100, window_size=250)


criterion = nn.L1Loss()
criterion = criterion.cuda()

model = MLP().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# writer = SummaryWriter('logs/fs5_sigmoid')

s_or_d = 0
tmp = np.load("./data/simu_20000_0.1_90_140_train.npy")
if s_or_d == 0:
    max = np.max(tmp[:, 1004])
    min = np.min(tmp[:, 1004])
else:
    max = np.max(tmp[:, 1005])
    min = np.min(tmp[:, 1005])


train_dataset = Dataset("./data/feature_x1x2y1y2_train.npy", s_or_d, 0)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = Dataset("./data/feature_x1x2y1y2_test.npy", s_or_d, 1)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

for epoch in range(1000):
    model.train()

    loss_total = 0
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # print(data.shape)

        # target = (target - min) / (max - min)

        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    tmp = './pth/x1x2_fcn_%d_%.4f.pth' % (epoch, loss_total/step)
    if epoch % 10 == 0:
        torch.save(model, tmp)
        print("Train_epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))

    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            # inv_norm
            # output = output * (max - min) + min

            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        if epoch % 10 == 0:
            print("Test_epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))

