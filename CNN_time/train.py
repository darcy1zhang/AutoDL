import torch.optim as optim
from dataset import *
from model import *
from sklearn.preprocessing import MinMaxScaler

criterion = nn.L1Loss()
criterion = criterion.cuda()

model = VGG().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

tmp = np.load("../data/simu_20000_0.1_90_140_train.npy")
max = np.max(tmp[:, 1004])
min = np.min(tmp[:, 1004])
print(max, min)

train_dataset = Dataset("./data/simu_20000_0.1_90_140_resonance_morlet.npy", 0, 0)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = Dataset("./data/simu_10000_0.1_90_140_test.npy", 0, 1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

for epoch in range(500):
    model.train()

    loss_total = 0
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # target = (target - min) / (max - min)

        output = model(data)

        loss = criterion(output, target)
        for param in model.parameters()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    tmp = './pth/S_model_%d_%.4f.pth' % (epoch, loss_total/step)
    if epoch % 10 == 0:
        torch.save(model, tmp)
    print("S--epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))

    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            # inv_norm
            output = output * (max - min) + min

            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        print("epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))
