import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

class CNN_four_layer(nn.Module):
    def __init__(self, kernel_size):
        super(CNN_four_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.linear = nn.Linear(63,1)
        
    def forward(self,x):
        feature = self.conv(x)
        # print(feature.shape[-1])
        output = self.linear(feature)
        return output
    

class OneToOneLayer(nn.Module):
    def __init__(self, input_size):
        super(OneToOneLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, input_size))

    def forward(self, x):
        return x * self.weights

class MLP_OTO(nn.Module):
    def __init__(self,feature_number):
        super(MLP_OTO, self).__init__()
        self.OneToOneLayer = OneToOneLayer(feature_number)
        self.seq = nn.Sequential(
            nn.Linear(feature_number, 2*feature_number),
            nn.ReLU(),
            nn.Linear(2*feature_number, 2*feature_number),
            nn.ReLU(),
            # nn.Linear(2*feature_number, 2*feature_number),
            # nn.ReLU(),
            nn.Linear(2*feature_number, 1),
        )
    def forward(self, x):
        x = self.OneToOneLayer(x)
        x = self.seq(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,feature_number):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(feature_number, 2*feature_number),
            nn.ReLU(),
            nn.Linear(2*feature_number, 2*feature_number),
            nn.ReLU(),
            # nn.Linear(2*feature_number, 2*feature_number),
            # nn.ReLU(),
            nn.Linear(2*feature_number, 1),
        )
    def forward(self, x):
        x = self.seq(x)
        return x

class VGG_contrast(nn.Module):
    def __init__(self):
        super(VGG_contrast, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 51, padding = 25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 8, 51, padding = 25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, 51, padding = 25),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 51, padding = 25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32 * 61, 1)
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        representation = x.view(-1, 1, 32 * 61)  # flatten the tensor

        output = self.fc1(representation)

        return output, representation
    

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 51, padding = 25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 8, 51, padding = 25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, 51, padding = 25),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 51, padding = 25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 51, padding = 25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32 * 61, 10)
        )

        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(16, -1, 32 * 61)  # flatten the tensor

        x = self.fc1(x)
        output = self.out(x)

        return output
    
   

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(7, 1)
    def forward(self, x):
        x = self.fc(x)
        return x
    
# # 5.2
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),    
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# 5.8
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Linear(5, 5),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# 5.1
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# 3.4
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),    
#         nn.Linear(5, 5),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(), 
#         nn.Linear(5, 5),
#         nn.Sigmoid(),     
#         nn.Linear(5, 5),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# 5.5
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Linear(5, 5),    
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# 6.2
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),    
#         nn.Sigmoid(),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# # 7.4
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

# # 10.3
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Linear(5, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x
    
    
# # 7.9
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.ReLU(),
#         nn.Linear(5, 5),
#         nn.ReLU(),
#         nn.Linear(5, 5),    
#         nn.ReLU(),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x
    
    
# # 17.3
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5), 
#         nn.Sigmoid(),
#         nn.Linear(5, 3),
#         nn.Sigmoid(),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

if __name__ == "__main__":
    model = VGG()
    input = torch.ones((16, 1, 1000))
    writer = SummaryWriter("./log")
    writer.add_graph(model, input)
    writer.close()