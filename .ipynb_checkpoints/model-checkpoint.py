import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

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
            nn.Linear(2*feature_number, 1),
        )
    def forward(self, x):
        x = self.OneToOneLayer(x)
        x = self.seq(x)
        return x
    
# class MLP(nn.Module):
#     def __init__(self,feature_number):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(feature_number, 2*feature_number),
#             nn.ReLU(),
#             nn.Linear(2*feature_number, 2*feature_number),
#             nn.ReLU(),
#             # nn.Linear(2*feature_number, 2*feature_number),
#             # nn.ReLU(),
#             nn.Linear(2*feature_number, 1),
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