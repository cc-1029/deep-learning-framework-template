import torch
from torch import nn
from torch.nn import functional as F


class PtMnistModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, 3)

        self.fc1 = nn.Linear(21632, 128)
        self.fc2 = nn.Linear(128, args.num_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = torch.reshape(x, (-1, 21632))
        print(x.shape)
        x = F.relu(self.fc1(x))
        return self.fc2(x)