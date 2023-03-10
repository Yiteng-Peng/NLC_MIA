import torch
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet5(pretrained=False, mode_path=None, device="cpu"):
    if pretrained:
        if "_@s" in mode_path:
            model = LeNet5().to(device)
            model.load_state_dict(torch.load(mode_path, map_location=device))
        elif "_@m" in mode_path:
            model = torch.load(mode_path, map_location=device)
    else:
        model = LeNet5().to(device)

    return model
