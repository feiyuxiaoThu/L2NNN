import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(config.in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.num_dense = (config.im_size // 4 - 3)**2 * 20
        self.fc1 = nn.Linear(self.num_dense, 50)
        self.fc2 = nn.Linear(50, config.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_dense)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class WideConvNet(nn.Module):
    def __init__(self, config):
        super(WideConvNet, self).__init__()
        self.conv1 = nn.Conv2d(config.in_channels, 32, 5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.num_dense = (config.im_size // 4) ** 2 * 64
        self.fc1 = nn.Linear(self.num_dense, 1024, bias=True)
        self.fc2 = nn.Linear(1024, config.num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_dense)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
