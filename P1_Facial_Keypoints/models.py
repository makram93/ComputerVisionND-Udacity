## TODO: define the convolutional neural network architecture

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 136)

        # Initialize the weights/bias with 0
        self.fc2.weight.data.zero_()

    def forward(self, xs):
        # transform the input
        x = self.conv1(xs)
        x = F.relu(self.bn1(F.max_pool2d(x, 2)))
        x = self.conv2(x)
        x = F.relu(self.bn2(F.max_pool2d(x, 2)))
        x = self.conv3(x)
        x = F.relu(self.bn3(F.max_pool2d(x, 2)))
        x = self.conv4(x)
        x = F.relu(self.bn4(F.max_pool2d(x, 2)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x