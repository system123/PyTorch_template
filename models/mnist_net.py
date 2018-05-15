import torch
from torch import nn
import torch.nn.functional as F

class MNISTNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.conv2_drop = nn.Dropout2d()
        self.fc1_drop = nn.Dropout() 

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = self.conv2_drop(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)

        return F.log_softmax(x)
