import torch.nn as nn
import torch.nn.functional as F


class PlayerCNN(nn.Module):

    def __init__(self):
        super(PlayerCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=11, out_channels=160, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=96, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=1)
        self.avgpool = nn.AvgPool1d(kernel_size=6)
        self.dense1 = nn.Linear(96, 256)
        self.dense2 = nn.Linear(256, 199)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x
