import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.transformer as transformer


class PlayerCNN(nn.Module):

    def __init__(self):
        super(PlayerCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=160, kernel_size=1)
        self.conv1_bn = nn.BatchNorm1d(160)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=96, kernel_size=1)
        self.conv2_bn = nn.BatchNorm1d(96)
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=1)
        self.conv3_bn = nn.BatchNorm1d(96)
        self.avgpool = nn.AvgPool1d(kernel_size=11)
        self.dense1 = nn.Linear(96, 256)
        self.dense1_bn = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 199)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # reshape so that channels are second axis in the tensor
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.dense1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x


class CNNTransformer(nn.Module):

    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.encoder_layer = transformer.TransformerEncoderLayer(d_model=32, nhead=1)
        self.encoder = transformer.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.avgpool = nn.AvgPool1d(kernel_size=11)
        self.dense1 = nn.Linear(32, 128)
        self.dense2 = nn.Linear(128, 199)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # learn embedding for transformer with conv1d layers
        x = x.permute(0, 2, 1)  # reshape so that channels are second axis in the tensor
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        # shape: (batch_size x channels x num_players) --> (num_players, batch_size, channels)
        x = x.permute(2, 0, 1)
        x = self.encoder(x)
        x = x.permute(1, 2, 0)
        x = self.avgpool(x)

        x = x.view(x.shape[0], x.shape[1])
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x
