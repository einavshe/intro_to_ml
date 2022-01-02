import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def init_xavier(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class BestModel(BaseModel):
    def __init__(self, n_input=1, n_output=30, stride=1, n_channel=32):
        super(BaseModel, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=5, stride=stride)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(2 * n_channel, n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(n_channel)
        self.pool4 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
