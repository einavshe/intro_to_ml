import torch
from torch import nn
import torch.nn.functional as F


# model A with SGD, model B with Adam
# model C, D with Adam


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

class M5(BaseModel):
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


class ModelAB(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class ModelC(BaseModel):

    def __init__(self, image_size, num_cls=30, dropouts_p=[0.5, 0.1]):
        super(BaseModel, self).__init__()
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_cls)
        self.dropouts_p = dropouts_p

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, self.dropouts_p[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropouts_p[1])
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class ModelD1(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(ModelD1, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_cls)
        self.fc2_bn = nn.BatchNorm1d(num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu((self.fc1_bn(self.fc1(x))))
        x = F.relu(self.fc2_bn((self.fc2(x))))
        return F.log_softmax(x)


class ModelD2(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(ModelD2, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_cls)
        self.fc2_bn = nn.BatchNorm1d(num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0_bn(F.relu(self.fc0(x)))
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        return F.log_softmax(x)


class ModelE(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(x)


class ModelF(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return F.log_softmax(x)


class BestModel(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(BestModel, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 1024)
        self.fc0_bn = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x)


class BestB(BaseModel):

    def __init__(self, image_size, num_cls=30):
        super(BaseModel, self).__init__()
        super(BestB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 256)
        self.fc0_bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc2_bn = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, num_cls)
        self.init_xavier()

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x)


model = ModelAB(image_size=28 * 28)
