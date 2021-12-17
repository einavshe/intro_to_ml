import torch
from torch import nn
import torch.nn.functional as F

# model A with SGD, model B with Adam
# model C, D with Adam


class ModelAB(nn.Module):

    def __init__(self,image_size, num_cls=10):
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


class ModelC(nn.Module):

    def __init__(self,image_size, num_cls=10, dropouts_p=[0.5, 0.1]):
        super(ModelAB, self).__init__()
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


class ModelD1(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(F.batch_norm(self.fc0(x)))
        x = F.relu(F.batch_norm((self.fc1(x))))
        x = F.relu(F.batch_norm(self.fc2(x)))
        return F.log_softmax(x)


class ModelD2(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_cls)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.batch_norm(F.relu(self.fc0(x)))
        x = F.batch_norm(F.relu(self.fc1(x)))
        x = F.batch_norm(F.relu(self.fc2(x)))
        return F.log_softmax(x)


class ModelE(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(ModelAB, self).__init__()
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


class ModelF(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(ModelAB, self).__init__()
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


model = ModelAB(image_size=28*28)
