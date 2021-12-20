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


class ModelD1(nn.Module):

    def __init__(self,image_size, num_cls=10):
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


class ModelD2(nn.Module):

    def __init__(self,image_size, num_cls=10):
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


class ModelE(nn.Module):

    def __init__(self,image_size, num_cls=10):
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


class ModelF(nn.Module):

    def __init__(self,image_size, num_cls=10):
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




class BestModel(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(BestModel, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_cls)
        self.dropouts_p = [0.63, 0.35, 0.42]

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = F.relu(self.fc0(x))
        x = F.dropout(x, self.dropouts_p[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropouts_p[1])
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropouts_p[2])
        x = self.fc3(x)

        return F.log_softmax(x)




class BestB(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(BestB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 256)
        self.fc0_bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc2_bn = nn.BatchNorm1d(50)
        self.fc3= nn.Linear(50, num_cls)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x)


class BestCnn(nn.Module):

    def __init__(self,image_size, num_cls=10):
        super(BestCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)



    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        x = x.view((x.size()[0],1,28,28))
        # conv 1
        t = self.conv1(x)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return F.log_softmax(t)



model = ModelAB(image_size=28*28)
