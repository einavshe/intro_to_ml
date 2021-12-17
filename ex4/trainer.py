import numpy as np
import torch.nn.functional as F
from torch import optim


class Trainer:
    def __init__(self, lr, optimizer_type, train_loader, test_loader):
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.train_loader = None
        self.num_epochs = 10
        self.train_loader = train_loader
        self.test_loader = test_loader
        pass

    def epoch(self, epoch_idx, model):
        correct = 0
        overall_loss = 0
        model.train()
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).cpu().sum()
            loss = F.nll_loss(output, labels)
            overall_loss += F.nll_loss(output, labels, size_average=False).item()
            loss.backward()
            self.optimizer.step()
        len_dataset = len(self.train_loader.dataset)
        return overall_loss / len_dataset, 100. * correct / len_dataset

    def train(self, model):
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        if self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
        for i in range(self.num_epochs):
            loss, accuracy = self.epoch(i, model)
            t_loss, t_accuracy = self.test(model)
            train_losses.append(loss)
            test_losses.append(t_loss)
            train_accs.append(accuracy)
            test_accs.append(t_accuracy)
        return train_losses, test_losses, train_accs, test_accs

    def test(self, model, test_loader=None):
        model.eval()
        test_loss = 0
        correct = 0
        if test_loader is None:
            test_loader = self.test_loader
        len_dataset = len(test_loader.dataset)
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
        test_loss /= len_dataset
        accuracy = 100. * correct / len_dataset
        print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
            test_loss, correct, len_dataset,
            accuracy))
        return test_loss, accuracy
