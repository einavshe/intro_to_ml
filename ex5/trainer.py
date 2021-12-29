import copy

import numpy as np
import torch.nn.functional as F
from torch import optim


class Trainer:
    def __init__(self, train_loader, test_loader, hyperparams_dict={}, save_path="best_model.pth"):
        self.lr = hyperparams_dict.get("lr", 0.01)
        self.optimizer = None
        self.scheduler = None
        self.num_epochs = hyperparams_dict.get("num_epochs", 10)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_val_accuracy = 0
        self.save_path = save_path

        #optimizer params
        self.w_decay = hyperparams_dict.get("w_decay", 0.00001)
        self.step_size = hyperparams_dict.get("step_size", 20)
        self.gamma = hyperparams_dict.get("gamma", 0.1)


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
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        for i in range(self.num_epochs):
            print(f"epoch {i}\n")
            loss, accuracy = self.epoch(i, model)
            t_loss, t_accuracy = self.test(model)
            self.scheduler.step()
            if t_accuracy > self.best_val_accuracy:
                model.save_model(self.save_path)
                self.best_val_accuracy = t_accuracy
            train_losses.append(loss)
            test_losses.append(t_loss)
            train_accs.append(accuracy)
            test_accs.append(t_accuracy)
        return train_losses, test_losses, train_accs, test_accs

    def test(self, model, test_loader=None, load_best=False):
        try:
            if load_best:
                model.load_model(self.save_path)
        except Exception as e:
            print(f"{self.save_path} doesnt exist\n{e}")
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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {}({:.2f} % )\n'.format(
            test_loss, correct, len_dataset,
            accuracy))
        return test_loss, accuracy

    def inference(self, model, test_loader):
        try:
            model.load_model(self.save_path)
        except Exception as e:
            print(f"{self.save_path} doesnt exist\n{e}")
        model.eval()
        test_loss = 0
        correct = 0
        preds = []
        if test_loader is None:
            test_loader = self.test_loader
        len_dataset = len(test_loader.dataset)
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            preds.extend(list(pred.numpy()))
        test_loss /= len_dataset
        accuracy = 100. * correct / len_dataset
        print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {}({:.2f} % )\n'.format(
            test_loss, correct, len_dataset,
            accuracy))
        return preds
