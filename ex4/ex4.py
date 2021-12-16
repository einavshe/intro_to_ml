import sys
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision import datasets
from trainer import Trainer
from models import ModelAB, ModelC, ModelD1, ModelD2, ModelE, ModelF
from plots import plot

np.random.seed(2021)
models_l = [ModelAB, ModelAB, ModelC, ModelD1, ModelD2, ModelE, ModelF]


# todo: make sure that the batch norm in model D is well implemented
#  ( maybe we need to separate between the fc and the relu)
# todo: extract calc acuuracy function from test and use it also for train



#modelA - change 1000 to 100 in FirstNet
#modelB - change 1000 to 100 in FirstNet and SGD to ADAM
#modelC- droput after(!) every x in forward  (modelA):
#         m = nn.Droput(p=0.2)   - p between 0.4 to 0.6
#         input = torch.randn(20,16)  - x
#         output = m(input)
#modelD - normalization before(!) ReLU (modelA) - nn.BatchNorm1d(size of fc)
#modelE - 
#         self.fc0 = nn.Linear(image_size, 128)
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, 10)
#         self.fc3 = nn.Linear(10, 10)
#         self.fc4 = nn.Linear(10, 10)

# modelF- change in modelE - ReLU to sigmoid


    

def experiments(train_loader, test_loader):
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    orig_test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True,
                              transform=test_transforms), batch_size=64, shuffle=False)
    for i, model in enumerate(models_l):
        model_n = model.__name__
        optim_type = "Adam" if i != 0 else "SGD"
        for lr in [2e-1, 2e-2, 2e-3, 2e-4]:
            t = Trainer(lr, optim_type, train_loader, test_loader)
            m = model(784)
            train_losses, val_losses, train_accs, val_accs = t.train(m)
            _, test_acc = t.test(m, orig_test_loader)
            print(f"{i}\t{model_n}\ttest acc:{test_acc}")
            plot(range(10), [train_losses, val_losses],["train", "val"], "losses", f"{model_n}_losses.png")
            plot(range(10), [train_accs, val_accs],["train", "val"], "avg accuracy", f"{model_n}_accs.png")


def get_loaders(train_x, train_y):
    num_samples = train_y.shape[0]
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)
    num_train_saples = int(0.8 * num_samples)
    train_idxs = idxs[:num_train_saples]
    test_idxs = idxs[:num_train_saples]
    train_dataset = TensorDataset(torch.from_numpy(train_x[train_idxs]).float(),
                                  torch.from_numpy(train_y[train_idxs]).int(), )
    test_dataset = TensorDataset(torch.from_numpy(train_x[test_idxs]).float(),
                                 torch.from_numpy(train_y[test_idxs]).int(), )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def ex4(x_train, y_train, x_test, out_path):
    train_transforms = transforms.Compose([
        transforms.ToTensor(), ])
    # transforms.Normalize((0.1307,), (0.3081,))])

    train_loader, test_loader = get_loaders(x_train, y_train)
    experiments(train_loader, test_loader)

    # todo t = Trainer(lr, optim_type, train_loader, test_loader)
    # t.train(my_best_model)
    return


def main(args):
    x_train_p, y_train_p, x_test_p, out_path = args[0], args[1], args[2], args[3]
    start = time.time()
    x_train, y_train, x_test = np.loadtxt(x_train_p), np.loadtxt(y_train_p, dtype=int), np.loadtxt(x_test_p)
    ex4(x_train, y_train, x_test, out_path)
    print(f"run time: {time.time() - start}")


if __name__ == "__main__":
    main(sys.argv[1:])
