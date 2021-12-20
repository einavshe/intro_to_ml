import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision import datasets
from trainer import Trainer
from models import ModelAB, ModelC, ModelD1, ModelD2, ModelE, ModelF, BestModel
from plots import plot


models_params = {
    "ModelA": {"lr":0.01},
    "ModelC": {"lr": 0.001, "dropouts_p": [0.3, 0.1], "Test_acc": 85},
    "ModelD1": {"lr":0.01, "Test_acc": 87.5},
    "ModelD2": {"lr":0.01, "Test_acc": 86.82}
}

np.random.seed(2021)
# models_l = [ModelAB, ModelAB, ModelC, ModelD1, ModelD2, ModelE, ModelF]
models_l = [BestModel]

# todo: make sure that the batch norm in model D is well implemented
#  ( maybe we need to separate between the fc and the relu)
# todo: extract calc acuuracy function from test and use it also for train
    

def experiments(train_loader, test_loader,x_test, out_path):
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    orig_test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True,
                              transform=test_transforms), batch_size=64, shuffle=False)
    for i, model in enumerate(models_l):
        model_n = model.__name__
        # optim_type = "SGD" if i == 0 else "Adam"
        optim_type = "Adam"
        num_epochs=40
        # for lr in [2e-1, 2e-2, 2e-3, 2e-4]:
        for lr in [0.001]:
            t = Trainer(lr, optim_type, train_loader, test_loader,num_epochs=num_epochs)
            m = model(784)
            train_losses, val_losses, train_accs, val_accs = t.train(m)
            _, test_acc = t.test(m, orig_test_loader, load_best=True)
            print(f"{i}\t{model_n}\ttest acc:{test_acc}")
            plot(range(num_epochs), [train_losses, val_losses],["train", "val"], "losses", f"{i}{model_n}_{lr}_losses.png")
            plot(range(num_epochs), [train_accs, val_accs],["train", "val"], "avg accuracy", f"{i}{model_n}_{lr}_accs.png")
            print("lr: ", lr)
    output = m(torch.from_numpy(x_test / 255.).float())
    preds = output.max(1, keepdim=True)[1].numpy()
    np.savetxt(out_path, preds)


def z_score(train):
    m = np.mean(train)
    dev = np.std(train)
    norma_train = (train - m)/ dev
    return norma_train


def get_loaders(train_x, train_y):
    num_samples = train_y.shape[0]
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)
    num_train_saples = int(0.8 * num_samples)
    train_idxs = idxs[:num_train_saples]
    test_idxs = idxs[num_train_saples:]
    train_dataset = TensorDataset(torch.from_numpy(train_x[train_idxs]/255.).float(),
                                  torch.from_numpy(train_y[train_idxs]).long(), )
    test_dataset = TensorDataset(torch.from_numpy(train_x[test_idxs]/255.).float(),
                                 torch.from_numpy(train_y[test_idxs]).long(), )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def ex4(x_train, y_train, x_test, out_path):
    # train_transforms = transforms.Compose([
    #     transforms.ToTensor(), ])
    # # transforms.Normalize((0.1307,), (0.3081,))])

    train_loader, test_loader = get_loaders(x_train, y_train)
    experiments(train_loader, test_loader,  x_test, out_path)

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
