import os
from pathlib import Path

from models import BestModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time

import numpy as np
import torch
from gcommand_loader import GCommandLoader
from torchvision import transforms
from torchvision import datasets

from trainer import Trainer
from plots import plot


hyper_params_dict = {
    "lr": 0.01,
    "num_epochs": 20,
    "w_decay": 0,
    "step_size": 20,
    "gamma": 0.1,
}


def experiments(train_loader, val_loader, test_loader):
    with open("log.txt", "a") as log_f:
        for num_epochs in [15]:#0, 20, 50]:
            for lr in [1e-2]:#, 1e-3, 1e-4]:
                print(lr, num_epochs)
                m = BestModel(n_input=1, n_output=30)
                step_size = 10
                t = Trainer(train_loader, val_loader, hyperparams_dict={"lr": lr, "num_epochs": num_epochs, "step_size":step_size},
                        save_path=f"{num_epochs}_{lr}_{step_size}_best_model.pth")
                train_losses, val_losses, train_accs, val_accs = t.train(m)
                log_f.write(f"lr: {lr},\tnum_e: {num_epochs},\tstep_s: {step_size},\tbest_acc: {t.best_val_accuracy}\n")
                plot_n = f"lr_{lr}_nume_{num_epochs}_step_s_{step_size}"
                plot(range(num_epochs), [train_losses, val_losses], ["train", "val"], "losses",
                     plot_n+"_loss.png")
                plot(range(num_epochs), [train_accs, val_accs], ["train", "val"], "avg accuracy",
                     plot_n+"_accs.png")

    preds = t.inference(m, test_loader)
    new_test_y = []
    classes = train_loader.dataset.classes
    files_names = [Path(path).name for path in test_loader.dataset.spects]
    for i, (f_name, y) in enumerate(zip(files_names, preds)):
        new_y = f"{f_name},{classes[int(y)]}"
        if i < len(preds) - 1:
            new_y = f"{new_y}\n"
        new_test_y.append(new_y)
    with open("test_y", "w") as f:
        f.writelines(new_test_y)


def get_loaders(data_p, batch_size=100):
    train_dataset = GCommandLoader(os.path.join(data_p, 'train'))
    val_dataset = GCommandLoader(os.path.join(data_p,  'valid'))
    test_dataset = GCommandLoader(os.path.join(data_p, 'test'), train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)

    return train_loader, val_loader, test_loader


def ex5(args):
    data_p = args[0] #'./data/'
    train_loader, val_loader,test_loader = get_loaders(data_p=data_p, batch_size=100)
    experiments(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    ex5(sys.argv[1:])
