import torch
from torch import nn
from torch import optim
from tqdm import tqdm, trange

import models

# setting training parameter
model_name = "Lenet5"
cur_model = model.lenet.LeNet5()
optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

save_mode = "m" # 'm' for origin model, 's' for state_dict
save_path = "pretrained/%s.pt" % model_name + "_" + save_mode


# save DL model as save_mode in save_path
# SIDE EFFECT: output the file in EXISTED folder
# model: torch model: model for save
# mode: str: s for state_dict, m for origin model, default for origin
def save(model, mode):
    if mode == "s":
        torch.save(model.state_dict(), save_path)
    elif mode == "m":
        torch.save(model, save_path)
    else:
        print("unknown mode, save as origin mode in ", save_path)
        torch.save(model, save_path)


# train model on the dataset trainloader
# model: torch.model: waiting for training
# trainloader: enumeration: dataset for model (X, y) X as data & y as label
# epoch: int: epoch for train
def train(model, trainloader, epoch, loss_func, optimizer):
    for epoch in trange(10):
        for X, y in trainloader:
            pred = model(X)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train(cur_model, , 10, , ,)