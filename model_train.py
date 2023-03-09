import torch
from torch import nn, optim
from tqdm import tqdm, trange

import models
from datasets.loader import data_loader

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting training parameter
# model
model_name = "Lenet5"
cur_model  = models.LeNet5()
optimizer  = optim.Adam(cur_model.parameters())
loss_func  = nn.CrossEntropyLoss()
epoch      = 10

# dataset
data_name  = "MNIST"
data_mode  = "train"
loader_tag = True
batch_size = 256
shuffle    = True

# save
save_mode = "s" # 'm' for origin model, 's' for state_dict
save_path = "pretrained/%s.pt" % (model_name + "_" + save_mode)


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
    model.to(DEVICE)

    for epoch in trange(10):
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model


if __name__ == "__main__":
    trainloader = data_loader(data_name, data_mode, loader_tag, batch_size, shuffle)
    cur_model = train(cur_model, trainloader, epoch, loss_func, optimizer)
    save(cur_model, save_mode)