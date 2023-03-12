from torch.utils.data import DataLoader
from tqdm import trange

from config import *
from datasets.loader import dataset

TRAIN_MODEL = MODEL(device=DEVICE, num_classes=NUM_CLASS)
# TRAIN_MODEL = models.vgg11_bn(pretrained=True, mode_path=SAVE_PATH, device=DEVICE, num_classes=NUM_CLASS)
OPTIMIZER = optim.Adam(TRAIN_MODEL.parameters())

def save(model, mode, path):
    if mode == "s":
        torch.save(model.state_dict(), path)
    elif mode == "m":
        torch.save(model, path)
    else:
        print("unknown mode, save as origin mode in ", path)
        torch.save(model, path)


def train(model, data, loss_func, optimizer):
    trainloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    for _ in trange(EPOCHS):
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(loss)


if __name__ == "__main__":
    print(type(TRAIN_MODEL))
    # get data
    data = dataset(DATA_NAME, TRAIN_MODE)

    # train the model
    train(TRAIN_MODEL, data, LOSS_FUNC, OPTIMIZER)

    # save model
    save(TRAIN_MODEL, SAVE_MODE, SAVE_PATH)
