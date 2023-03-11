from torch.utils.data import DataLoader
from tqdm import trange

from conifg import *
from datasets.loader import dataset

def save(model, mode):
    if mode == "s":
        torch.save(model.state_dict(), SAVE_PATH)
    elif mode == "m":
        torch.save(model, SAVE_PATH)
    else:
        print("unknown mode, save as origin mode in ", SAVE_PATH)
        torch.save(model, SAVE_PATH)


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
    # get data
    data = dataset(DATA_NAME, TRAIN_MODE)

    # train the model
    train(TRAIN_MODEL, data, LOSS_FUNC, OPTIMIZER)

    # save model
    save(TRAIN_MODEL, SAVE_MODE)
