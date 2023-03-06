import os

from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

# you could add your own dataset loader into data_loader
# remember it should return an iterable class

DATASET_PATH = "E:/PengYiteng/NLC_MIA/datasets"

def _MINST_loader(batch_size, shuffle, mode, loader):
    if mode == "train":
        train = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=True)
        X_train = train.data.unsqueeze(1)/255.0 # 归一化
        y_train = train.targets
        if loader:
            return DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)
        else:
            return X_train, y_train
    elif mode == "test":
        test = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=False)
        X_test = test.data.unsqueeze(1)/255.0 # 归一化
        y_test = test.targets
        if loader:
            return DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=shuffle)
        else:
            return X_test, y_test


def data_loader(dataset_name, mode, loader:bool, batch_size=1, shuffle=False):
    if mode != "train" and mode != "test":
        print("Wrong mode, mode should be train or test")

    if dataset_name == 'ImageNet':
        pass
    elif dataset_name == 'CIFAR10':
        pass
    elif dataset_name == 'CIFAR100':
        pass
    elif dataset_name == 'MNIST':
        return _MINST_loader(batch_size, shuffle, mode, loader)