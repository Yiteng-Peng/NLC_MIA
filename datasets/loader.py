import os

from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

# you could add your own dataset loader into data_loader
# remember it should return an iterable class

DATASET_PATH = "E:/PengYiteng/NLC_MIA/datasets"


def _CIFAR10(mode):
    if mode == "train":
        train = datasets.CIFAR10(os.path.join(DATASET_PATH, 'data'), download=True, train=True)
        X_train = train.data.unsqueeze(1) / 255.0  # 归一化
        y_train = train.targets
        return X_train, y_train
    elif mode == "test":
        test = datasets.CIFAR10(os.path.join(DATASET_PATH, 'data'), download=True, train=False)
        X_test = test.data.unsqueeze(1) / 255.0  # 归一化
        y_test = test.targets
        return X_test, y_test
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


def _CIFAR100(mode):
    if mode == "train":
        train = datasets.CIFAR100(os.path.join(DATASET_PATH, 'data'), download=True, train=True)
        X_train = train.data.unsqueeze(1) / 255.0  # 归一化
        y_train = train.targets
        return X_train, y_train
    elif mode == "test":
        test = datasets.CIFAR100(os.path.join(DATASET_PATH, 'data'), download=True, train=False)
        X_test = test.data.unsqueeze(1) / 255.0  # 归一化
        y_test = test.targets
        return X_test, y_test
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


def _MINST(mode):
    if mode == "train":
        train = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=True)
        X_train = train.data.unsqueeze(1)/255.0 # 归一化
        y_train = train.targets
        return X_train, y_train
    elif mode == "test":
        test = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=False)
        X_test = test.data.unsqueeze(1)/255.0 # 归一化
        y_test = test.targets
        return X_test, y_test
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


# 输出所有数据和标签的组合，在这里面进行归一化或者整理数据等操作
# name：数据集的名字 mode：模式，只有train和test
def datasets(name, mode):
    if mode != "train" and mode != "test":
        print("Wrong mode, mode should be train or test")

    if name == 'CIFAR10':
        return _CIFAR10(mode)
    elif name == 'CIFAR100':
        return _CIFAR100(mode)
    elif name == 'MNIST':
        return _MINST(mode)
