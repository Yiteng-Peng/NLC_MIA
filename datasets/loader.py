import os

import torchvision
from torchvision import datasets
from torch.utils.data import TensorDataset

# 这个目录设置的是主目录下的相对路径
DATASET_PATH = "./datasets"

def _CIFAR10(mode):
    if mode == "train":
        return datasets.CIFAR10(os.path.join(DATASET_PATH, 'data'), transform=torchvision.transforms.ToTensor(),
                                download=True, train=True)
    elif mode == "test":
        return datasets.CIFAR10(os.path.join(DATASET_PATH, 'data'), transform=torchvision.transforms.ToTensor(),
                                download=True, train=False)
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


def _CIFAR100(mode):
    if mode == "train":
        return datasets.CIFAR100(os.path.join(DATASET_PATH, 'data'), transform=torchvision.transforms.ToTensor(),
                                 download=True, train=True)
    elif mode == "test":
        return datasets.CIFAR100(os.path.join(DATASET_PATH, 'data'), transform=torchvision.transforms.ToTensor(),
                                 download=True, train=False)
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


def _MINST(mode):
    if mode == "train":
        train = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=True)
        X_train = train.data.unsqueeze(1)/255.0 # 归一化
        y_train = train.targets
        return TensorDataset(X_train, y_train)
    elif mode == "test":
        test = datasets.MNIST(os.path.join(DATASET_PATH, 'data'), download=True, train=False)
        X_test = test.data.unsqueeze(1)/255.0 # 归一化
        y_test = test.targets
        return TensorDataset(X_test, y_test)
    else:
        raise NotImplementedError("Wrong mode, mode should be train or test")


# 输出所有数据和标签的组合，在这里面进行归一化或者整理数据等操作
# name：数据集的名字 mode：模式，只有train和test
def dataset(name, mode):
    if mode != "train" and mode != "test":
        print("Wrong mode, mode should be train or test")

    if name == 'CIFAR10':
        return _CIFAR10(mode)
    elif name == 'CIFAR100':
        return _CIFAR100(mode)
    elif name == 'MNIST':
        return _MINST(mode)
