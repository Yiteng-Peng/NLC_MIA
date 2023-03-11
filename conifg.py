import torch
from torch import nn, optim
import models

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# dataset
DATA_NAME = "CIFAR100"
TRAIN_MODE = "train"
TEST_MODE = "test"

# model
MODEL_NAME = "vgg13_bn"

#####################################################################################
# save
SAVE_MODE = "s"     # 'm' for origin model, 's' for state_dict
SAVE_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + SAVE_MODE)

# train model
LOSS_FUNC = nn.CrossEntropyLoss()
EPOCHS = 2
BATCH_SIZE = 256
SHUFFLE = True

#####################################################################################
# load
LOAD_MODE = "s"
LOAD_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + LOAD_MODE)

# test model
TEST_BATCH = 10000
