import torch
from torch import nn, optim
import models

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# dataset
DATA_NAME = "CIFAR10"
TRAIN_MODE = "train"
TEST_MODE = "test"
COV_MODE = TEST_MODE

# model
MODEL_NAME = "vgg19_bn"

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
TEST_BATCH = 1000

#####################################################################################
# coverage setting
COV_NAME = "NC"
COV_FILE_PATH = "./cov_data"
COV_OUTPUT_MODE = "txt"
COV_FILE_NAME = "%s-%s-%s-%s" % (MODEL_NAME, DATA_NAME, COV_OUTPUT_MODE, COV_NAME)
# 扩展名在写入的时候添加

# input size
INPUT_CHANNEL = 3
INPUT_SIZE    = 32
