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

# model
MODEL_NAME = "vgg11_bn"

#####################################################################################
# save
SAVE_MODE = "s"     # 'm' for origin model, 's' for state_dict
SAVE_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + SAVE_MODE)

# train model
TRAIN_MODEL = models.vgg11_bn(device=DEVICE)
OPTIMIZER = optim.Adam(TRAIN_MODEL.parameters())
LOSS_FUNC = nn.CrossEntropyLoss()
EPOCHS = 10
BATCH_SIZE = 16
SHUFFLE = True

#####################################################################################
# load
LOAD_MODE = "s"
# LOAD_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + LOAD_MODE)

# test model
# TEST_MODEL = models.lenet5(pretrained=True, mode_path=LOAD_PATH, device=DEVICE)
