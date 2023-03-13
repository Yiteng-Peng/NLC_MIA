import torch
from torch import nn, optim
import models

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# dataset
DATA_NAME = "CIFAR10"
NUM_CLASS = 10

TRAIN_MODE = "train"
TEST_MODE = "test"
COV_MODE = TEST_MODE

# model
MODEL_NAME = "resnet34"
MODEL = models.resnet34

# train model
LOSS_FUNC = nn.CrossEntropyLoss()
EPOCHS = 20
BATCH_SIZE = 256
SHUFFLE = True

################################# train model #####################################
# save
SAVE_MODE = "s"     # 'm' for origin model, 's' for state_dict
SAVE_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + SAVE_MODE)

################################ test model ########################################
# load
LOAD_MODE = "s"
LOAD_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + LOAD_MODE)

# test model
TEST_BATCH = 1000

################################### coverage #####################################
# coverage setting
COV_NAME = "NC"
COV_FILE_PATH = "./cov_data"
COV_OUTPUT_MODE = "txt"
COV_FILE_NAME = "%s-%s-%s-%s" % (MODEL_NAME, DATA_NAME, COV_OUTPUT_MODE, COV_NAME)
# 扩展名在写入的时候添加

# input size
INPUT_CHANNEL = 3
INPUT_SIZE    = 32

################################### mia attack #####################################
NUM_SHADOW = 3
# target model
MIA_MODE = "s"
MIA_MODEL_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + MIA_MODE)

# dataset save path
MIA_DATA_MODE = "split"  # split: 从原始训练集切分的数据 syn: 从模型中生成的影子数据 
MIA_DATA_PATH = "mia_data/dataset/%s.data" % (MODEL_NAME + "_" + DATA_NAME + "_@" + MIA_DATA_MODE)

# shadow model
MIA_SHADOW_EPOCH = 10
MIA_SHADOW_BATCH = 32
MIA_SHADOW_PATH = "mia_data/results/%s.npy" % (MODEL_NAME + "_" + str(NUM_SHADOW))
MIA_RESULT_BATCH = 1000

# attack model
MIA_ATTACK_EPOCH = 10
MIA_ATTACK_BATCH = 1000
MIA_ATTACK_PATH = "mia_data/pretrained/%s" % (MODEL_NAME + "_" + DATA_NAME + "_@s_all.pt")
# 扩展名在写入的时候添加，因为要分片
