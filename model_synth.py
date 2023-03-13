import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.loader import dataset
from config import *
from mia.synthesis.data_synthesis import synthesize_batch

# 用的就是训练好的模型，来尝试生成数据，所以这个script中不需要自己的宏定义
SYNTH_MODEL = MODEL(pretrained=True, mode_path=LOAD_PATH, device=DEVICE, num_classes=NUM_CLASS)

if __name__ == "__main__":
    for i in range(NUM_CLASS):
        cls_data = synthesize_batch(MIA_DATA_SHAPE, SYNTH_MODEL, i, 
                                    10, k_max=128, dtype="float")
        print(cls_data.shape)
        exit(0)