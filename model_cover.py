import torch
import coverage
from torch.utils.data import DataLoader

from coverage import tool
from datasets.loader import dataset
from config import *

# model
MIA_MODEL = models.vgg13_bn(device=DEVICE, num_classes=10)

if __name__ == "__main__":
    #1. Get layer size of model
    input_shape = (1, INPUT_CHANNEL, INPUT_SIZE, INPUT_SIZE)
    cur_model = MIA_MODEL
    
    random_input = torch.randn(input_shape).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(cur_model, random_input)

    #2. Calculation
    criterion = coverage.MDSC(cur_model, layer_size_dict, hyper=0.1, num_class=10, min_var=0)   
    # 通过测试数据算coverage
    cov_data = dataset(DATA_NAME, COV_MODE)
    # 如果不是用loader的形式，而是数据流的形式就用这种方式
    cov_loader = DataLoader(torch.utils.data.Subset(cov_data, range(10)))
    # 有一些覆盖率指标是不需要利用训练数据的，就可以不用build
    criterion.build(cov_loader)
    # for data, _ in cov_loader:
    #     data = data.to(DEVICE)
    #     criterion.step(data)

    criterion.assess(cov_loader)

    # #3. Result
    cov = criterion.current
    print(cov)
    
    # 测MIA攻击的时候如果要修改就在这边弄
    # criterion.mia_set(MIA_FILE_PATH, MIA_FILE_NAME, MIA_MODE)
    # criterion.mia_assess(test_loader)