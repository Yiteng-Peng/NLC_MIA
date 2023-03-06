import torch
import coverage
import models
from coverage import tool
from datasets.loader import data_loader

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model_name = "Lenet5"
cur_model  = models.LeNet5()

# dataset
data_name  = "MNIST"
data_mode  = "train"
loader_tag = True

# load
load_mode = "s"
load_path = "pretrained/%s.pt" % (model_name + "_" + load_mode)

# input size
input_channel = 1
input_size    = 28

# mia setting
mia_file_path = "./mia_data"
mia_mode      = "txt"
mia_file_name = "%s-%s" % (model_name, data_name)       # 扩展名在写入的时候添加

def load(model, mode):
    if mode == "s":
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        return model
    elif mode == "m":
        model = torch.load(load_path, map_location=DEVICE)
        return model
    else:
        print("unknown mode")

if __name__ == "__main__":
    #1. Get layer size of model
    input_size = (1, input_channel, input_size, input_size)
    cur_model = load(cur_model, load_mode)
    random_input = torch.randn(input_size).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(cur_model, random_input)

    #2. Calculation
    criterion = coverage.NC(cur_model, layer_size_dict, hyper=0.25)
    
    # 有一些覆盖率指标是不需要利用训练数据的，就可以不用build
    # criterion.build(train_loader)
    
    # 通过测试数据算coverage
    test_loader = data_loader(data_name, data_mode, loader_tag)

    criterion.mia_set(mia_file_path, mia_file_name, mia_mode)
    criterion.mia_assess(test_loader)
    
    # 如果不是用loader的形式，而是数据流的形式就用这种方式
    # for data in data_stream:
    #     criterion.step(data)

    # #3. Result
    # cov = criterion.current
    # print(cov)