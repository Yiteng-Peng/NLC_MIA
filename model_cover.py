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
data_mode  = "test"
loader_tag = True

# input size
input_channel = 1
input_size    = 28

if __name__ == "__main__":
    #1. Get layer size of model
    input_size = (1, input_channel, input_size, input_size)
    cur_model = cur_model.to(DEVICE)
    random_input = torch.randn(input_size).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(cur_model, random_input)

    #2. Calculation
    criterion = coverage.NC(cur_model, layer_size_dict, hyper=0.25)
    
    # 有一些覆盖率指标是不需要利用训练数据的，就可以不用build
    # criterion.build(train_loader)
    
    # 通过测试数据算coverage
    test_loader = data_loader(data_name, data_mode, loader_tag)
    criterion.assess(test_loader)
    
    # 如果不是用loader的形式，而是数据流的形式就用这种方式
    # for data in data_stream:
    #     criterion.step(data)

    # #3. Result
    cov = criterion.current
    print(cov)