import torch
import models
from datasets.loader import data_loader

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model_name = "Lenet5"
cur_model  = models.LeNet5()

# dataset
data_name  = "MNIST"
data_mode  = "test"
loader_tag = False

# load
load_mode = "s"
load_path = "pretrained/%s.pt" % (model_name + "_" + load_mode)


# save DL model as save_mode in save_path
# SIDE EFFECT: output the file in EXISTED folder
# model: torch model: model for save
# mode: str: s for state_dict, m for origin model, default for origin
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


def test(model, test_data, test_label):
    with torch.no_grad():
        pred = model(test_data)
        acc_test = (pred.argmax(dim=1) == test_label).float().mean().item()
        return acc_test


if __name__ == "__main__":
    cur_model = load(cur_model, load_mode)

    # data get
    test_data, test_label = data_loader(data_name, data_mode, loader_tag)
    test_data, test_label = test_data.to(DEVICE), test_label.to(DEVICE)

    cur_acc = test(cur_model, test_data, test_label)
    print(cur_acc)