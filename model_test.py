import torch
import models
from datasets.loader import data_loader

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
data_name  = "MNIST"
data_mode  = "test"
loader_tag = False

# load
model_name = "Lenet5"
load_mode = "s"
load_path = "pretrained/%s.pt" % (model_name + "_@" + load_mode)


def test(model, test_data, test_label):
    with torch.no_grad():
        pred = model(test_data)
        acc_test = (pred.argmax(dim=1) == test_label).float().mean().item()
        return acc_test


if __name__ == "__main__":
    cur_model = models.lenet5(pretrained=True, mode_path=load_path)

    # data get
    test_data, test_label = data_loader(data_name, data_mode, loader_tag)
    test_data, test_label = test_data.to(DEVICE), test_label.to(DEVICE)

    cur_acc = test(cur_model, test_data, test_label)
    print(cur_acc)