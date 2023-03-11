import tqdm
import numpy as np
from torch.utils.data import DataLoader

from datasets.loader import dataset
from conifg import *

TEST_MODEL = models.vgg13_bn(pretrained=True, mode_path=LOAD_PATH, device=DEVICE, num_classes=100)

def test(model, test_data):
    # MNIST
    # data, label = test_data.tensors[0].to(DEVICE), test_data.tensors[1].to(DEVICE)
    # others
    # data, label = torch.Tensor(test_data.data).to(DEVICE), torch.Tensor(test_data.targets).to(DEVICE)
    
    # From: [batch_size, depth, height, width, channels]
    # To: [batch_size, channels, depth, height, width]
    # input = input.permute(0, 4, 1, 2, 3)

    with torch.no_grad():
        testloader = DataLoader(test_data, batch_size=TEST_BATCH)
        acc_list = []
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
            
        return np.mean(acc_list)


if __name__ == "__main__":
    print(type(TEST_MODEL))
    # get model
    cur_model = TEST_MODEL

    # data get
    test_data = dataset(DATA_NAME, TEST_MODE)

    # model test
    cur_acc = test(cur_model, test_data)
    print(cur_acc)
