from datasets.loader import dataset
from conifg import *

def test(model, test_data):
    # MNIST
    data, label = test_data.tensors[0].to(DEVICE), test_data.tensors[1].to(DEVICE)
    # others
    # data, label = torch.Tensor(test_data.data).to(DEVICE), torch.Tensor(test_data.targets).to(DEVICE)

    with torch.no_grad():
        pred = model(data)
        acc_test = (pred.argmax(dim=1) == label).float().mean().item()
        return acc_test


if __name__ == "__main__":
    # get model
    cur_model = TEST_MODEL

    # data get
    test_data = dataset(DATA_NAME, TEST_MODE)

    # model test
    cur_acc = test(cur_model, test_data)
    print(cur_acc)
