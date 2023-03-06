import torch
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = DEVICE
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)

    def init_variable(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    # 通过每一个数据计算coverage，每一步的计算在step中
    def assess(self, data_loader):
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)

    # 首先算每一层的coverage，得到一个字典
    def step(self, data):
        cove_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain < 0:
            print("Neg")
        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        # 更新每一次的覆盖率，只增不减
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    # 计算覆盖率和上一次的差值
    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current