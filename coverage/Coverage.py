import torch
from tqdm import tqdm
import os
import pickle

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = DEVICE
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)

        self.mia_tag = False

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

        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        # 更新每一次的覆盖率，只增不减
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    # 计算覆盖率和上一次的差值,或许可以考虑train和test覆盖率的变化情况?
    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current

    '''
    在研究MIA时，我们更关注单个数据的情况，因此这里给出单个数据的
    '''
    # 记录coverage的路径和格式
    # 文本记录 or 列表数组
    def mia_set(self, file_path, file_name, save_mode):
        if save_mode != "txt" and save_mode != "list":
            print("save_mode should txt or list")
            return

        self.mia_list = []
        self.mia_mode = save_mode == "txt"      # txt: True, list: False
        self.mia_tag = True
        self.mia_path = os.path.join(file_path, file_name)

    '''
    考虑到存储的体积问题，我们一万个数据存储一次，当然这个后期可以改
    '''
    def mia_assess(self, data_loader):
        if not self.mia_tag:
            print("please call mia_set first")
            return

        data_counter = 0
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.mia_step(data)

            data_counter += 1
            if data_counter == 10000:
                self.mia_save(data_counter)
                break

        self.mia_save("end")

    def mia_step(self, data):
        cove_dict = self.calculate(data)
        cover = self.coverage(cove_dict)

        if cover is not None:
            self.mia_list.append(cover)

    def mia_save(self, index):
        if len(self.mia_list) == 0:
            return

        if self.mia_mode:
            # True: txt
            file_name = self.mia_path + "-" + str(index) + ".txt"
            with open(file_name, "w") as fp:
                for each in self.mia_list:
                    fp.write(str(each) + "\n")
            print("Writing ", file_name, " ...")
        else:
            # False: list
            file_name = self.mia_path + "-" + str(index) + ".list"
            with open(file_name, "wb") as fp:
                pickle.dump(self.mia_list, fp)
            print("Writing ", file_name, " ...")

        self.mia_list.clear()

