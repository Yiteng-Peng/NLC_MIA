import torch
from tqdm import tqdm
import os, pickle
import numpy as np

import coverage.tool as tool

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


class SurpriseCoverage(Coverage):
    def init_variable(self, hyper, min_var, num_class):
        self.name = self.get_name()
        assert self.name in ['LSC', 'DSC', 'MDSC']
        assert hyper is not None
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.data_count = 0
        self.current = 0
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def build(self, data_loader):
        print('Building Mean & Var...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            # print(data.size())
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_meam_var(data, label)
        self.set_mask()
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, label)
        self.to_numpy()
        if self.name == 'LSC':
            self.set_kde()

    def assess(self, data_loader):
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.step(data, label)

    def step(self, data, label):
        cove_set = self.calculate(data, label)
        gain = self.gain(cove_set)
        if gain is not None:
            self.update(cove_set, gain)

    def set_meam_var(self, data, label):
        batch_size = label.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = ((self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
            + (self.data_count - batch_size) * ((layer_output - self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new):
        new_rate = self.coverage(cove_set_new)
        return new_rate - self.current

    def save(self, path):
        print('Saving recorded %s in %s...' % (self.name, path))
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s in %s...' % (self.name, path))
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)
