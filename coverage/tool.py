import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

PAD_LENGTH = 32


'''
作用：在指定维度（默认是最后一个维度）上作缩放，默认是缩放到1~0上
参数：out：需要被缩放的数据，dim：维度，rmax：缩放的最大值，rmin：缩放的最小值
'''
def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    '''
        out_max = out.max()
        out_min = out.min()
    Note that the above max/min is incorrect when batch_size > 1
    '''
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

'''
作用：能判断的层的类型
参数：module: 层的类
'''
def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv3d)
            or isinstance(module, nn.RNN)
            or isinstance(module, nn.LSTM)
            or isinstance(module, nn.GRU)
            )

'''
作用：遍历所有子模块以确定能获得所有的层的信息
参数：name：当前的模块名 module：当前的模块（层的集合，或者层） namelist：模块名合集 module：模块合集
'''
def iterate_module(name, module, name_list, module_list):
    if is_valid(module):
        return name_list + [name], module_list + [module]
    else:

        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list

'''
作用：返回层的字典，键为层的名字，值为层的属性。
例子：{'Conv2d-1': Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))}
参数：model：机器学习模型
'''
def get_model_layers(model):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [])
        assert len(name_list) == len(module_list)   # 确保已经查询完毕
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])
    return layer_dict

'''
作用：获取层的输出规模，规模用数组表示
例子：{'Conv2d-1': [6, 28, 28], 'Conv2d-2': [16, 10, 10], 'Linear-1': [120], 'Linear-2': [84], 'Linear-3': [10]}
参数：model：模型，data：为了获取数据规模给的随机数据，pad_length：模型为了输入的时候需要给的padding，只有RNN，LSTM，GRU会使用
'''
def get_layer_output_sizes(model, data, pad_length=PAD_LENGTH):
    output_sizes = {}
    hooks = []
    name_counter = {}
    layer_dict = get_model_layers(model)

    # pytorch内部hook函数，每次forward()计算输出后都会调用该钩子。它不应该在内部改变output
    # 输入是模型/层，模型/层的输入，输出
    # 闭包
    def hook(module, input, output):
        class_name = module.__class__.__name__
        # 统计同一种类型的层的数目
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = [output[0].size(2)]
        else:
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = list(output.size()[1:])

    # 给每一层加hook
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    unrolled_output_sizes = {}
    for k in output_sizes.keys():
        if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
            for i in range(pad_length):
                unrolled_output_sizes['%s-%d' % (k, i)] = output_sizes[k]
        else:
            unrolled_output_sizes[k] = output_sizes[k]
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])
    return unrolled_output_sizes

def get_layer_output(model, data, pad_length=PAD_LENGTH):
    with torch.no_grad():
        name_counter = {}        
        layer_output_dict = {}
        layer_dict = get_model_layers(model)
        
        # 用字典的方式取出模型中层的输出
        def hook(module, input, output):
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output[0]
            else:
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            final_out = model(data)
        finally:
            for h in hooks:
                h.remove()

        unrolled_layer_output_dict = {}
        for k in layer_output_dict.keys():
            if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
                assert pad_length == len(layer_output_dict[k])
                for i in range(pad_length):
                    unrolled_layer_output_dict['%s-%d' % (k, i)] = layer_output_dict[k][i]
            else:
                unrolled_layer_output_dict[k] = layer_output_dict[k]

        for layer, output in unrolled_layer_output_dict.items():
            if len(output.size()) == 4: # (N, K, H, w)
                output = output.mean((2, 3))
            unrolled_layer_output_dict[layer] = output.detach()
        return unrolled_layer_output_dict

class Estimator(object):
    def __init__(self, feature_num, num_class=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.CoVariance = torch.zeros(num_class, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(num_class, feature_num).to(self.device)
        self.Amount = torch.zeros(num_class).to(self.device)

    def calculate(self, features, labels=None):
        N = features.size(0)
        C = self.num_class
        A = features.size(1)

        if labels is None:
            labels = torch.zeros(N).type(torch.LongTensor).to(self.device)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        # self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
        #               .mul(weight_CV)).detach() + additional_CV.detach()

        # self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        # self.Amount += onehot.sum(0)

        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave, 
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        self.Ave = dic['Ave']
        self.CoVariance = dic['CoVariance']
        self.Amount = dic['Amount']

    def transform(self, features, labels):
        CV = self.CoVariance[labels]
        (N, A) = features.size()
        transformed = torch.bmm(F.normalize(CV), features.view(N, A, 1))
        return transformed.squeeze(-1)

class EstimatorFlatten(object):
    def __init__(self, feature_num, num_class=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.CoVariance = torch.zeros(num_class, feature_num).to(self.device)
        self.Ave = torch.zeros(num_class, feature_num).to(self.device)
        self.Amount = torch.zeros(num_class).to(self.device)

    def calculate(self, features, labels=None):
        N = features.size(0)
        C = self.num_class
        A = features.size(1)

        if labels is None:
            labels = torch.zeros(N).type(torch.LongTensor).to(self.device)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        
        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave,
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        self.Ave = dic['Ave']
        self.CoVariance = dic['CoVariance']
        self.Amount = dic['Amount']

    def transform(self, features, labels):
        CV = self.CoVariance[labels]
        (N, A) = features.size()
        transformed = torch.bmm(features.view(N, 1, A), F.normalize(CV))
        return transformed.transpose(1, 2).squeeze(-1)


if __name__ == '__main__':
    pass