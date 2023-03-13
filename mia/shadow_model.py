import copy
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from config import DEVICE, LOSS_FUNC

class ShadowModels:
    # 假设每个数据是元组的形式，即形如(X,y)的形式
    def __init__(self, data, num_models, target_classes, learner_model,
                 epochs, batch_size, result_batch):
        self.num_models = num_models                                                # 模型个数
        self.target_classes = target_classes                                        # 目标分类个数
        self._splits = self._split_data(data, self.num_models)                      # 将数据集拆分为和模型数目n相同的n组数据
        self.models = self._make_model_list(learner_model, self.num_models)         # 复制n个模型
        self.results = self.train_predict_shadows(epochs, batch_size, result_batch) # 训练模型，从而得到影子数据集的prediction和标签

    @staticmethod
    def _split_data(data, n_splits):
        # 将模型保持分布不变的情况下，进行拆分，拆分为n个子数据集，每个数据集
        split_size = [len(data) // n_splits for _ in range(n_splits-1)]
        split_size.append(len(data) - (len(data) // n_splits) * (n_splits-1))
        
        return torch.utils.data.random_split(data, split_size)

    @staticmethod
    def _make_model_list(learner, n):
        try:
            if isinstance(learner, nn.Module):
                models_list = [copy.deepcopy(learner) for _ in range(n)]
        except NameError:
            print("using pytorch shadow models")
            raise NameError
        return models_list

    def train_predict_shadows(self, epochs, batch_size, result_batch):
        # "in" : 1   "out" : 0
        # TRAIN and predict
        results = []
        for model, data_subset in tqdm(zip(self.models, self._splits)):
            optimizer = optim.Adam(model.parameters())
            
            half_data_len = len(data_subset) // 2
            split_size = [len(data_subset) - half_data_len, half_data_len]
            data_train, data_test = torch.utils.data.random_split(data_subset, split_size)

            # train
            trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                for X, y in trainloader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    loss = LOSS_FUNC(pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # get data label
            trainloader = DataLoader(data_train, batch_size=result_batch)
            testloader = DataLoader(data_test, batch_size=result_batch)
            with torch.no_grad():
                # data IN training set labeled 1
                train_pred_list = []
                y_train_list = []
                for X, y in trainloader:
                    X = X.to(DEVICE)
                    pred = model(X)
                    train_pred_list.append(pred.cpu())
                    y_train_list.append(y.unsqueeze(dim=1).cpu())
                pred_train = np.vstack(train_pred_list)
                y_train = np.vstack(y_train_list)
                res_in = np.hstack((pred_train, y_train, np.ones_like(y_train)))

                # data OUT training set labeled 0
                test_pred_list = []
                y_test_list = []
                for X, y in testloader:
                    X = X.to(DEVICE)
                    pred = model(X)
                    test_pred_list.append(pred.cpu())
                    y_test_list.append(y.unsqueeze(dim=1).cpu())
                pred_test = np.vstack(test_pred_list)
                y_test = np.vstack(y_test_list)
                res_out = np.hstack((pred_test, y_test, np.zeros_like(y_test)))

                results.append(np.vstack((res_in, res_out)))

        return np.vstack(results)
    
    def save_results(self, path):
        print("Shape of shadow models datasets: ", self.results.shape)
        np.save(path, self.results)
        
    @staticmethod
    def load_results(path):
        return np.load(path)
