import copy
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_func = nn.CrossEntropyLoss()

class ShadowModels:
    # 假设每个数据是元组的形式，即形如(X,y)的形式
    def __init__(self, data, num_models, target_classes, learner_model, epochs, batch_size):
        # 模型个数
        self.num_models = num_models
        # 目标分类个数
        self.target_classes = target_classes
        # 将数据集拆分为和模型数目相同的num_models组数据
        self._splits = self._split_data(data, self.num_models)
        # num_models个模型
        self.models = self._make_model_list(learner_model, self.num_models)

        # 训练模型，从而得到影子数据集的prediction
        self.results = self.train_predict_shadows(epochs, batch_size)

    @staticmethod
    def _split_data(data, n_splits):
        # 将模型保持分布不变的情况下，进行拆分，拆分为n个子数据集，每个数据集
        return torch.utils.data.random_split(data, [1 / n_splits for i in range(n_splits)])

    @staticmethod
    def _make_model_list(learner, n):
        try:
            if isinstance(learner, nn.Module):
                models_list = [copy.deepcopy(learner) for _ in range(n)]
        except NameError:
            print("using pytorch shadow models")
            pass

        return models_list

    def train_predict_shadows(self, epochs, batch_size):
        # "in" : 1   "out" : 0
        # TRAIN and predict

        results = []
        for model, data_subset in tqdm(zip(self.models, self._splits)):

            optimizer = optim.Adam(model.parameters())

            data_train, data_test = torch.utils.data.random_split(data_subset, [0.5, 0.5])
            X_train = data_train.data.unsqueeze(1) / 255.0  # 归一化
            y_train = data_train.targets
            X_test = data_test.data.unsqueeze(1) / 255.0  # 归一化
            y_test = data_test.targets

            trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

            # train
            for epoch in range(epochs):
                for X, y in trainloader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    loss = loss_func(pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # get data label
            with torch.no_grad():
                # data IN training set labeled 1
                pred_train = model(X_train)
                res_in = np.hstack((pred_train, y_train, np.ones_like(y_train)))

                # data OUT training set labeled 0
                pred_test = model(X_test)
                res_out = np.hstack((pred_test, y_test, np.ones_like(y_test)))

                results.append(np.vstack((res_in, res_out)))

        return np.vstack(results)
