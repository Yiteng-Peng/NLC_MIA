import copy

import torch
from torch import nn, optim
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_func = nn.CrossEntropyLoss()

class AttackModels:
    def __init__(self, target_classes, attack_model):
        self.target_classes = target_classes
        self.attack_models = [copy.deepcopy(attack_model) for _ in range(target_classes)]

    def build(self, shadow_data, epochs):
        membership_label = shadow_data[:, -1]
        class_label = shadow_data[:, -2]
        data = shadow_data[:, :-2]

        # train model
        for epoch in range(epochs):
            for i, model in enumerate(self.attack_models):
                optimizer = optim.Adam(model.parameters())

                X = data[class_label == i]
                y = membership_label[class_label == i]
                X, y = X.to(DEVICE), y.to(DEVICE)

                pred = model(X)
                loss = loss_func(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def predict(self, X_pred, y, batch=False):
        if not batch:
            cls_model = self.attack_models[y]
            pred = cls_model(X_pred)
            return pred.argmax()
        else:
            model_classes = np.unique(y)
            res = []
            for cls in model_classes:
                X_pred_cls = X_pred[y == cls]
                model = self.attack_models[cls]
                attack_res = model(X_pred_cls)
                res.append(attack_res)

            return np.concatenate(res)
