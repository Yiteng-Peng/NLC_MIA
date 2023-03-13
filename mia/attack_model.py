import copy
from tqdm import tqdm, trange
import torch
from torch import nn, optim
import numpy as np

from config import DEVICE, LOSS_FUNC

class AttackModels:
    def __init__(self, target_classes, attack_model, pretrained=False, path=None):
        self.target_classes = target_classes
        if pretrained:
            self.attack_models = self.load(attack_model, path)
        else:
            model = attack_model(device=DEVICE, num_classes=target_classes)
            self.attack_models = [copy.deepcopy(model) for _ in range(target_classes)]

    def build(self, shadow_data, epochs):
        membership_label = shadow_data[:, -1]
        class_label = shadow_data[:, -2]
        data = shadow_data[:, :-2]

        # train model
        for epoch in trange(epochs):
            for i, model in enumerate(self.attack_models):
                optimizer = optim.Adam(model.parameters())
                X = data[class_label == i]
                y = membership_label[class_label == i]
                X, y = torch.Tensor(X).to(DEVICE), torch.Tensor(y).long().to(DEVICE)
                pred = model(X)
                loss = LOSS_FUNC(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def predict(self, X_pred, y, batch=False):
        with torch.no_grad():
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
                    res.append(attack_res.cpu())
                res = np.concatenate(res)
                return np.argmax(res, axis=1)
        
    def save(self, path):
        model_list = []
        for i in range(self.target_classes):
            model_list.append(self.attack_models[i].state_dict())
        torch.save(model_list, path)
    
    def load(self, model, path):
        model_list = []
        state_dict_list = torch.load(path, map_location=DEVICE)
        for i in range(self.target_classes):
            model_list.append(model(pretrained=True, weight_dict=state_dict_list[i], 
                                   device=DEVICE, num_classes=self.target_classes))
        return model_list