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
            for i, model in tqdm(enumerate(self.attack_models)):
                optimizer = optim.Adam(model.parameters())
                X = data[class_label == i]
                y = membership_label[class_label == i]
                X, y = torch.Tensor(X).to(DEVICE), torch.Tensor(y).to(DEVICE)
                pred = model(X)
                loss = LOSS_FUNC(pred, y)
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
        
    def save(self, path):
        for i in range(self.target_classes):
            model_path = path + str(i) + ".pt"
            torch.save(self.attack_models[i].state_dict(), model_path)
    
    def load(self, model, path):
        model_list = []
        for i in range(self.target_classes):
            model_path = path + str(i) + ".pt"
            model_list.append(model(pretrained=True, mode_path=model_path, 
                                   device=DEVICE, num_classes=self.target_classes))
        return model_list