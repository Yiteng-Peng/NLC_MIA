import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        
    def forward(self, x):
        return self.layers(x)
    
def make_mlp_layers(cfg, in_channels, out_channels):
    layers = []
    
    out_features = in_channels
    for each_features in cfg:
        in_features = out_features
        out_features = each_features
        
        linear_layer = nn.Linear(in_features, out_features)
        layers += [linear_layer, 
                   nn.ReLU(inplace=True)]
    
    # 分类
    layers += [nn.Linear(cfg[-1], out_channels)]
    if out_channels == 2:
        layers += [nn.Sigmoid()]
    
    return nn.Sequential(*layers)

cfgs = {
    "A": [20, 40 ,20]
}

def _mlp(cfg_index, pretrained, weight_dict, device, num_classes=10, out_channels=2):
    if pretrained:
        model = MLP(make_mlp_layers(cfgs[cfg_index], num_classes, out_channels)).to(device)
        model.load_state_dict(weight_dict)
    else:
        model = MLP(make_mlp_layers(cfgs[cfg_index], num_classes, out_channels)).to(device)

    return model

def mlp_a(pretrained=False, weight_dict=None, device="cpu", **kwargs):
    return _mlp("A", pretrained, weight_dict, device, **kwargs)