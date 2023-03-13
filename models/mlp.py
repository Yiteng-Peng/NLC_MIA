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

def _mlp(cfg_index, pretrained, mode_path, device, num_classes=10, out_channels=2):
    if pretrained:
        if "_@s" in mode_path:
            model = MLP(make_mlp_layers(cfgs[cfg_index], num_classes, out_channels)).to(device)
            model.load_state_dict(torch.load(mode_path, map_location=device))
        elif "_@m" in mode_path:
            model = torch.load(mode_path, map_location=device)
        else:
            raise NameError("Wrong model name, can't get model type, check '_@' in the model name")
    else:
        model = MLP(make_mlp_layers(cfgs[cfg_index], num_classes, out_channels)).to(device)

    return model

def mlp_a(pretrained=False, mode_path=None, device="cpu", **kwargs):
    return _mlp("A", pretrained, mode_path, device, **kwargs)