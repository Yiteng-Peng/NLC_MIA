import os

import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 1 * 1, 4096),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg_index, batch_norm, pretrained, mode_path, device, **kwargs):
    if pretrained:
        if "_@s" in mode_path:
            kwargs["init_weights"] = False
            model = VGG(make_layers(cfgs[cfg_index], batch_norm=batch_norm), **kwargs).to(device)
            model.load_state_dict(torch.load(mode_path, map_location=device))
        elif "_@m" in mode_path:
            model = torch.load(mode_path, map_location=device)
        else:
            raise NameError("Wrong model name, can't get model type, check '_@' in the model name")
    else:
        model = VGG(make_layers(cfgs[cfg_index], batch_norm=batch_norm), **kwargs).to(device)

    return model


def vgg11_bn(pretrained=False, mode_path=None, device="cpu", **kwargs):
    return _vgg("A", True, pretrained, mode_path, device, **kwargs)


def vgg13_bn(pretrained=False, mode_path=None, device="cpu", **kwargs):
    return _vgg("B", True, pretrained, mode_path, device, **kwargs)


def vgg16_bn(pretrained=False, mode_path=None, device="cpu", **kwargs):
    return _vgg("D", True, pretrained, mode_path, device, **kwargs)


def vgg19_bn(pretrained=False, mode_path=None, device="cpu", **kwargs):
    return _vgg("E", True, pretrained, mode_path, device, **kwargs)
