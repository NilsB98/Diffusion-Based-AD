from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch


class CustomFE(nn.Module):
    def __init__(self, state_dict_path=None, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, device=device)
        self.conv2 = nn.Conv2d(64, 256, 3, 2, 1, device=device)
        self.conv3 = nn.Conv2d(256, 512, 3, 2, 1, device=device)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1, device=device)
        self.activation = nn.ReLU()

        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        l1 = self.conv1(x)
        x = self.activation(l1)
        l2 = self.conv2(x)
        x = self.activation(l2)
        l3 = self.conv3(x)
        x = self.activation(l3)
        out = self.conv4(x)

        return {'out_layer': out, 'layer_3': l3, 'layer_2': l2}


class VGG16FE(nn.Module):
    def __init__(self, state_dict_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        return_nodes = {
            'features.10': 'features_1',
            'features.15': 'features_2',
            'features.25': 'out_layer',
        }

        self.model = create_feature_extractor(vgg16(), return_nodes)

        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        return self.model(x)


class ResNetFE(nn.Module):
    def __init__(self, state_dict_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        return_nodes = {'layer2': 'features_1', 'layer3': 'out_layer'}

        self.model = create_feature_extractor(wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT), return_nodes)

        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        return self.model(x)
