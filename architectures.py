import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Resnet import resnet152_denoise, resnet101_denoise

def get_normalize_layer() -> torch.nn.Module:
    return NormalizeLayer()

class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, input: torch.tensor):
        # RGB to BGR
        permute_RGBtoBGR = [2, 1, 0]
        input = input[:, permute_RGBtoBGR, :, :]
        # normalize
        out = (input / 0.5) - 1
        return out

def get_architecture(denoise = True, model_name = "resnet152") -> torch.nn.Module:
    """
    load adversarially pre-trianed model by facebook https://github.com/facebookresearch/ImageNet-Adversarial-Training
    the checkpoint is converted from tensorflow to pytorch
    """
    if model_name == "Resnet101-DenoiseAll":
        model = resnet101_denoise()
        model.load_state_dict(torch.load("./adv_denoise_model/Adv_Denoise_Resnext101.pytorch"))
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        return nn.Sequential(get_normalize_layer(), model)

    if denoise:
        model = resnet152_denoise()
        model.load_state_dict(torch.load("./adv_denoise_model/Adv_Denoise_Resnet152.pytorch")) 
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    else:
        model = torchvision.models.resnet152(False)
        model.load_state_dict(torch.load("./adv_denoise_model/res152-adv.checkpoint"))
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    normalize_layer = get_normalize_layer()
    return torch.nn.Sequential(normalize_layer, model)