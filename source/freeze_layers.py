
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models



model_ft = models.resnet50(pretrained=True)


def freeze_layers(model_ft,freeze_layer_until=7):
    layers_to_be_trained=[]
    ct = 0
    for child in model_ft.children():
        ct += 1
        if ct <= freeze_layer_until :
            for param in child.parameters():
                param.requires_grad = False
        else:
            layers_to_be_trained.append(child)
    return layers_to_be_trained