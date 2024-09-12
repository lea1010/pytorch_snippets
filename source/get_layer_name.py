
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models



def get_layer_names(model):

    layers=[]
    for name, module in model.named_modules():
        print(name)
        layers.append(name)
    print(len(layers))
    return layers


def get_model_parameter(model):
    params=[]
    c=1
    for module in model_ft.children():
        print(c,module)
        c+=1
    for param in model.parameters():
        # print (c,param.shape)

        params.append(param)
    return params


model_ft = models.resnet50(pretrained=False)
get_layer_names(model_ft)



get_model_parameter(model_ft)