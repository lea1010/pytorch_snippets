import torch.nn as nn
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from pytorch.ModelTrainer import *
from torchvision.models.resnet import BasicBlock


class MyResnet2(models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(MyResnet2, self).__init__(block, layers, num_classes)
        self.conv_feat = nn.Conv2d(in_channels=512,
                                   out_channels=6000,
                                   kernel_size=1)
        self.fc = nn.Linear(in_features=6000,
                            out_features=6000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.conv_feat(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class simpleCNN(nn.Module):
    def __init__(self,outChl):
        super(simpleCNN, self).__init__()
        # out= [(inputDim−kernelDim+2*Padding)/Stride]+1
        self.conv1 = nn.Conv2d(3, 8, 5,3,1)   #inChl,outChl,kernel,stride,pad
        self.conv2 = nn.Conv2d(8, 16, 3,2)
        # self.conv3 = nn.Conv2d(16, 32, 3,1)
        self.conv1by1 =nn.Conv2d(3,1,1) # 1x1 layer to squash it to 1 channel
        # self.conv1by1_2 =nn.Conv2d(16,8,1)
        self.fc1 = nn.Linear(16 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 56)
        self.fc3 = nn.Linear(56, outChl)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop=nn.Dropout(0.3)
        self.drop_2d = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #228 x228->pool(74x74) -> 37x37
        x = self.pool(F.relu(self.drop_2d(self.conv2(x))))  # pool(18) ->9x9
        # x = self.pool(F.relu(self.drop_2d(self.conv3(x))))  # pool(16) ->8x8
        # x = self.conv1by1_2(x)
        # print("after Conv",x.shape)
        x = x.view(-1, 16 * 9 * 9)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x



class simpleCNN2(nn.Module):
    def __init__(self,outChl):
        super(simpleCNN2, self).__init__()
        # out= [(inputDim−kernelDim+2*Padding)/Stride]+1
        self.conv1 = nn.Conv2d(3, 8, 5)   #inChl,outChl,kernel,stride,pad
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3,1)
        self.conv4 = nn.Conv2d(32, 64, 3,1)
        # self.conv1by1 =nn.Conv2d(3,1,1) # 1x1 layer to squash it to 1 channel
        self.conv1by1 =nn.Conv2d(64,16,1)
        self.fc1 = nn.Linear(16 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 56)
        self.fc3 = nn.Linear(56, outChl)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop=nn.Dropout(0.3)
        # self.drop_2d = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #228 x228->pool(220) -> 110
        x = self.pool(F.relu(self.conv2(x)))  # pool(108) ->54
        x = self.pool(F.relu(self.conv3(x)))  # pool(52) ->26
        x = self.pool(F.relu(self.conv4(x)))  # pool(24) ->12
        x = self.conv1by1(x) # 16x12x12

        # print("after Conv",x.shape)
        x = x.view(-1, 16 * 12 * 12)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x





# model = MyResnet2(BasicBlock, [3, 4, 6, 3], 1000)
# x = Variable(torch.randn(1, 3, 224, 224))
# output = model(x)
