import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from models.dual_attention_module import DAM

from models.inceptionv3 import *

# from models.vgg import VGG
# from models.resnet import ResNet
# from models.SEC import SEC





class base(nn.Module):
    def __init__(self, net = None, num_classes = 200, channel = 2048):
        super(base,self).__init__()
        self.model = net
        self.bn = nn.BatchNorm1d(768)
        self.fc = nn.Linear(2048, num_classes, bias=None)

    def forward(self, x):
        if self.training:
            feature, states = self.model.forward_2(x)
        else :
            feature = self.model.forward_2(x)
        feature = self.fc(feature)


        return feature