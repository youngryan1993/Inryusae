import torch
import torch.nn as nn
import numpy as np




class Metirc_module(nn.Module):

    def __init__(self, size = 2048, em_size = 128, out1 = 512, out2 = 1024, normal=True, is_Training=True):

        super(Metirc_module, self).__init__()

        self.bn = nn.BatchNorm1d(size, affine=is_Training)
        self.fc1 = nn.Linear(size, em_size)
        self.fc2 = nn.Linear(em_size, out1)
        self.fc3 = nn.Linear(out1, out2)
        self.CEloss_ = nn.softmax