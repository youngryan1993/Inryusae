import torch
import torch.nn as nn

class SEC(nn.module):
    def __init__(self, size, r):
        super(SEC, self).__init__()

        # # self.attention_module = DAM.forward()
        self.pconv = nn.Conv2d(self.num_features, self.num_channel, 1, padding=0)  # padding = 0 or 1
        self.avgpool = nn.AvgPool2d((size[2], size[3]))
        self.fc = nn.Sequential(
            nn.Linear(size[1], size[1]//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(size[1]//r, size[1], bias=False),
            nn.Sigmoid()
            )
        # self.fc = nn.Linear(self.num_att * self.num_features * self.expansion, self.num_classes)
        # self.softmax_layer = nn.Softmax(dim = 0)
        # self.softmax_loss = nn.CrossEntropyLoss()

    def forward(self, x):

        b,c,_,_ = x.size()
        # squeeze
        y = self.avgpool(x).view(b,c)


        # excitation
        y = self.fc(y). view(b,c,1,1)

        return x * y.expand_as(x)