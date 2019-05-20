import torch
import torch.nn as nn
import torch.nn.functional as F


class DAM(nn.Module):

    def __init__(self, size):
        super(DAM, self).__init__()
        self.bn_layer = nn.BatchNorm1d(size[0])


    def PAM(self, Feature, alpha=0.5):
        # Position Attention Module

        # Feature.size() = BxCxHxW
        size = Feature.size()

        # A.size() = BxCxHxW
        a = Feature

        #apply bn and relu
        Feature = self.bn_layer(Feature)
        Feature = F.relu(Feature)


        # c.size() = BxCxN     - B, C and D in the paper
        c = Feature.reshape(size[0], size[1], -1)

        # cb = C' * B
        # cb.size() = NxN
        cb = torch.matmul(c.permute(0, 2, 1), c)

        # S.size() = NxN ( row : prob.)
        s = F.softmax(cb, dim=1)

        # Feed A to convolutional layer with bn and relu ? apply just bn & relu ?
        # d.size() = CxN

        # d = b & c
        # e.size() = B x C x N
        e = torch.matmul(c, s.permute(0, 2, 1))

        # E.size() = C x H x W
        e = e.reshape(size)

        e = alpha*e + a

        return e



    def CAM(self, Feature, beta=0.5):
        # Channel Attention Module

        # Feature.size() = CxHxW
        size = Feature.size()

        # CAM is calculated w.o bn and relu
        # c.size() = CxN     - C in the paper
        a = Feature.reshape(size[0], size[1] -1)

        # aa = A * A'
        # aa.size() = C x C
        aa = torch.matmul(a, a.permute(0, 2, 1)).permute(0, 2, 1)

        # X : channel attention map
        x = F.softmax(aa, dim=0)

        # E.size() = B x C x N
        e = torch.matmul(x.permute(0, 2, 1), a)

        # E.size() = B x C x H x W
        e = e.reshape(size)

        e = beta*e + a

        return e



    def forward(self, Feature, alpha, beta):

        # sum fusion of PAM & CAM's result

        result = DAM.PAM(Feature, alpha) + DAM.CAM(Feature, beta)

        return result









