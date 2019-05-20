import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from models.dual_attention_module import DAM

from models.inceptionv3 import *
from models.vgg import VGG
from models.resnet import ResNet
from models.SEC import SEC



class PBML(nn.Module):
    def __init__(self, num_classes, alpha, beta, directory, net = None):
        super(PBML,self).__init__()
        self.num_classes = num_classes
        self.baseline = 'inception'
        self.num_features = 768
        self.num_channel = 512
        self.expansion = 1
        self.alpha = alpha
        self.beta = beta
        self.num_clusters = num_clusters

        if net is not None:
            self.features = net.forward()
            if isinstance(net, ResNet):
                self.baseline = 'resnet'
                self.expansion = self.features[-1][-1].expansion
                self.num_features = 512
            elif isinstance(net, VGG):
                self.baseline = 'vgg'
                self.num_features = 512
        else :
            self.features = inception_v3(pretrained=True).forward_1()



        # # self.attention_module = DAM.forward()
        self.pconv1 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv2 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv3 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv4 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv5 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv6 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv7 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1
        self.pconv8 = nn.Conv2d(self.num_features, 1, 1, padding=0)  # padding = 0 or 1

        self.pmd1 = nn.Linear(self.num_features, 256)
        self.pmd2 = nn.Linear(self.num_features, 256)
        self.pmd3 = nn.Linear(self.num_features, 256)
        self.pmd4 = nn.Linear(self.num_features, 256)
        self.pmd5 = nn.Linear(self.num_features, 256)
        self.pmd6 = nn.Linear(self.num_features, 256)
        self.pmd7 = nn.Linear(self.num_features, 256)
        self.pmd8 = nn.Linear(self.num_features, 256)



        #
        # self.fc = nn.Linear(self.num_att * self.num_features * self.expansion, self.num_classes)
        # self.softmax_layer = nn.Softmax(dim = 0)
        # self.softmax_loss = nn.CrossEntropyLoss()








    def forward(self, x, loss_flag):

        feature_maps = self.features(x)
        b, c, h, w = feature_maps.size

        # Experimental : Dual Attention Map
        attention_maps = DAM(feature_maps, self.alpha, self.beta)

        # B x num_att x 14 x 14

        avgpool = nn.AvgPool2d([h, w])


        part1 = self.pconv1(feature_maps)
        part1 = F.sigmoid(part1)
        pa_1  = torch.mul(feature_maps, part1)
        pa_1  = avgpool(pa_1)
        pm = self.pmd1(pa_1)


        part2 = self.pconv2(feature_maps)
        part2 = F.sigmoid(part2)
        pa_2  = torch.mul(feature_maps, part2)
        pa_2  = avgpool(pa_2)
        pm = torch.cat((pm, self.pmd2(pa_2)), dim=1)



        part3 = self.pconv3(feature_maps)
        part3 = F.sigmoid(part3)
        pa_3  = torch.mul(feature_maps, part3)
        pa_3  = avgpool(pa_3)
        pm = torch.cat((pm, self.pmd3(pa_3)), dim=1)

        part4 = self.pconv4(feature_maps)
        part4 = F.sigmoid(part4)
        pa_4  = torch.mul(feature_maps, part4)
        pa_4  = avgpool(pa_4)
        pm = torch.cat((pm, self.pmd4(pa_4)), dim=1)

        part5 = self.pconv5(feature_maps)
        part5 = F.sigmoid(part5)
        pa_5  = torch.mul(feature_maps, part5)
        pa_5  = avgpool(pa_5)
        pm = torch.cat((pm, self.pmd5(pa_5)), dim=1)

        part6 = self.pconv6(feature_maps)
        part6 = F.sigmoid(part6)
        pa_6  = torch.mul(feature_maps, part6)
        pa_6  = avgpool(pa_6)
        pm = torch.cat((pm, self.pmd6(pa_6)), dim=1)

        part7 = self.pconv7(feature_maps)
        part7 = F.sigmoid(part7)
        pa_7  = torch.mul(feature_maps, part7)
        pa_7  = avgpool(pa_7)
        pm = torch.cat((pm, self.pmd7(pa_7)), dim=1)

        part8 = self.pconv8(feature_maps)
        part8 = F.sigmoid(part8)
        pa_8  = torch.mul(feature_maps, part8)
        pa_8  = avgpool(pa_8)
        pm = torch.cat((pm, self.pmd8(pa_8)), dim=1)


        return pm
        #
        # # branch 1
        # kmeans = []
        #
        # # TRAIN KMEANS INDEPENDENTLY
        # # for i in range(part1.size()[1]):
        # # LOAD EACH KMEANS MODEL
        # # kmeans_part1 = MiniBatchKMeans(n_clusters=self.num_clusters , random_state=0).partial_fit(part1) ### 1 image's part
        # prob_p1 = self.softmax_layer(part1)
        # CEloss1 = self.softmax_loss(prob_p1, kmeans_part1.predict(part1))
        #
        # prob_p2 = self.softmax_layer(part2)
        # CEloss2 = self.softmax_loss(prob_p2, kmeans_part2.predict(part1))
        #
        # prob_p3 = self.softmax_layer(part3)
        # CEloss3 = self.softmax_loss(prob_p3, kmeans_part3.predict(part1))
        #
        # prob_p4 = self.softmax_layer(part4)
        # CEloss4 = self.softmax_loss(prob_p4, kmeans_part4.predict(part1))
        #
        # prob_p5 = self.softmax_layer(part5)
        # CEloss5 = self.softmax_loss(prob_p5, kmeans_part5.predict(part1))
        #
        # prob_p6 = self.softmax_layer(part6)
        # CEloss6 = self.softmax_loss(prob_p6, kmeans_part6.predict(part1))
        #
        # prob_p7 = self.softmax_layer(part7)
        # CEloss7 = self.softmax_loss(prob_p7, kmeans_part7.predict(part1))
        #
        # prob_p8 = self.softmax_layer(part8)
        # CEloss8 = self.softmax_loss(prob_p8, kmeans_part8.predict(part1))
        #
        # loss = CEloss1 + CEloss2 + CEloss3 + CEloss4+ CEloss5+ CEloss6+ CEloss7+ CEloss8
        #
        # graph = torch.tensor(np.array((28, )))
        # graph[0] = torch.matmul(prob_p1.permute(0, 2, 1), prob_p2)
        # graph[1] = torch.matmul(prob_p1.permute(0, 2, 1), prob_p3)
        # graph[2] = torch.matmul(prob_p1.permute(0, 2, 1), prob_p4)
        # graph[3]= torch.matmul(prob_p1.permute(0, 2, 1), prob_p5)
        # graph[4]= torch.matmul(prob_p1.permute(0, 2, 1), prob_p6)
        # graph[5]= torch.matmul(prob_p1.permute(0, 2, 1), prob_p7)
        # graph[6]= torch.matmul(prob_p1.permute(0, 2, 1), prob_p8)
        # graph[7]= torch.matmul(prob_p2.permute(0, 2, 1), prob_p3)
        # graph[8]= torch.matmul(prob_p2.permute(0, 2, 1), prob_p4)
        # graph[9] = torch.matmul(prob_p2.permute(0, 2, 1), prob_p5)
        # graph[10] = torch.matmul(prob_p2.permute(0, 2, 1), prob_p6)
        # graph[11] = torch.matmul(prob_p2.permute(0, 2, 1), prob_p7)
        # graph[12] = torch.matmul(prob_p2.permute(0, 2, 1), prob_p8)
        # graph[13] = torch.matmul(prob_p3.permute(0, 2, 1), prob_p4)
        # graph[14] = torch.matmul(prob_p3.permute(0, 2, 1), prob_p5)
        # graph[15] = torch.matmul(prob_p3.permute(0, 2, 1), prob_p6)
        # graph[16] = torch.matmul(prob_p3.permute(0, 2, 1), prob_p7)
        # graph[17] = torch.matmul(prob_p3.permute(0, 2, 1), prob_p8)
        # graph[18] = torch.matmul(prob_p4.permute(0, 2, 1), prob_p5)
        # graph[19] = torch.matmul(prob_p4.permute(0, 2, 1), prob_p6)
        # graph[20] = torch.matmul(prob_p4.permute(0, 2, 1), prob_p7)
        # graph[21] = torch.matmul(prob_p4.permute(0, 2, 1), prob_p8)
        # graph[22] = torch.matmul(prob_p5.permute(0, 2, 1), prob_p6)
        # graph[23] = torch.matmul(prob_p5.permute(0, 2, 1), prob_p7)
        # graph[24] = torch.matmul(prob_p5.permute(0, 2, 1), prob_p8)
        # graph[25] = torch.matmul(prob_p6.permute(0, 2, 1), prob_p7)
        # graph[26] = torch.matmul(prob_p6.permute(0, 2, 1), prob_p8)
        # graph[27] = torch.matmul(prob_p7.permute(0, 2, 1), prob_p8)
        #
        # for i in range(28):
        #     if i == 0:
        #         rp = graph[0]
        #     else:
        #         rp = torch.cat(rp, graph[i], dim=0)


        # branch 2
        #
        # for i in range(part_extraction.size()[1]):
        #     part_extraction[:,:,:]
