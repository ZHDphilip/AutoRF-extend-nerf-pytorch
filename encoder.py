# ================= #
# Author: Zihao DONG
# File name: encoder.py
# Description: encoder structure of AutoRF using pytorch
#       ResNet34 Backbone, replace BatchNorm2D with InstanceNorm2D
#       use ResNet34 to extract feature map of shape 256 * H/16 * W/16
#       replicate above feature map and pass to 2 separate heads to produce shape
#          and feature code, each of dimension 128
# ================= #

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import copy

from PIL import Image
import torchvision.transforms as transforms


class ResNet34_Backnone_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # replace BatchNorm2D using nn.InstanceNorm2D
        resnet = models.resnet34(norm_layer=nn.InstanceNorm2d)
        
        # take out pre layers from the resnet
        # size after pre layers: [-1, 64, H/2, W/2]
        self.prelayers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # take out layer1, output size [-1, 64, H/4, W/4]
        self.layer1 = resnet.layer1
        # take out layer2, output size [-1, 128, H/8, W/8]
        self.layer2 = resnet.layer2
        # take out layer3, output size [-1, 256, H/16, W/16]
        self.layer3 = resnet.layer3

        # define 2 separate heads, pass feature map through 2 heads
        # and apply adaptive max pooling to obtain shape/appearance code
        # quoting from the paper:
        #   The first four layers of this architecture are shared 
        #   while the following two layers are replicated to form 
        #   separate heads for shape and appearance encoding. 
        # so my under standing is to keep 2 copies of resnet.layer4, 
        # and use as the heads for shape and appearance code
        # add an adaptive max pooling to make dim = 128 instead

        # head for shape code, output dimension 128
        self.shape_head = nn.Sequential(
            copy.deepcopy(resnet.layer4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128)
        )
        # self.shape_pooling = nn.AdaptiveMaxPool2d(128)

        # head for appearance code, output dimension 128
        self.appear_head = nn.Sequential(
            copy.deepcopy(resnet.layer4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128)
        )
        # self.appear_pooling = nn.AdaptiveMaxPool2d(128)

    def forward(self, x):
        # pass through first 4 layers to get 256 * H/16 * W/16 feature map
        x = self.prelayers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # pass through shape head to get shape code
        shape_code = self.shape_head(x)
        # shape_code = self.shape_pooling(shape_code)

        # pass through appearance head to get appearance code
        appear_code = self.appear_head(x)
        # appear_code = self.appear_pooling(appear_code)
        return shape_code, appear_code