# ================= #
# Author: Zihao DONG
# File name: decoder.py
# Description: This file contains torch implementation of 
#   Shape Decoder: MLP made of 5 ResNet blocks with hidden dim 128
#       At each layer, aggregate positional encoding with feature map by 
#       a per-channel mean pooling. Map positional enc by a linear layer
#       output a density prediction for a given 3D point expressed in NOCS 
#   Color Decoder: 
# ================= #
import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# shape decoder, takes in point position in NOCS + shape code
# outputs the density of that 3D point as a scalar value sigma in R_+
class ShapeDecoder(nn.Module):
    def __init__(self, pos_dim=3, norm_layer=nn.InstanceNorm2d):
        # param:
        #   pos_dim: the dimension of the position embedding, this is ambiguous according to 
        #       the paper, so I use a parameter here to determine later
        #   norm_layer: the callable torch nn module to use in BasicBlock
        super().__init__()
        # linear network used to reshape positional embedding for aggregation with feature code
        self.pos_emb_reshape_layer = nn.Linear(in_features=pos_dim, out_features=128)

        # pool layer to aggregate the transformed positional embedding with feature map
        self.aggregate = None
        
        # self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # 5 layer of BasicBlock, each with hidden dimension 128
        # self.layer1 = BasicBlock(inplanes=32, planes=128, norm_layer=norm_layer)
        # self.layer2 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer3 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer4 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer5 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        self.layer1 = nn.Linear(in_features=128, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=128)
        self.layer5 = nn.Linear(in_features=128, out_features=128)

        # fc layer to produce the density output
        self.fc = nn.Linear(128, 1)

    def forward(self, shape_code, pos_emb):
        print(pos_emb.shape)
        pos_emb = self.pos_emb_reshape_layer(pos_emb)
        # aggregate shape code and positional embedding
        print(pos_emb.shape)
        feature = torch.mean(shape_code + pos_emb, dim=0)
        print(feature.shape)
        # feature = self.conv(feature)
        feature = self.layer1(feature)
        feature = torch.mean(feature + pos_emb, dim=0)
        feature = self.layer2(feature)
        feature = torch.mean(feature + pos_emb, dim=0)
        # need to return output of this layer for color decoder
        feature_out = self.layer3(feature)
        feature = torch.mean(feature_out + pos_emb, dim=0)
        feature = self.layer4(feature)
        feature = torch.mean(feature + pos_emb, dim=0)
        feature = self.layer5(feature)
        feature = torch.mean(feature + pos_emb, dim=0)
        out = self.fc(feature)
        return out, feature_out


# color decoder, takes in color code and intermediate features from shape decoder
# in order to take shape information into consideration when generating RGB outputs
class ColorDecoder(nn.Module):
    def __init__(self, pos_dim=3, view_dir_dim=3, norm_layer=nn.InstanceNorm2d):
        # param:
        #   pos_dim: the dimension of the position embedding, this is ambiguous according to 
        #       the paper, so I use a parameter here to determine later
        #   norm_layer: the callable torch nn module to use in BasicBlock
        super().__init__()
        # linear network used to reshape positional embedding for aggregation with feature code
        self.pos_emb_reshape_layer = nn.Linear(in_features=pos_dim, out_features=128)
        # linear network used to reshape view direction positional embedding for aggregation with feature code
        self.view_emb_reshape_layer = nn.Linear(in_features=view_dir_dim, out_features=128)

        # pool layer to aggregate the transformed positional embedding with feature map
        self.aggregate = None

        # 5 layer of BasicBlock, each with hidden dimension 128
        # self.layer1 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer2 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer3 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer4 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        # self.layer5 = BasicBlock(inplanes=128, planes=128, norm_layer=norm_layer)
        self.layer1 = nn.Linear(in_features=128, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=128)
        self.layer5 = nn.Linear(in_features=128, out_features=128)

        # fc layer to produce the density output
        self.fc = nn.Linear(128, 3)

    def forward(self, appear_code, shape_code, pos_emb, view_direction):
        # reshape positional embedding to match shape code dimension
        pos_emb = self.pos_emb_reshape_layer(pos_emb)
        view_dir = self.view_emb_reshape_layer(view_direction)
        # aggregate shape code and positional embedding
        feature = torch.mean(appear_code + pos_emb, dim=0)
        feature = self.layer1(feature)
        feature = torch.mean(feature + pos_emb, dim=0)
        feature = self.layer2(feature)
        # in this layer aggregate shape code as well
        feature = torch.mean(feature + shape_code + pos_emb, dim=0)
        feature = self.layer3(feature)
        feature = torch.mean(feature + pos_emb + view_dir, dim=0)
        feature = self.layer4(feature)
        feature = torch.mean(feature + pos_emb + view_dir, dim=0)
        feature = self.layer5(feature)
        feature = torch.mean(feature + pos_emb + view_dir, dim=0)
        RGB_output = self.fc(feature)
        return RGB_output