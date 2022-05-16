import torch
import torch.nn as nn

from encoder import *
from decoder import *

class AutoRF(nn.Module):
    def __init__(self, pos_dim=3, view_dir_dim=3, norm_layer=nn.InstanceNorm2d):
        super(AutoRF, self).__init__()
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        # encoder
        self.encoder = ResNet34_Backnone_Encoder()
        # shape decoder
        self.shape_decoder = ShapeDecoder(pos_dim=self.pos_dim, norm_layer=norm_layer)
        # color decoder
        self.color_decoder = ColorDecoder(pos_dim=self.pos_dim, view_dir_dim=self.view_dir_dim, norm_layer=norm_layer)
    def forward(self, img, pos_emb, view_dir):
        shape_code, appear_code = self.encoder(img)
        sigma, intermediate_output = self.shape_decoder(shape_code, pos_emb)
        rgb = self.color_decoder(appear_code, intermediate_output, pos_emb, view_dir)
        ret = torch.cat([sigma, rgb], -1)
        return ret


# Loss functions
'''
L_RGB := 1/|W_RGB| * SUM(L2 Dist)
'''
def L_RGB(mask, img, rgb_output):
    normalizer = torch.sum(mask)
    dist = (img - rgb_output) ** 2
    Loss = torch.sum(
        torch.mul(
            mask, dist
        )
    )
    return Loss * normalizer


'''
L_OCC := -1/|W_OCC| * SUM(log(Mask * (0.5 - alpha) + 0.5))
'''
def L_OCC(mask, alpha):
    new_mask = torch.where(
        mask == 0, -1, mask
    )
    Log = torch.log2(new_mask * (0.5 - alpha) + 0.5)
    Loss = -torch.mean(Log.flatten())
    return Loss


'''
Loss = L_RGB + lambda * L_OCC'''
def Loss_AutoRF(weight, mask, img, rgb_output, alpha):
    return L_RGB(mask, img, rgb_output) + weight * L_OCC(mask, alpha)