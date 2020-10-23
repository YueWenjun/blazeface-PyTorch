# -*- coding: utf-8 -*-

'''
I wanna try a shallower net with singleblazeblock here
'''

import torch
import torch.nn as nn
from blazefaceNet import *

def shallow_backbone():
    layers = list()
    layers.append(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=2, stride=2))# 图片的大小：128*128->64*64
    layers.append(nn.ReLU(inplace=True))#inplace设置为了节约显存,生成的新值会把原值覆盖
    layers += Singleblazeblock(24,24)
    layers += Singleblazeblock(24,48,stride=2)#64*64->32*32
    layers += Singleblazeblock(48,48)
    layers += Singleblazeblock(48,24,stride=2)#32*32->16*16
    layers += Singleblazeblock(24,96)
    layers += Singleblazeblock(96,24)
    layers += Singleblazeblock(24,96)#第一次提取层
    layers += Singleblazeblock(96,24,stride=2)#16*16->8*8
    layers += Singleblazeblock(24,96)
    layers += Singleblazeblock(96,24)
    layers += Singleblazeblock(24,96)#第二次提取层
    return layers

class shallowFaceNet(nn.Module):
    def __init__(self, size):
        super(shallowFaceNet, self).__init__()
        self.size = size
        self.Net_backbone = nn.ModuleList(shallow_backbone())
        extractor = multibox(self.Net_backbone)
        self.loc  = nn.ModuleList(extractor[0])
        self.conf = nn.ModuleList(extractor[1])
    
    def forward(self, x):
        extractor_sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.Net_backbone[k](x)
        extractor_sources.append(x)

        for k in range(23, len(self.Net_backbone)):
            x = self.Net_backbone[k](x)
        extractor_sources.append(x)

        for (x, l, c) in zip(extractor_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, 2),
            # self.priorsbox
        )

        return output