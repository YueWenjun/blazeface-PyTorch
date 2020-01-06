# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

#this class was defined to print everylayer
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)      #print(x.shape)
        return x

class BlazeFaceNet(nn.Module):
    def __init__(self, size):
        super(BlazeFaceNet, self).__init__()
        self.size = size
        self.Net_backbone = nn.ModuleList(backbone())
        extractor = multibox(self.Net_backbone)
        self.loc  = nn.ModuleList(extractor[0])
        self.conf = nn.ModuleList(extractor[1])
        
    
def backbone():
    layers = list()
    layers.append(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=2, stride=2))#128*128->64*64
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

def Singleblazeblock(in_channels, out_channels, kernel_size=5, padding=2, stride=1):
    layers = list()
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels))
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1))
    layers.append(nn.ReLU(inplace=True))
    return layers

def multibox(backbone, cfg=[2,6]):
    loc_layers = list()
    conf_layers = list()
    extractor_layerth = [22, 34]
    # print(len(backbone))
    for i, j in enumerate(extractor_layerth):
        loc_layers += [nn.Conv2d(backbone[j-1].out_channels, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(backbone[j-1].out_channels, cfg[i]*2, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)

blazefaceNet = BlazeFaceNet(128)
