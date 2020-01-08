# -*- coding: utf-8 -*-
from faceDataset import FaceDataset
from blazefaceNet import *
import argparse
import torch.nn.init as init #pytorch的初始化

# from data import *
# from utils.augmentations import BlazefaceAugmentation
# from blazeface import build_blazeface
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.optim as optim
# from layers.modules import MultiBoxLoss
# import torch.utils.data as data
# import time
# from torch.autograd import Variable
# import torchvision.transforms as transforms

# import os
# import sys


# import numpy as np

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def train():
    myfaceDataset = FaceDataset("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/JPEGImages")
    print(len(myfaceDataset))
    print("myfaceDataset: ",str(myfaceDataset))

    myBlazefaceNet = BlazeFaceNet(128)

    if args.resume:
        print('Resuming training, loading weights from{}'.format(args.resume))
        myBlazefaceNet.load_weights(args.resume)
    else:
        print('Initializing weights...')
        myBlazefaceNet.Net_backbone.apply(weights_init)

parser = argparse.ArgumentParser(description = "blazeface-PyTorch")
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
args = parser.parse_args()
train()