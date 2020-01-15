# -*- coding: utf-8 -*-
from faceDataset import FaceDataset
from blazefaceNet import *
import argparse
import torch.nn.init as init #pytorch的初始化
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torch.autograd import Variable
from multiboxloss import MultiBoxLoss

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
    myfaceDataset = FaceDataset("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/JPEGImages","/home/danale/disk/ywj/data/VOCdevkit/VOC2007/Annotations")
    print(len(myfaceDataset))
    print("myfaceDataset: ",str(myfaceDataset))

    myBlazefaceNet = BlazeFaceNet(128)

    if args.resume:
        print('Resuming training, loading weights from{}'.format(args.resume))
        myBlazefaceNet.load_weights(args.resume)
    else:
        print('Initializing weights...')
        myBlazefaceNet.Net_backbone.apply(weights_init)
        myBlazefaceNet.loc.apply(weights_init)
        myBlazefaceNet.conf.apply(weights_init)

    optimizer = optim.SGD(myBlazefaceNet.parameters(), lr=0.001, momentum=0.9)

    criterion = MultiBoxLoss()

    myBlazefaceNet.train()

    print(myBlazefaceNet)

    #Loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    # batch_size = 16
    print("Loading the dataset...")

    # epoch_size = len(myfaceDataset)//batch_size

    faceDataLoader = DataLoader(myfaceDataset, batch_size = 1, num_workers = 0, shuffle = True)

    for i_batch, sample_batched in enumerate(faceDataLoader):
        image = sample_batched["image"]
        image = Variable(image)
        print(type(image))
        print(image.size())
        targets = sample_batched["faces"]
        # targets = [Variable(ann, volatile=True) for ann in targets]
        t0 = time.time()
        output = myBlazefaceNet(image.float())
        # print("output[0](loc): ",output[0].size())
        # print("output[1](conf): ",output[1].size())
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_c + loss_l
        loss.backward()
        optimizer.step()
        t1 = time.time()
        print("timer: %.4f sec." %(t1-t0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "blazeface-PyTorch")
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    args = parser.parse_args()
    train()