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
from priorbox import Priorbox

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

    blazeface_net = BlazeFaceNet(128)
    priorbox = Priorbox()
    prior_boxes = priorbox.produce_priorboxes()# it should be torch.Size([896, 4])

    if args.resume:
        print('Resuming training, loading weights from{}'.format(args.resume))
        blazeface_net.load_weights(args.resume)
    else:
        print('Initializing weights...')
        blazeface_net.Net_backbone.apply(weights_init)
        blazeface_net.loc.apply(weights_init)
        blazeface_net.conf.apply(weights_init)

    optimizer = optim.SGD(blazeface_net.parameters(), lr=0.001, momentum=0.9)

    criterion = MultiBoxLoss(overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=3, neg_pos=0.5, neg_overlap=False, encode_target=True)

    blazeface_net.train()

    print(blazeface_net)

    #Loss counters
    loss_l = 0
    loss_c = 0
    batch_size = 16
    epoch_num = 100 #defined by youself
    print("Loading the dataset...")

    batch_num = len(myfaceDataset)//batch_size
    faceDataLoader = DataLoader(myfaceDataset, batch_size = batch_size, num_workers = 0, shuffle = True)# define num_workers by yourself

    for epoch in range(epoch_num):
        # print("epoch: {}/{}".format(epoch, epoch_num))
        
        for i, batch_data in enumerate(faceDataLoader):
            print("epoch: {}/{} batch: {}/{}".format(epoch, epoch_num, i, batch_num))
            image = batch_data["image"]#我不知道这个是不是要有一个长长的list,这里本来有batch_size张图片的,这个地方要继续改正
            image = Variable(image)
            print("input image type: ", type(image))
            print("input image size: ", image.size())

            targets =batch_data["faces"]
            print("targets.dtype: ", targets.dtype)
            print("targets.size: ", targets.size())

            t0 = time.time()
            output = blazeface_net(image.float())
            print("output[0](loc)size: ",output[0].size(),"output[0](dtype)",output[0].dtype)
            print("output[1](conf)size: ",output[1].size(),"output[0](dtype)",output[0].dtype)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(output, prior_boxes, targets)
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