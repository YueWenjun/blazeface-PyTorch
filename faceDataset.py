# -*- coding: utf-8 -*-
#from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

class FaceDataset(Dataset):
    """
    root:     图像等存放地址的根路径
    augment:  是否需要图像增强
    """
    def __init__(self, root, augment=None):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.augment = augment

    def __getitem__(self, index):
        #
        #这里的open_Image是读取图像的函数, 可以用PIL、opencv等库进行读取
        if self.augment:
            image = open_Image(self.image_files[index])
            image = self.augment(image)
            return to_tensor(image)
        else:
            return to_tensor(open_Image(self.image_files[index]))

    def __len__(self):
        #返回图像的数量
        return len(self.image_files)

    def __str__(faceDataset):
        return "This dataset is a instance of faceDataset Class inherited from Dataset Class"

    def open_Image(root):
        #读取图像的函数,用什么实现呢?
        pass
    
    def augment(image):
        #要进行什么增强处理呢?
        pass

# faceDataset = FaceDataset("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/JPEGImages")
# print(len(faceDataset))
# print(str(faceDataset))