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
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
plt.ion()
plt.switch_backend('agg')

class FaceDataset(Dataset):
    """
    root:     图像等存放地址的根路径
    augment:  是否需要图像增强
    """
    def __init__(self, picture_root, xml_root, transform=True):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(picture_root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.xml_files = np.array([x.path for x in os.scandir(xml_root) if x.name.endswith(".xml")])
        self.transform = transform
        self.rescale = Rescale(128)
        self.totensor = ToTensor()

    def __getitem__(self, index):
        
        image = plt.imread(self.image_files[index])
        faces = parse_rec(self.xml_files[index])
        sample = {"image":image, "faces":faces}
        if self.transform:
            sample = self.rescale(sample)
            sample = self.totensor(sample)
        return sample

    def __len__(self):
        #返回图像的数量
        return len(self.image_files)

    def __str__(faceDataset):
        return "This dataset is a instance of faceDataset Class inherited from Dataset Class"

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size:(int) image to a square
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    def __call__(self, sample):
        image, faces = sample["image"], sample["faces"]
        h, w = image.shape[:2]
        img = transform.resize(image, (self.output_size, self.output_size))
        for face in faces:
            face["bbox"][0] = int(face["bbox"][0]*self.output_size/w)
            face["bbox"][1] = int(face["bbox"][1]*self.output_size/h)
            face["bbox"][2] = int(face["bbox"][2]*self.output_size/w)
            face["bbox"][3] = int(face["bbox"][3]*self.output_size/h)
        return {"image": img,
                "faces": faces}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self,sample):
        image, faces = sample["image"], sample["faces"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image),
                "faces": faces}
    
def parse_rec(xml_filename):
    """ Parse my label file in xml format """
    tree = ET.parse(xml_filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text) - 1,
            int(bbox.find('ymin').text) - 1,
            int(bbox.find('xmax').text) - 1,
            int(bbox.find('ymax').text) - 1,
        ]
        objects.append(obj_struct)

    return objects

def show_facesdetected(image, faces):
    """show image with bbox"""
    print(type(image))
    plt.imshow(image)
    for face in faces:
        plt.scatter(face["bbox"][0],face["bbox"][1])
        plt.scatter(face["bbox"][2],face["bbox"][3])
        plt.pause(0.001)

def show_facesdetected_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, faces_batch = sample_batched["image"], sample_batched["faces"]
    batch_size = len(images_batch)
    for i in range(batch_size):
        print(i, images_batch[i].shape, faces_batch[i])


if __name__ == "__main__":
    faceDataset = FaceDataset("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/JPEGImages","/home/danale/disk/ywj/data/VOCdevkit/VOC2007/Annotations")
    print(len(faceDataset))
    print(str(faceDataset))
    # print(parse_rec("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/Annotations/004999.xml"))
    
    scale = Rescale(128)
    
    # for i in range(len(faceDataset)):
    #     sample = faceDataset[i]
    #     print(i, sample["image"].shape, sample["faces"])
    #     show_facesdetected(**sample)
    #     sample = scale(sample)
    #     show_facesdetected(**sample)

    #     # plt.show()
    #     # break

    #     totensor = ToTensor()
    #     sample = totensor(sample)
    #     print(i, sample["image"].shape, sample["faces"])
    faceDataLoader = DataLoader(faceDataset, batch_size = 1, num_workers = 0, shuffle = True)
    for i_batch, sample_batched in enumerate(faceDataLoader):
        print("*****************************************")
        print(i_batch, sample_batched["image"].size(),sample_batched["faces"])
        show_facesdetected_batch(sample_batched)