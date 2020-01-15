# -*- coding: utf-8 -*-
from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

class Priorbox(object):
    """
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    input: nothing
    output: 
    """
    def __init__(self):
        super(Priorbox, self).__init__()
        self.image_size = 128
        self.variance = [0.1, 0.2]
        self.feature_maps = [16, 8]
        self.min_sizes = [8, 32]
        self.max_sizes = [32, 128]
        self.steps = [8,16]
        self.aspect_ratios = 1
    
    def produce_priorboxes(self):
        mean = []
        for k , f in enumerate(self.feature_maps):
            if k == 0:
                for i,j in product(range(f), repeat = 2):
                    f_k = self.image_size / self.steps[k]
                    cx = (j+0.5) / f_k
                    cy = (i+0.5) / f_k
                    
                    s_k = self.min_sizes[k] / self.image_size #拿到待检测人脸的比例
                    mean += [cx, cy, s_k, s_k]  #中心坐标和边长的比例

                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]
            else:
                for i,j in product(range(f), repeat = 2):
                    f_k = self.image_size / self.steps[k]
                    cx = (j+0.5) / f_k
                    cy = (i+0.5) / f_k

                    s_k = self.min_sizes[k] / self.image_size
                    mean += [cx, cy, s_k, s_k]

                    s_k_1 = 40 / self.image_size
                    mean += [cx, cy, s_k_1, s_k_1]

                    s_k_2 = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_2, s_k_2]

                    s_k_3 = (self.min_sizes[k]+self.max_sizes[k])/(2*self.image_size)
                    mean += [cx, cy, s_k_3, s_k_3]

                    s_k_4 = (self.min_sizes[k]+self.max_sizes[k])/(2*self.image_size)
                    mean += [cx, cy, s_k_3, s_k_3]
                    
                    s_k_5 = (self.min_sizes[k]+self.max_sizes[k])/(2*self.image_size)
                    mean += [cx, cy, s_k_3, s_k_3]
        
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

if __name__ == "__main__":
    priorbox = Priorbox()
    x = priorbox.produce_priorboxes()
    print(x.size())# it should be torch.Size([896, 4])