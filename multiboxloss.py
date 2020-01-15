# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes
           with (default) 'priorboxes'.
        2) Produce locatization target by 'encoding' variance into offsets of ground
           truth boxes and their matched 'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x, l ,g)) / N
        where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by a which is set to 1 by cross val
        Arg:
            c: class confidences,
            l: predicted boxed
            g: ground truth boxes
            N: number of matched default boxes
    """

    def __init__(self,num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        