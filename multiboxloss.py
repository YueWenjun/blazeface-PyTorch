# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from box_utils import match, log_sum_exp


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

    def __init__(self, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
    
    def forward(self, predictions, prior_boxes, targets):
        """
        Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
                                 and prior boxes from blazefaceNet.
                loc shape:  torch.size(batch_size, num_prior_boxes, 4)
                conf shape: torch.size(batch_size, num_prior_boxes, num_classes)
                prior_boxes shape: torch.size(num_prior_boxes, 4)
            
            targets (Tensor): a doubletensor of ground truth boxes and labels for a batch
        """
        loc_data, conf_data = predictions
        batch_size = loc_data.size(0)
        num_prior_boxes = loc_data.size(1)
        priorboxes = prior_boxes

        # match prior_boxes with ground truth boxes
        loc_target = torch.Tensor(batch_size, num_prior_boxes, 4)
        conf_target = torch.LongTensor(batch_size, num_prior_boxes)
        for idx in range(batch_size):
            truths = targets[idx].data
            labels = torch.ones([truths.size(0),1])
            print("multiboxloss中的truths和labels: ",truths,labels)
            defaults = priorboxes.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_target, conf_target, idx)
        if self.use_gpu == True:
            loc_target = loc_target.cuda()
            conf_target = conf_target.cuda()
        loc_target = Variable(loc_target, requires_grad=False)
        conf_target = Variable(conf_target, requires_grad=False)

        #----------------------------------------------------
        pos = conf_target > 0
        print(conf_target.shape)
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_target.view(-1, 1))

        # Hard Negative Mining
        #loss_c = loss_c.view(pos.size()[0], pos.size()[1]) #add line 
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        #loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_target[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        #----------------------------------------------------

        
        N = num_pos.data.sum()
        print("N: ", N, type(N))
        loss_l /= N
        loss_C /= N
        return loss_l, loss_c

if __name__ == "__main__":
    criterion = MultiBoxLoss(overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=3, neg_pos=0.5, neg_overlap=False, encode_target=True)