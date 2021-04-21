"""
Util classes & methods for computing loss.
Reference: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _neg_loss(pred, gt):
    """
    Focal loss same as CornerNet.
    Args:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = th.pow(1 - gt, 4)

    loss = 0.

    pos_loss = th.log(pred) * th.pow(1 - pred, 2) * pos_inds
    neg_loss = th.log(1 - pred) * th.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    
    return loss

def _reg_loss(regr, gt_regr, mask):
    """
    Smooth L1 regression loss
        Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    
    return regr_loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegLoss(nn.Module):
    """
    Regression loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """
    def __init__(self):
        super(RegLoss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss