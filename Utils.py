import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def dice_ratio(seg, gt):
    """
    define the dice ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    same = (seg * gt).sum()

    dice = 2*float(same)/float(gt.sum() + seg.sum())

    return dice
