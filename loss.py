"""
Loss Functions Module

Provides various loss functions for camouflaged object detection training,
including IoU loss, structure loss, DIoU loss, and Binary Dice loss.

Original Author: Haiyang Mei (mhy666@mail.dlut.edu.cn)
Reference: CVPR2021_PFNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## IoU Loss #############################
###################################################################

class IOU(torch.nn.Module):
    """IoU (Intersection over Union) Loss.

    Computes 1 - IoU between sigmoid-activated predictions and targets.
    """
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)
        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### Structure Loss #############################
###################################################################

class structure_loss(torch.nn.Module):
    """Structure Loss.

    Combines weighted binary cross-entropy and weighted IoU loss,
    where edge-aware weights emphasize boundary regions.
    """
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        # Edge-aware weight: higher weight near object boundaries
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


def Diou(bboxes1, bboxes2):
    """Compute Distance-IoU (DIoU) between two sets of bounding boxes.

    DIoU considers the overlap area, center distance, and diagonal distance
    of the enclosing box, providing more stable box regression compared to
    standard IoU and GIoU.

    Args:
        bboxes1: Bounding boxes of shape (N, 4) in [xmin, ymin, xmax, ymax] format.
        bboxes2: Bounding boxes of shape (M, 4) in [xmin, ymin, xmax, ymax] format.

    Returns:
        DIoU matrix of shape (N, M), values clamped to [-1, 1].
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


class BinaryDiceLoss(torch.nn.Module):
    """Binary Dice Loss.

    Computes the Dice loss for binary segmentation tasks.
    """
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # Batch size
        N = targets.size()[0]
        # Smoothing factor
        smooth = 1
        # Flatten spatial dimensions
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # Compute intersection
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # Average loss across the batch
        loss = 1 - N_dice_eff.sum() / N
        return loss