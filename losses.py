import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class AdaptiveScaleLoss(nn.Module):
    def __init__(self,base_loss = BCEDiceLoss()):
        super().__init__()
        self.weights = torch.ones(3)
        self.base_loss = base_loss

    def forward(self, pred, label):
        pred_shrink_maps = pred[:, 0:1, :, :]
        pred_threshold_maps = pred[:, 1:2, :, :]
        pred_binary_maps = pred[:, 2:3, :, :]

        label_shrink_maps = label[:, 0:1, :, :]
        label_threshold_maps = label[:, 1:2, :, :]
        label_binary_maps = label[:, 2:3, :, :]
        
        loss_shrink_maps = self.base_loss(pred_shrink_maps, label_shrink_maps)
        loss_threshold_map = self.base_loss(pred_threshold_maps, label_threshold_maps)
        loss_binary_maps = self.base_loss(pred_binary_maps, label_binary_maps)
        return loss_shrink_maps + loss_threshold_map + loss_binary_maps


