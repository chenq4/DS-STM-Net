import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

# Relative Volume Difference
def relative_volume_difference(output, target):

    # Calculate volumes
    volume1 = torch.sum(output)
    volume2 = torch.sum(target)
    if volume1 + volume2 == 0:
        return 0
    return (volume1 - volume2) / (0.5 * (volume1 + volume2))



import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt


def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def get_sensitivity(SR,GT,threshold=0.5):
    if torch.is_tensor(SR):
        SR = torch.sigmoid(SR)
    
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)*1+(GT==1)*1)==2
    FN = ((SR==0)*1+(GT==1)*1)==2
    FP = ((SR==1)*1+(GT==0)*1)==2
        
    SE = torch.sum(TP)/(torch.sum(TP+FN) + 1e-6)     
    Precision = torch.sum(TP)/(torch.sum(TP+FP) + 1e-6)
    return SE,Precision