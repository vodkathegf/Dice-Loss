# Dice Loss is not built-in in PyTorch, so I felt that I will need it within my projects.

import torch
import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Dice Loss 
        commonly used in semantic segmentation tasks

        params:
            -y_true: (Tensor), ground truth segmentation mask
            -y_pred: (Tensor), pedicted segmentation mask

        returns:
            -loss: (Tensor), calculated Dice Loss
        """
        # the Dice Coefficient needs to be calculated at first.
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice_coefficient = 2.0 * intersection / (union + 1e-8) #the epsilon term is for in case of union = 0
        return 1 - dice_coefficient

