import torch
import torch.nn as nn


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)
