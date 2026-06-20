"""Losses for the MonoWAD detection head.

Consolidates the legacy detection losses (visualDet3D/networks/heads/losses.py) and
depth losses (visualDet3D/networks/heads/depth_losses.py). Behaviour-faithful ports,
with the only change being device-safety: the legacy code hard-coded ``.cuda()`` on
freshly-allocated zero tensors; here we allocate ``zeros_like`` so the loss follows
the input's device (required under Lightning, which owns device placement).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("balance_weights", balance_weights)

    def forward(self, classification, targets, gamma=None, balance_weights=None):
        """
        input:
            classification :[..., num_classes]  linear output
            targets        :[..., num_classes] == -1(ignored), 0, 1
        return:
            cls_loss       :[..., num_classes]  loss with 0 in trimmed/ignored indexes
        """
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = torch.sigmoid(classification)
        focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        bce = -(targets * F.logsigmoid(classification)) * balance_weights - (
            (1 - targets) * F.logsigmoid(-classification)
        )
        cls_loss = focal_weight * bce

        # neglect 0.3 < iou < 0.4 anchors (targets == -1)
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))
        # clamp tiny losses to prevent overfitting on over-confident/correct ones
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros_like(cls_loss), cls_loss)
        return cls_loss


class ModifiedSmoothL1Loss(nn.Module):
    def __init__(self, L1_regression_alpha: float):
        super().__init__()
        self.alpha = L1_regression_alpha

    def forward(self, normed_targets, pos_reg):
        regression_diff = torch.abs(normed_targets - pos_reg)
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / self.alpha),
            0.5 * self.alpha * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.alpha,
        )
        # clip tiny residuals to avoid overfitting
        regression_loss = torch.where(
            torch.le(regression_diff, 0.01),
            torch.zeros_like(regression_loss),
            regression_loss,
        )
        return regression_loss


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """Discretize a continuous depth map into bin indices (UD/LID/SID).

    Ported from visualDet3D/networks/heads/depth_losses.py. MonoWAD uses ``mode="LID"``.
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        import math

        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # remove indices outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = 0
        indices = indices.type(torch.int64)
    return indices


class DepthFocalLoss(object):
    def __init__(self, max_depth=192, start_depth=0, focal_coefficient=0.0):
        self.max_depth = max_depth
        self.start_depth = start_depth
        self.end_depth = start_depth + max_depth - 1
        self.focal_coefficient = focal_coefficient
        self.eps = 1e-40
        self.variance = 0.5

    def __call__(self, estCost, gtDepth):
        N, C, H, W = estCost.shape
        scaled_gtDepth = gtDepth.clone()  # N, 1, H, W

        lower_bound = self.start_depth
        upper_bound = lower_bound + int(self.max_depth)
        mask = (scaled_gtDepth > lower_bound) & (scaled_gtDepth < upper_bound)
        mask = mask.detach_().type_as(scaled_gtDepth)

        if mask.sum() < 1.0:
            scaled_gtProb = torch.zeros_like(estCost)  # let this sample have loss 0
        else:
            gtDepth = scaled_gtDepth * mask

            index = torch.linspace(self.start_depth, self.end_depth, self.max_depth)
            index = index.to(gtDepth.device)
            index = index.repeat(N, H, W, 1).permute(0, 3, 1, 2).contiguous()

            mask = (gtDepth > self.start_depth) & (gtDepth < self.end_depth)
            mask = mask.detach().type_as(gtDepth)
            gtDepth = gtDepth * mask

            scaled_distance = (-torch.abs(index - gtDepth)) / self.variance
            probability = F.softmax(scaled_distance, dim=1)
            scaled_gtProb = probability * mask + self.eps

        estProb = F.log_softmax(estCost, dim=1)
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estProb) * weight * mask.float()).sum(dim=1, keepdim=True).mean()
        return loss
