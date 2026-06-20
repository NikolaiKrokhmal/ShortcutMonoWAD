"""MonoWAD top-level detector (the legacy ``MonoWAD_3D``).

Ported from visualDet3D/networks/detectors/Detector.py. Wires the MonoWAD core
(backbone -> weather codebook -> diffusion -> depth-aware transformer) to the
anchor-based 3D head, and owns the depth loss.

Replaces the legacy ``@DETECTOR_DICT.register_module`` registry: instantiated via
Hydra (``_target_: monowad.models.detector.MonoWAD_3D``). ``network_cfg`` is a
mapping-like object (plain dict / OmegaConf DictConfig) carrying ``obj_types``,
``mono_backbone`` and ``head``.

I/O contract is unchanged from legacy so the training/eval glue (now in the
LightningModule) is a drop-in:
    train_forward -> (cls_loss, reg_loss, l_proposed, loss_dict)
    test_forward  -> (scores, bboxes, cls_indexes)   [batch size 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .MonoWAD import MonoWAD
from .heads.detection_3d_head import AnchorBasedDetection3DHead
from .losses import DepthFocalLoss, bin_depths


class MonoWAD_3D(nn.Module):
    def __init__(self, network_cfg) -> None:
        super().__init__()
        self.obj_types = network_cfg.obj_types
        self.build_head(network_cfg)
        self.build_core(network_cfg)
        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.mono_core = MonoWAD(network_cfg.mono_backbone)

    def build_head(self, network_cfg):
        self.bbox_head = AnchorBasedDetection3DHead(**(network_cfg.head))
        self.depth_loss = DepthFocalLoss(96)

    def train_forward(self, left_images, annotations, P2, depth_gt=None, foggy_images=None):
        features, depth, l_proposed = self.mono_core(
            dict(image=left_images, P2=P2, foggy=foggy_images, training=True)
        )
        depth_output = depth

        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2, image=left_images))
        anchors = self.bbox_head.get_anchor(left_images, P2)
        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        depth_gt = bin_depths(depth_gt, mode="LID", depth_min=1, depth_max=80, num_bins=96, target=True)

        if reg_loss.mean() > 0 and depth_gt is not None and depth_output is not None:
            depth_gt = depth_gt.unsqueeze(1)
            depth_loss = 1.0 * self.depth_loss(depth_output, depth_gt)
            loss_dict["depth_loss"] = depth_loss
            reg_loss += depth_loss
            self.depth_output = depth_output.detach()
        else:
            loss_dict["depth_loss"] = torch.zeros_like(reg_loss)

        loss_dict["proposed_loss"] = l_proposed
        return cls_loss, reg_loss, l_proposed, loss_dict

    def test_forward(self, left_images, P2, foggy_images=None, eval_weather_type: str = "clear"):
        assert left_images.shape[0] == 1  # image batch size 1 recommended for testing
        inputs = left_images if eval_weather_type == "clear" else foggy_images
        features, _ = self.mono_core(dict(image=inputs, P2=P2, foggy=foggy_images, training=False))

        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2, image=left_images))
        anchors = self.bbox_head.get_anchor(left_images, P2)
        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes

    def forward(self, inputs, eval_weather_type: str = "clear"):
        if isinstance(inputs, list) and len(inputs) >= 4:
            return self.train_forward(*inputs)
        return self.test_forward(*inputs, eval_weather_type=eval_weather_type)
