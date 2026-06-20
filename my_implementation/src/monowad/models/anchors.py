"""Anchor module for the MonoWAD detection head.

Ported from visualDet3D/networks/heads/anchors.py. Two deliberate changes vs. legacy:

- **Device-safe.** The legacy code hard-coded ``device="cuda"`` for the ``useful_mask``
  and ``.cuda()`` for the numpy-input branch. Here masks/anchors follow the input
  tensor's device so the module works under Lightning (CPU or any GPU).
- **Flat prior path.** Legacy loaded ``<preprocessed_path>/training/anchor_{mean,std}_<cls>.npy``.
  The HDF5 migration stores the priors flat next to the split data, so we load
  ``<preprocessed_path>/anchor_{mean,std}_<cls>.npy`` directly.

The precomputed ``.npy`` priors are per-(size, ratio) [z, sin2a, cos2a, w, h, l] mean/std
statistics; they must be generated offline (legacy det_precompute step).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    """Anchor module for multi-level dense output."""

    def __init__(
        self,
        preprocessed_path: str,
        pyramid_levels: List[int],
        strides: List[float],
        sizes: List[float],
        ratios,
        scales,
        readConfigFile: bool = True,
        obj_types: List[str] = (),
        filter_anchors: bool = True,
        filter_y_threshold_min_max: Optional[Tuple[float, float]] = (-0.5, 1.8),
        filter_x_threshold: Optional[float] = 40.0,
        anchor_prior_channel: int = 6,
    ) -> None:
        super().__init__()
        # coerce config sequences (lists / OmegaConf ListConfig / ndarray) to fixed types
        self.pyramid_levels = list(pyramid_levels)
        self.strides = list(strides)
        self.sizes = list(sizes)
        self.ratios = np.array(ratios, dtype=np.float64)
        self.scales = np.array(scales, dtype=np.float64)
        self.obj_types = list(obj_types)

        self.shape = None
        self.P2 = None
        self.readConfigFile = readConfigFile
        self.scale_step = 1 / (np.log2(self.scales[1]) - np.log2(self.scales[0]))
        if self.readConfigFile:
            self.anchors_mean_original = np.zeros(
                [len(self.obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), anchor_prior_channel]
            )
            self.anchors_std_original = np.zeros(
                [len(self.obj_types), len(self.scales) * len(self.pyramid_levels), len(self.ratios), anchor_prior_channel]
            )
            for i in range(len(self.obj_types)):
                npy_file = os.path.join(preprocessed_path, "anchor_mean_{}.npy".format(self.obj_types[i]))
                self.anchors_mean_original[i] = np.load(npy_file)  # [30, 2, 6] [z, sin2a, cos2a, w, h, l]
                std_file = os.path.join(preprocessed_path, "anchor_std_{}.npy".format(self.obj_types[i]))
                self.anchors_std_original[i] = np.load(std_file)

        self.filter_y_threshold_min_max = filter_y_threshold_min_max
        self.filter_x_threshold = filter_x_threshold

    def anchors2indexes(self, anchors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sizes = np.sqrt((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]))
        sizes_diff = sizes - (np.array(self.sizes) * self.scales)[:, np.newaxis]
        sizes_int = np.argmin(np.abs(sizes_diff), axis=0)

        ratio = (anchors[:, 3] - anchors[:, 1]) / (anchors[:, 2] - anchors[:, 0])
        ratio_diff = ratio - self.ratios[:, np.newaxis]
        ratio_int = np.argmin(np.abs(ratio_diff), axis=0)
        return sizes_int, ratio_int

    def forward(self, image: torch.Tensor, calibs=(), is_filtering=False):
        shape = image.shape[2:]
        if self.shape is None or not (shape == self.shape):
            self.shape = image.shape[2:]

            image_shape = np.array(image.shape[2:])
            image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

            # compute anchors over all pyramid levels
            all_anchors = np.zeros((0, 4)).astype(np.float32)
            for idx, _p in enumerate(self.pyramid_levels):
                anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
                all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

            if self.readConfigFile:
                sizes_int, ratio_int = self.anchors2indexes(all_anchors)
                self.anchor_means = image.new(self.anchors_mean_original[:, sizes_int, ratio_int])  # [types, N, 6]
                self.anchor_stds = image.new(self.anchors_std_original[:, sizes_int, ratio_int])  # [types, N, 6]
                self.anchor_mean_std = torch.stack([self.anchor_means, self.anchor_stds], dim=-1).permute(
                    1, 0, 2, 3
                )  # [N, types, 6, 2]

            all_anchors = np.expand_dims(all_anchors, axis=0)
            if isinstance(image, torch.Tensor):
                self.anchors = image.new(all_anchors.astype(np.float32))  # [1, N, 4]
            elif isinstance(image, np.ndarray):
                self.anchors = torch.tensor(all_anchors.astype(np.float32))
            self.anchors_image_x_center = self.anchors[0, :, 0:4:2].mean(dim=1)  # [N]
            self.anchors_image_y_center = self.anchors[0, :, 1:4:2].mean(dim=1)  # [N]

        if calibs is not None and len(calibs) > 0:
            P2 = calibs  # [B, 3, 4]
            if self.P2 is not None and torch.all(self.P2 == P2) and self.P2.shape == P2.shape:
                if self.readConfigFile:
                    return self.anchors, self.useful_mask, self.anchor_mean_std
                return self.anchors, self.useful_mask

            self.P2 = P2
            fy = P2[:, 1:2, 1:2]  # [B, 1, 1]
            cy = P2[:, 1:2, 2:3]  # [B, 1, 1]
            cx = P2[:, 0:1, 2:3]  # [B, 1, 1]
            N = self.anchors.shape[1]
            if self.readConfigFile and is_filtering:
                anchors_z = self.anchor_means[:, :, 0]  # [types, N]
                world_x3d = (self.anchors_image_x_center * anchors_z - anchors_z.new(cx) * anchors_z) / anchors_z.new(fy)
                world_y3d = (self.anchors_image_y_center * anchors_z - anchors_z.new(cy) * anchors_z) / anchors_z.new(fy)
                self.useful_mask = torch.any(
                    (world_y3d > self.filter_y_threshold_min_max[0])
                    * (world_y3d < self.filter_y_threshold_min_max[1])
                    * (world_x3d.abs() < self.filter_x_threshold),
                    dim=1,
                )  # [B, N] any one type lies in target range
            else:
                self.useful_mask = torch.ones([len(P2), N], dtype=torch.bool, device=self.anchors.device)
            if self.readConfigFile:
                return self.anchors, self.useful_mask, self.anchor_mean_std
            return self.anchors, self.useful_mask
        return self.anchors

    @property
    def num_anchors(self):
        return len(self.pyramid_levels) * len(self.ratios) * len(self.scales)

    @property
    def num_anchor_per_scale(self):
        return len(self.ratios) * len(self.scales)


def generate_anchors(base_size=16, ratios=None, scales=None):
    """Enumerate anchor windows over (aspect ratio x scale) for a reference window."""
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    return all_anchors.reshape((K * A, 4))
