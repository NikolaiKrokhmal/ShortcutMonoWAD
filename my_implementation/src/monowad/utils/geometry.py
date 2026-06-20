"""Camera-model geometry helpers for KITTI mono 3D detection.

Ported verbatim (behaviour-wise) from visualDet3D/utils/utils.py and
visualDet3D/networks/utils/utils.py so the monowad package no longer depends on the
legacy tree. These are pure camera-projection utilities — shared by the dataset
(reprojection) and, later, the anchors / detection head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def alpha2theta_3d(alpha, x, z, P2):
    """Convert observation angle ``alpha`` to global yaw ``theta`` given 3D position.

    Args:
        alpha (torch.Tensor | float | np.ndarray): size [...]
        x, z  (same type/shape as alpha): 3D position components
        P2    (torch.Tensor | np.ndarray): [3, 4] calibration
    Returns:
        theta, same type/shape as alpha
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        return alpha + torch.atan2(x + offset, z)
    return alpha + np.arctan2(x + offset, z)


def theta2alpha_3d(theta, x, z, P2):
    """Convert global yaw ``theta`` to observation angle ``alpha`` given 3D position.

    Args:
        theta (torch.Tensor | float | np.ndarray): size [...]
        x, z  (same type/shape as theta): 3D position components
        P2    (torch.Tensor | np.ndarray): [3, 4] calibration
    Returns:
        alpha, same type/shape as theta
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(theta, torch.Tensor):
        return theta - torch.atan2(x + offset, z)
    return theta - np.arctan2(x + offset, z)


class BBox3dProjector(nn.Module):
    """Project 3D boxes into the image.

    forward:
        input:
            bbox_3d [N, 7]: unnormalized x, y, z, w, h, l, alpha
            tensor_p2 [3, 4]: calibration
        output:
            abs_corners [N, 8, 3]: corner points in camera frame
            homo_coord  [N, 8, 3]: corner points in image (homogeneous, /z) frame
            thetas      [N]:       global yaw per box
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "corner_matrix",
            torch.tensor(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                ]
            ).float(),
        )  # [8, 3]

    def forward(self, bbox_3d, tensor_p2):
        relative_eight_corners = 0.5 * self.corner_matrix * bbox_3d[:, 3:6].unsqueeze(1)  # [N, 8, 3]
        thetas = alpha2theta_3d(bbox_3d[..., 6], bbox_3d[..., 0], bbox_3d[..., 2], tensor_p2)
        _cos = torch.cos(thetas).unsqueeze(1)  # [N, 1]
        _sin = torch.sin(thetas).unsqueeze(1)  # [N, 1]
        rotated_corners_x, rotated_corners_z = (
            relative_eight_corners[:, :, 2] * _cos + relative_eight_corners[:, :, 0] * _sin,
            -relative_eight_corners[:, :, 2] * _sin + relative_eight_corners[:, :, 0] * _cos,
        )  # relative_eight_corners == [N, 8, 3]
        rotated_corners = torch.stack(
            [rotated_corners_x, relative_eight_corners[:, :, 1], rotated_corners_z], dim=-1
        )  # [N, 8, 3]
        abs_corners = rotated_corners + bbox_3d[:, 0:3].unsqueeze(1)  # [N, 8, 3]
        camera_corners = torch.cat(
            [abs_corners, abs_corners.new_ones([abs_corners.shape[0], self.corner_matrix.shape[0], 1])],
            dim=-1,
        ).unsqueeze(3)  # [N, 8, 4, 1]
        camera_coord = torch.matmul(tensor_p2, camera_corners).squeeze(-1)  # [N, 8, 3]
        homo_coord = camera_coord / (camera_coord[:, :, 2:] + 1e-6)  # [N, 8, 3]
        return abs_corners, homo_coord, thetas


def calc_iou(a, b):
    """Pairwise 2D IoU between anchor boxes ``a`` [N, 4] and gt boxes ``b`` [M, 4].

    Boxes are [x1, y1, x2, y2]. Returns [N, M]. Ported from
    visualDet3D/networks/utils/utils.py.
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    return intersection / ua


class ClipBoxes(nn.Module):
    """Clip 2D box coords to the image bounds. Mutates and returns ``boxes``.

    Ported from visualDet3D/networks/utils/utils.py.
    """

    def __init__(self, width=None, height=None) -> None:
        super().__init__()

    def forward(self, boxes, img):
        _, _, height, width = img.shape
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)
        boxes[:, 2] = torch.clamp(boxes[:, 2], max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], max=height)
        return boxes


class BackProjection(nn.Module):
    """Back-project image-plane 3D box centers to camera coords using P2.

    forward:
        bbox3d [N, 7]: homo_x, homo_y, z, w, h, l, alpha
        p2 [3, 4]
        return [N, 7]: x3d, y3d, z, w, h, l, alpha
    Ported from visualDet3D/networks/utils/utils.py.
    """

    def forward(self, bbox3d, p2):
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]

        z3d = bbox3d[:, 2:3]  # [N, 1]
        x3d = (bbox3d[:, 0:1] * z3d - cx * z3d - tx) / fx  # [N, 1]
        y3d = (bbox3d[:, 1:2] * z3d - cy * z3d - ty) / fy  # [N, 1]
        return torch.cat([x3d, y3d, bbox3d[:, 2:]], dim=1)
