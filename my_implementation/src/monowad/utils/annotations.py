"""Annotation compounding for the detection head.

Ported verbatim (logic-wise) from ``visualDet3D/utils/utils.py:compound_annotation``.
Used by the LightningModule's ``training_step`` to pack a batch's ragged per-frame
labels + 2D/3D boxes into the dense ``[B, max_length, 12]`` target tensor the head's
``loss`` consumes.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def compound_annotation(
    labels: List[List[str]],
    max_length: int,
    bbox2d: List[np.ndarray],
    bbox_3d: List[np.ndarray],
    obj_types: Sequence[str],
) -> np.ndarray:
    """Compound ragged per-frame annotations into one padded array.

    Args:
        labels: per-frame list of class-name strings.
        max_length: padded object count (max over the batch); can vary per step.
        bbox2d: per-frame ``[N, 4]`` arrays ``[left, top, right, bottom]``.
        bbox_3d: per-frame ``[N, 7]`` arrays ``[cam_x, cam_y, z, w, h, l, alpha]``.
        obj_types: ordered class names; the class index is ``obj_types.index(name)``.

    Returns:
        ``np.ndarray`` of shape ``[batch, max_length, 12]`` laid out as
        ``[x1, y1, x2, y2, cls_index, cx, cy, z, w, h, l, alpha]``. Padded slots
        (and empty frames) are filled with ``-1`` (so ``cls_index == -1`` flags empty).
    """
    obj_types = list(obj_types)
    annotations = np.ones([len(labels), max_length, bbox_3d[0].shape[-1] + 5]) * -1
    for i in range(len(labels)):
        label = labels[i]
        for j in range(len(label)):
            annotations[i, j] = np.concatenate(
                [bbox2d[i][j], [obj_types.index(label[j])], bbox_3d[i][j]]
            )
    return annotations
