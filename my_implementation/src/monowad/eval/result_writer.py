"""Write per-frame detections to KITTI result ``.txt`` files.

Ported near-verbatim from ``visualDet3D/data/kitti/utils.py::write_result_to_file``.
The only departure: the file name is passed in explicitly (``name``) rather than being
derived from a running integer index, so the caller can name files by the KITTI
``frame_id``. The KITTI evaluator (``kitti_common.get_label_annos``) globs ``*.txt`` and
sorts by ``int(stem)``, so naming by zero-padded frame id keeps detections aligned with
the ground-truth annotations loaded for the same (sorted) ids.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def write_result_to_file(
    base_result_path: str,
    name: str,
    scores,
    bbox_2d,
    bbox_3d_state_3d=None,
    thetas=None,
    obj_types=("Car", "Pedestrian", "Cyclist"),
    threshold: float = 0.4,
) -> None:
    """Write one frame's detections in KITTI label format.

    Args:
        base_result_path: directory the ``{name}.txt`` is written into.
        name: file stem (e.g. the zero-padded KITTI frame id ``"000123"``).
        scores: per-detection confidence (list / np.ndarray / cpu tensor).
        bbox_2d: ``[N, 4]`` 2D boxes ``x1, y1, x2, y2`` in *original* image resolution.
        bbox_3d_state_3d: ``[N, 7]`` ``[x, y, z, w, h, l, alpha]`` in camera coords
            (``y`` is the 3D-box *center*; KITTI wants the bottom-center, handled here).
            ``None`` for a 2D-only result.
        thetas: ``[N]`` global orientation (ry). ``None`` -> all ``-10``.
        obj_types: per-detection class name.
        threshold: drop detections scoring below this.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    text_to_write = ""
    with open(os.path.join(base_result_path, name + ".txt"), "w") as f:
        if bbox_3d_state_3d is None:
            bbox_3d_state_3d = np.ones([bbox_2d.shape[0], 7], dtype=int)
            bbox_3d_state_3d[:, 3:6] = -1
            bbox_3d_state_3d[:, 0:3] = -1000
            bbox_3d_state_3d[:, 6] = -10
        else:
            for i in range(len(bbox_2d)):
                # KITTI receives the bottom-center: shift y down by half the height.
                bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5 * bbox_3d_state_3d[i][4]

        if thetas is None:
            thetas = np.ones(bbox_2d.shape[0]) * -10

        if len(scores) > 0:
            for i in range(len(bbox_2d)):
                if scores[i] < threshold:
                    continue
                bbox = bbox_2d[i]
                text_to_write += (
                    "{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} "
                    "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} \n"
                ).format(
                    obj_types[i], bbox_3d_state_3d[i][-1], bbox[0], bbox[1], bbox[2], bbox[3],
                    bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                    bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                    thetas[i], scores[i],
                )
        f.write(text_to_write)
