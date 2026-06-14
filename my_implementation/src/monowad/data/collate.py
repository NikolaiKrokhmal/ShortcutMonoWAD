"""collate_fn for batching KittiMonoDataset samples.

Extracted from ``dataset.py`` so the ``DataModule`` (and anything wiring up a
``DataLoader``) can import the collation in isolation, without reaching into the
dataset class.

The shape of the returned tuple mirrors the legacy collate and is driven by what
the sample dict carries:

* train  (depth present)              -> 7-tuple: rgb, calib, labels, bbox2d, bbox3d, depth, foggy
* val    (foggy present, no depth)    -> 6-tuple: rgb, calib, labels, bbox2d, bbox3d, foggy
* test   (no foggy, no depth)         -> 5-tuple: rgb, calib, labels, bbox2d, bbox3d

Ragged per-frame label / box lists stay Python lists (variable N per frame).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def collate_fn(batch: List[dict]):
    """Stack a batch into the tuple the detector expects. Shape mirrors the legacy
    collate: train (with depth) -> 7-tuple, val (foggy, no depth) -> 8-tuple,
    test (no foggy, no depth) -> 7-tuple. Ragged per-frame label/box lists stay lists.

    The eval paths (val/test) additionally carry ``original_Ps`` (full-res calib, list of
    [3, 4] arrays) and ``frame_ids`` (KITTI ids, list of str) so ``validation_step`` can
    remap predicted 2D boxes back to original resolution and name KITTI result files.
    Training omits them (it never evaluates) to keep the train tuple unchanged."""
    rgb = np.stack([b["image"] for b in batch]).transpose(0, 3, 1, 2)  # [B, 3, H, W]
    rgb = torch.from_numpy(rgb).float()
    calib = torch.from_numpy(np.stack([b["calib"] for b in batch])).float()
    labels = [b["label"] for b in batch]
    bbox2ds = [b["bbox2d"] for b in batch]
    bbox3ds = [b["bbox3d"] for b in batch]

    has_foggy = batch[0]["foggy"] is not None
    depths = [b["depth"] for b in batch]

    if depths[0] is not None:  # training
        foggy = np.stack([b["foggy"] for b in batch]).transpose(0, 3, 1, 2)
        foggy = torch.from_numpy(foggy).float()
        depth = torch.from_numpy(np.stack(depths)).float()
        return rgb, calib, labels, bbox2ds, bbox3ds, depth, foggy

    # eval extras, carried as plain lists (ragged across frames, like labels/boxes)
    original_Ps = [b["original_P"] for b in batch]
    frame_ids = [b["frame_id"] for b in batch]

    if has_foggy:  # validation
        foggy = np.stack([b["foggy"] for b in batch]).transpose(0, 3, 1, 2)
        foggy = torch.from_numpy(foggy).float()
        return rgb, calib, labels, bbox2ds, bbox3ds, foggy, original_Ps, frame_ids

    return rgb, calib, labels, bbox2ds, bbox3ds, original_Ps, frame_ids  # test
