"""KITTI AP evaluation entry point (ported from visualDet3D/evaluator/kitti).

The numba rotated-IoU kernels in ``.rotate_iou`` JIT-compile *at import time* and need a
CUDA device, which transitively makes ``.eval`` / ``.evaluate`` GPU-only to import. So
those heavy modules are imported **lazily** inside :func:`evaluate_kitti` -- importing
``monowad.eval`` (or this module) stays cheap and CPU-safe; only scoring touches the GPU.

This wrapper takes the evaluated frame ids directly (a list) instead of a split-file path
like the legacy ``evaluate()``, and sorts both the detections and the ground truth by id
so the evaluator's positional gt<->dt matching stays aligned regardless of the order the
frames were produced in.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence


def evaluate_kitti(
    result_path: str,
    label_path: str,
    val_ids: Sequence,
    current_classes: Sequence,
    gpu: int = 0,
) -> List[str]:
    """Score KITTI AP for the result ``.txt`` files in ``result_path``.

    Args:
        result_path: dir of detection ``{id}.txt`` files (one per evaluated frame).
        label_path: KITTI ``label_2`` dir holding ground-truth ``{id}.txt`` files.
        val_ids: frame ids that were evaluated (int or str); GT is loaded for these.
        current_classes: class names or indices to score (e.g. ``["Car"]`` or ``[0]``).
        gpu: CUDA device index for the rotated-IoU kernels.

    Returns:
        One formatted AP-table string per class (as the legacy evaluator produced).
    """
    from numba import cuda  # lazy: triggers rotate_iou kernel compilation (needs GPU)

    from .eval import get_official_eval_result
    from .kitti_common import get_label_annos

    cuda.select_device(gpu)
    sorted_ids = sorted(int(i) for i in val_ids)
    dt_annos = get_label_annos(result_path)              # globs *.txt, sorted by int(stem)
    gt_annos = get_label_annos(label_path, sorted_ids)   # loaded in the same sorted order
    return [get_official_eval_result(gt_annos, dt_annos, c) for c in current_classes]


# AP lines look like e.g. ``3d   AP:89.12, 79.34, 78.10`` (easy, moderate, hard).
_AP_LINE = re.compile(r"^(bbox|bev|3d)\s+AP:\s*([\d.,\s]+)$", re.MULTILINE)


def parse_ap(result_text: str) -> Dict[str, List[float]]:
    """Pull the numeric AP triples out of one class's formatted result string.

    Returns a dict like ``{"bbox": [e, m, h], "bev": [...], "3d": [...]}`` with whatever
    metric lines are present. Useful for ``self.log()`` of the headline numbers.
    """
    out: Dict[str, List[float]] = {}
    for metric, values in _AP_LINE.findall(result_text):
        out[metric] = [float(v) for v in values.split(",") if v.strip()]
    return out
