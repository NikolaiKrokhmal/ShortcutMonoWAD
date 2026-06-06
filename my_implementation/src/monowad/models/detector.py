"""MonoWAD detector network.

Instantiated via Hydra (_target_: monowad.models.detector.MonoWAD), replacing the
legacy DETECTOR_DICT registry. Phase 3 of PLAN.md.
"""
from __future__ import annotations

import torch.nn as nn


class MonoWAD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
