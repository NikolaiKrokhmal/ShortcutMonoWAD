"""Small reusable layers for the detection head.

Ported from visualDet3D/networks/lib/blocks.py — only ``AnchorFlatten`` is used by
the MonoWAD head, so the rest of the legacy block zoo is intentionally dropped.
"""
from __future__ import annotations

import torch.nn as nn


class AnchorFlatten(nn.Module):
    """Reshape dense anchor outputs to per-anchor rows.

    Forward args:
        x: [B, num_anchors * output_channel, H, W]
    Returns:
        [B, num_anchors * H * W, output_channel]
    """

    def __init__(self, num_output_channel: int) -> None:
        super().__init__()
        self.num_output_channel = num_output_channel

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, self.num_output_channel)
        return x
