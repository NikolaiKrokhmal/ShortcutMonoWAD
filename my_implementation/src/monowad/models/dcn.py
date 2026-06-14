"""Modulated deformable convolution (DCNv2), backed by torchvision.

Drop-in replacement for the legacy custom-CUDA ``ModulatedDeformConvPack`` from
``visualDet3D/networks/lib/ops/dcn``. That version required compiling a CUDA
extension (``deform_conv_ext``); here we delegate to ``torchvision.ops.deform_conv2d``,
which ships a CPU + CUDA modulated implementation and needs no build step.

Parameter layout is identical to the legacy module — ``weight``, ``bias``,
``conv_offset.{weight,bias}`` — so original MonoWAD checkpoints load unchanged.
The offset/mask convention also matches: ``conv_offset`` emits ``3 * k * k`` channels
per deformable group, split into (offset_x, offset_y, mask) with a sigmoid on mask.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class ModulatedDeformConvPack(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        # zero-init the offset/mask predictor → starts as a plain conv (legacy behaviour)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
