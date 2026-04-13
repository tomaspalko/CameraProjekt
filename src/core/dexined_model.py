"""DexiNed — Dense Extreme Inception Network for Edge Detection.

PyTorch model architecture.  Based on the official implementation:
  https://github.com/xavysp/DexiNed/blob/master/model.py

Pre-trained weights (BIPED dataset):
  https://huggingface.co/xavysp/dexined  (recommended, auto-downloaded)
  or manually from the official Google Drive link in the README.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int,
                 use_bn: bool = True, use_act: bool = True) -> None:
        super().__init__()
        self.use_bn  = use_bn
        self.use_act = use_act
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_features) if use_bn else nn.Identity()
        self.act  = nn.GELU() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int | None = None,
                 stride: int = 1, use_act: bool = True) -> None:
        super().__init__()
        out_features = out_features or mid_features
        self.conv1 = nn.Conv2d(in_features,  mid_features, 3, padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_features)
        self.bn2   = nn.BatchNorm2d(out_features)
        self.act   = nn.GELU()
        self.use_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x) if self.use_act else x


class _DenseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=2, dilation=2, bias=False)
        self.bn   = nn.BatchNorm2d(out_features)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.act(self.bn(self.conv(x)))], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_dense: int) -> None:
        super().__init__()
        layers = []
        ch = in_features
        for _ in range(n_dense):
            layers.append(_DenseLayer(ch, out_features))
            ch += out_features
        self.layers = nn.ModuleList(layers)
        self.transition = _SingleConvBlock(ch, out_features, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.transition(x)


class _UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        self.up_scale = up_scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# DexiNed main model
# ---------------------------------------------------------------------------

class DexiNed(nn.Module):
    """Dense Extreme Inception Network for Edge Detection (DexiNed).

    Input:  float32 tensor (B, 3, H, W), BGR, mean-subtracted
    Output: list of 7 tensors (B, 1, H, W) — 6 intermediate + 1 fused
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.block_1 = _DoubleConvBlock(3,  32,  64, stride=2)
        self.block_2 = _DoubleConvBlock(64, 128, use_act=False)

        # Dense blocks
        self.dblock_3 = _DenseBlock(128, 256, n_dense=2)
        self.dblock_4 = _DenseBlock(256, 512, n_dense=3)
        self.dblock_5 = _DenseBlock(512, 512, n_dense=3)
        self.dblock_6 = _DenseBlock(512, 256, n_dense=3)

        # Side convolutions (skip connections → 1-ch edge map)
        self.side_1 = _UpConvBlock(64,  1)
        self.side_2 = _UpConvBlock(128, 1)
        self.side_3 = _UpConvBlock(256, 1)
        self.side_4 = _UpConvBlock(512, 1)
        self.side_5 = _UpConvBlock(512, 1)
        self.side_6 = _UpConvBlock(256, 1)

        # Fuse 6 side outputs
        self.conv_fuse = nn.Conv2d(6, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]

        # Encoder
        b1 = self.block_1(x)
        b2 = self.block_2(b1)

        # Pooling between dense blocks
        b3 = self.dblock_3(F.max_pool2d(b2, 3, stride=2, padding=1))
        b4 = self.dblock_4(F.max_pool2d(b3, 3, stride=2, padding=1))
        b5 = self.dblock_5(F.max_pool2d(b4, 3, stride=2, padding=1))
        b6 = self.dblock_6(F.max_pool2d(b5, 3, stride=2, padding=1))

        # Side predictions (all upsampled to input resolution)
        s1 = self.side_1(b1, H, W)
        s2 = self.side_2(b2, H, W)
        s3 = self.side_3(b3, H, W)
        s4 = self.side_4(b4, H, W)
        s5 = self.side_5(b5, H, W)
        s6 = self.side_6(b6, H, W)

        # Fused output
        fused = self.conv_fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))

        return [s1, s2, s3, s4, s5, s6, fused]
