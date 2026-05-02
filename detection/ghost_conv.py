"""GhostConv: эффективный свёрточный блок из GhostNet.

Han, K. et al. "GhostNet: More Features from Cheap Operations." CVPR 2020.
https://arxiv.org/abs/1911.11907

Идея: часть выходных карт получается обычной свёрткой (primary), оставшиеся —
дешёвой depthwise-свёрткой по primary-картам. Так мы получаем те же
``out_channels`` при заметно меньшем числе FLOPs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GhostConv(nn.Module):
    """Замена nn.Conv2d на GhostConv-блок."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        primary_channels = math.ceil(out_channels / ratio)
        cheap_channels = out_channels - primary_channels
        self.cheap_channels = cheap_channels

        activation: nn.Module = nn.ReLU(inplace=True) if relu else nn.Identity()
        cheap_activation: nn.Module = nn.ReLU(inplace=True) if relu else nn.Identity()

        # Основная свёртка
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                primary_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(primary_channels),
            activation,
        )

        # Дешёвая depthwise-свёртка над primary-картами
        if cheap_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    primary_channels,
                    cheap_channels,
                    dw_size,
                    stride=1,
                    padding=dw_size // 2,
                    groups=math.gcd(primary_channels, cheap_channels),
                    bias=False,
                ),
                nn.BatchNorm2d(cheap_channels),
                cheap_activation,
            )
        else:
            self.cheap_operation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        if self.cheap_channels == 0:
            return x1[:, : self.out_channels, :, :]
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]


def replace_conv_with_ghost(model: nn.Module, min_channels: int = 16) -> nn.Module:
    """Рекурсивно заменить подходящие nn.Conv2d на GhostConv.

    Заменяются только полноразмерные (kernel_size != 1) обычные (groups == 1)
    свёртки с достаточным числом каналов на входе и выходе.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            ks = child.kernel_size
            kernel_size = ks[0] if isinstance(ks, tuple) else ks
            stride_t = child.stride
            stride = stride_t[0] if isinstance(stride_t, tuple) else stride_t

            if (
                child.in_channels >= min_channels
                and child.out_channels >= min_channels
                and child.groups == 1
                and kernel_size != 1
            ):
                ghost = GhostConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                setattr(model, name, ghost)
                continue

        replace_conv_with_ghost(child, min_channels=min_channels)

    return model
