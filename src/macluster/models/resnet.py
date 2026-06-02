"""CIFAR-style ResNet (He et al., 2016) in MLX.

ResNet-{6n+2}: a 3x3 stem, then three stages of ``n`` BasicBlocks with channel
widths 16/32/64 and stride-2 downsampling between stages, global average pool,
and a linear head. ``resnet20`` (n=3) is the small model, ``resnet56`` (n=9)
the large one — they span ~3x in parameters / FLOPs, which is the compute-load
axis the proposal asks for.

NOTE: MLX convolutions use NHWC layout, so inputs are (N, 32, 32, 3).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.0)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_c)
        self.use_proj = stride != 1 or in_c != out_c
        if self.use_proj:
            self.proj_conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            self.proj_bn = nn.BatchNorm(out_c)

    def __call__(self, x: mx.array) -> mx.array:
        out = _relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.proj_bn(self.proj_conv(x)) if self.use_proj else x
        return _relu(out + identity)


class ResNetCIFAR(nn.Module):
    def __init__(self, n: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(16)
        self.layer1 = self._stage(16, 16, n, stride=1)
        self.layer2 = self._stage(16, 32, n, stride=2)
        self.layer3 = self._stage(32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _stage(in_c: int, out_c: int, n: int, stride: int) -> list[BasicBlock]:
        blocks = [BasicBlock(in_c, out_c, stride)]
        blocks += [BasicBlock(out_c, out_c, 1) for _ in range(n - 1)]
        return blocks

    def __call__(self, x: mx.array) -> mx.array:
        x = _relu(self.bn1(self.conv1(x)))
        for blk in self.layer1:
            x = blk(x)
        for blk in self.layer2:
            x = blk(x)
        for blk in self.layer3:
            x = blk(x)
        x = mx.mean(x, axis=(1, 2))  # global average pool over H, W (NHWC)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=3, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=9, num_classes=num_classes)


def num_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(p.size for _, p in tree_flatten(model.trainable_parameters())))
