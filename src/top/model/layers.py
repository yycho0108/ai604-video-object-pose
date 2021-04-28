#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ConvUpsample(nn.Module):
    """
    2x Upsampling block via transposed convolution.
    Partly adopted from :
    "https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/msra_resnet.py"
    """

    @dataclass
    class Settings(Serializable):
        kernel_size: int = 4

    def __init__(self, opts: Settings, c_in: int, c_out: int):
        super().__init__()
        self.opts = opts
        self.conv = nn.ConvTranspose2d(c_in, c_out, opts.kernel_size,
                                       stride=2,
                                       padding=1,
                                       output_padding=0,
                                       bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """
    NOTE(ycho): Not sure what to actually do, so it's just a placeholder for now.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class KeypointLayer2D(nn.Module):
    """
    Placeholder for classification-style keypoint prediction in 2D.
    Relatively straightforward implementation: infers dense per-pixel keypoint.
    Probably works-ish, but should be replaced with a more principled alternative.

    TODO(ycho): Consider representing keypoint output as {cls, offset}.
    """

    @dataclass
    class Settings(Serializable):
        kernel_size: int = 3
        num_keypoints: int = (8 + 1 + 1)

    def __init__(self, opts: Settings, c_in: int):
        super().__init__()
        self.opts = opts
        self.conv_kpt_cls = nn.Conv2d(
            c_in, opts.num_keypoints, opts.kernel_size,
            padding=opts.kernel_size // 2)

    def forward(self, inputs):
        x = inputs
        c_logit = self.conv_kpt_cls(x)
        return (c_logit)


class KeypointLayer3D(nn.Module):
    """
    CenterNet-style 3D keypoint regressor.
    TODO(ycho): Finish implementation.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.conv_hm = nn.Conv2d(c_in, 1, 3, padding=1)
        self.conv_invd = nn.Conv2d(c_in, 1, 3, padding=1)

    def forward(self, inputs):
        # heatmap ...
        hm = F.sigmoid(self.conv_hm(inputs))
        # depth ...
        depth = 1.0 / F.sigmoid(self.conv_invd(inputs) + 1e-6) - 1.0
