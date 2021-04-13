#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.nn.functional as F


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


class KeypointRegressor2D(nn.Module):
    """
    Placeholder for keypoint regressor in 2D.
    Probably works-ish, but should be replaced with a more principled alternative.
    """
    NUM_KPTS = 8 + 1

    def __init__(self, c_in: int):
        self.conv_kpt_cls = nn.Conv2d(c_in, NUM_KPTS, 3)
        self.conv_kpt_pos = nn.Conv2d(c_in, 2, 3)

    def forward(self, inputs):
        x = inputs
        c_logit = self.conv_kpt_cls(x)

        # bounding box scale
        p = self.conv_kpt_pos(x)

        # x = F.softmax(x, dim=1)
        return (c_logit, p)


class KeypointRegressor3D(nn.Module):
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
