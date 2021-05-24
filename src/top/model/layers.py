#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple, Dict, Hashable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix.dsnt import (
    spatial_expectation2d, spatial_softmax2d
)

from top.data.schema import Schema


class ConvUpsample(nn.Module):
    """2x Upsampling block via transposed convolution.

    Partly adopted from : "https://github.com/xingyizhou/CenterNet/blob/
    master/src/lib/models/networks/msra_resnet.py"
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
    """NOTE(ycho): Not sure what to actually do, so it's just a placeholder for
    now."""

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


class DisplacementLayer2D(nn.Module):
    @dataclass
    class Settings(Serializable):
        hidden: Tuple[int, ...] = ()
        # NOTE(ycho): 9 vertices by default, including centroid.
        num_keypoints: int = 9

    def __init__(self, opts: Settings, c_in: int):
        super().__init__()

        channels = [c_in]
        for h in opts.hidden:
            channels.append(h)

        # Add intermediate convolutional layers.
        layers = []
        for prv, nxt in zip(channels[:-1], channels[1:]):
            conv = nn.Conv2d(prv, nxt, kernel_size=3, padding=1, bias=True)
            # TODO(ycho): Determine if layers such as batchnorm
            # would be necessary here.
            relu = nn.ReLU(inplace=True)
            layers.extend([conv, relu])

        # Add final unbounded output.
        out_channels = opts.num_keypoints * 2
        layers.append(
            nn.Conv2d(
                channels[-1],
                out_channels, kernel_size=1, stride=1, padding=0))

        self.output = nn.Sequential(*layers)

    def forward(self, inputs: th.Tensor):
        return self.output(inputs)


class HeatmapLayer2D(nn.Module):
    """Layer for computing dense heatmaps.

    Output heatmaps have the same spatial dimension as the input.
    NOTE(ycho): Beware that this class produces logits on which further
    mapping such as the activation through the sigmoid function has to
    be applied to convert to a quantity interpretable as probability.
    """

    @dataclass
    class Settings(Serializable):
        hidden: Tuple[int, ...] = ()
        num_class: int = 9

    def __init__(self, opts: Settings, c_in: int):
        super().__init__()

        channels = [c_in]
        for h in opts.hidden:
            channels.append(h)

        # Add intermediate convolutional layers.
        layers = []
        for prv, nxt in zip(channels[:-1], channels[1:]):
            conv = nn.Conv2d(prv, nxt, kernel_size=3, padding=1, bias=True)
            relu = nn.ReLU(inplace=True)
            layers.extend([conv, relu])

        # Add final unbounded output which produces logits.
        layers.append(
            nn.Conv2d(
                channels[-1],
                opts.num_class, kernel_size=1, stride=1, padding=0))

        self.output = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.output(inputs)


class KeypointLayer2D(nn.Module):
    """CenterNet-style keypoint prediction in 2D.

    We treat keypoint locations as offsets from the center such that
    the keypoint offsets are predicted in a "multi-head" fashion from the center.
    See Section 4.2 from [1].

    [1] Zhou, Xingyi et al. “Objects as Points.” ArXiv abs/1904.07850 (2019): n. pag.
    """

    @dataclass
    class Settings(Serializable):
        kernel_size: int = 3
        num_keypoints: int = 9
        heatmap: HeatmapLayer2D.Settings = HeatmapLayer2D.Settings()

    def __init__(self, opts: Settings, c_in: int):
        super().__init__()
        self.opts = opts

        # TODO(ycho): Fancier kpt
        self.offset = nn.Conv2d(c_in,
                                opts.num_keypoints * 2,
                                opts.kernel_size,
                                padding=opts.kernel_size // 2
                                )

        self.heatmap = HeatmapLayer2D(
            opts.heatmap, c_in=c_in)

    def forward(self, inputs: th.Tensor):
        """

        input: th.Tensor feature map.

        Returns:
            offset  : [N,Kx2,H,W] Dense keypoint offsets from center
            heatmap : [N,K,H,W] Additional keypoint heatmaps for refinement.
        """
        offset = self.offset(inputs)
        heatmap = self.heatmap(inputs)
        return (offset, heatmap)


class KeypointLayer3D(nn.Module):
    """CenterNet-style 3D keypoint regressor.

    TODO(ycho): Finish implementation.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.conv_hm = nn.Conv2d(c_in, 1, 3, padding=1)
        self.conv_invd = nn.Conv2d(c_in, 1, 3, padding=1)

    def forward(self, inputs):
        # heatmap ...
        hm = th.sigmoid(self.conv_hm(inputs))
        # depth ...
        depth = 1.0 / th.sigmoid(self.conv_invd(inputs) + 1e-6) - 1.0


class DsntLayer2D(nn.Module):
    """Dense Heatmap -> sparse keypoint locations.

    @see kornia.geometry.subpix.dsnt.spatial_expectation2d
    """

    @dataclass
    class Settings(Serializable):
        temperature: float = 1.0

    def __init__(self, opts: Settings):
        self.opts = opts
        self.temperature = th.as_tensor(self.opts.temperature)
        super().__init__()

    def forward(self, inputs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """NCHW -> NC2."""
        # Convert logits to probabilities.
        # TODO(ycho): Consider if this is preferable to elementwise sigmoid.
        prob = spatial_softmax2d(inputs, temperature=self.temperature)
        kpts = spatial_expectation2d(prob, normalized_coordinates=True)
        return (prob, kpts)


def spatial_nms(x, kernel_size: int):
    # Alternatively, kernel_size ~ 3 * sigma
    pad = (kernel_size - 1) // 2
    local_max = th.nn.functional.max_pool2d(
        x, (kernel_size, kernel_size), stride=1, padding=pad)
    peak_mask = (local_max == x).float()
    return x * peak_mask


def top_k(scores, k: int):
    batch_size, types, h, w = scores.size()
    scores, indices = th.topk(scores.view(batch_size, types, -1), k)
    indices = indices % (h * w)
    i = (indices / w)
    j = (indices % w)
    return (scores, indices, i, j)


class HeatmapCoordinatesLayer:
    """Convert dense heatmap to peak coordinates.

    NOTE(ycho): Currently, the returned coordinates are evaluated as an
    un-normalized floating-point grid over the image shape, ordered in (i-j)
    order rather than (x-y) order.
    """

    @dataclass
    class Settings(Serializable):
        nms_kernel_size: int = 9
        max_num_instance: int = 8

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs: th.Tensor):
        heatmap = inputs
        batch, cat, height, width = heatmap.shape
        num_points = heatmap.shape[1]
        heatmap = spatial_nms(heatmap, self.opts.nms_kernel_size)
        scores, inds, i, j = top_k(heatmap, k=self.opts.max_num_instance)
        return scores, th.stack([i, j], axis=-1)
