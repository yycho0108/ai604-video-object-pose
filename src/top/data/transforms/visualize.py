#!/usr/bin/env python3
"""
Set of transforms related to visualization.
"""

__all__ = ['DrawKeypoints', 'DrawDisplacementMap']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.data.schema import Schema


class DrawKeypoints:
    """
    Draw keypoints (as inputs['points']) on an image as-is.
    Mostly intended for debugging.
    """

    @dataclass
    class Settings(Serializable):
        kernel_size: int = 5
        # NOTE(ycho): configurable input, in case of augmented or cropped
        # image.
        key_in: Schema = Schema.IMAGE
        key_out: str = 'rendered_keypoints'  # No Schema assigned for now

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs):
        outputs = inputs
        image = inputs[self.opts.key_in]

        # Points in UV-coordinates
        # consistent with the objectron format.
        h, w = image.shape[-2:]
        points_uv = th.as_tensor(inputs[Schema.KEYPOINT_2D])

        # NOTE(ycho): The Objectron dataset flipped their convention
        # so that the point is ordered in a minor-major axis order.
        points = points_uv * th.as_tensor([w, h, 1.0])
        out = th.zeros_like(inputs[self.opts.key_in])
        num_inst = inputs[Schema.INSTANCE_NUM]
        n = int(num_inst)

        r = self.opts.kernel_size // 2
        for i in range(n):
            for x, y, _ in points[i]:
                i0, i1 = int(y), int(x)
                out[..., i0 - r: i0 + r, i1 - r:i1 + r] = 255
        outputs[self.opts.key_out] = out
        return outputs


class DrawDisplacementMap:

    @dataclass
    class Settings(Serializable):
        sigma: float = 0.1
        key_in: str = ''
        key_out: str = ''

    def __init__(self, opts: Settings):
        self.opts = opts
        vertices = list(itertools.product(
            *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
        vertices = np.insert(vertices, 0, [0, 0, 0], axis=0)
        vertices = th.as_tensor(vertices, dtype=th.float32)
        vertices = vertices
        # Map vertices to colors. =RGB(0.25~0.75)
        colors = (0.5 + 0.5 * vertices)
        self.colors = colors  # 9x3

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        """
        Provide visualization of displacement map ...
        """

        # Parse inputs.
        if self.opts.key_in and isinstance(inputs, dict):
            displacement_map = inputs[self.opts.key_in]
        else:
            displacement_map = inputs

        # NOTE(ycho): convert displacement maps to weight,
        # where low displacement == high visual weight
        displacement_map = displacement_map.view(-1,
                                                 displacement_map.shape[-3] // 2,
                                                 2,
                                                 displacement_map.shape[-2],
                                                 displacement_map.shape[-1])
        distance_map = th.norm(displacement_map, dim=-3)
        mask = th.isfinite(distance_map)
        denom = (2 * self.opts.sigma * self.opts.sigma)
        vis_weights = th.where(
            mask, th.exp(-distance_map / denom),
            th.as_tensor(0.0, device=mask.get_device())
        )
        vis = th.einsum(
            '...khw, kc -> ...chw',
            vis_weights,
            self.colors.to(
                vis_weights.get_device()))

        # Format outputs.
        if self.opts.key_out and isinstance(inputs, dict):
            outputs = inputs.copy()
            outputs[self.opts.key_out] = vis
        else:
            outputs = vis
        return outputs
