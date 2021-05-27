#!/usr/bin/env python3
"""Set of transforms related to visualization."""

__all__ = ['DrawKeypoints', 'DrawKeypointMap', 'DrawBoundingBoxFromKeypoints']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable

import numpy as np
import cv2
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.data.schema import Schema


class DrawKeypoints:
    """Draw keypoints (as inputs['points']) on an image as-is.

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


class DrawBoundingBoxFromKeypoints:
    """Draw keypoints (as inputs['points']) on an image as-is.

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

        # Clone image to avoid overwriting and bring it to cpu+numpy.
        image = image.permute(1, 2, 0).clone()
        if isinstance(image, th.Tensor):
            image = image.detach().cpu().numpy()

        # NOTE(ycho): Ensure contiguity.
        image = np.ascontiguousarray(image)

        def _as_point(x):
            """make point array compatible with cv2 requirements."""
            return (int(x[0]), int(x[1]))

        # Draw 3d bounding box per object.
        for i in range(n):
            box = points[i]
            for j, (src, dst) in enumerate([
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    (0, 2), (4, 6), (1, 3), (5, 7),
                    (0, 1), (2, 3), (4, 5), (6, 7)]):
                color = (1.0, 0.0, 0.0) if j < 4 else (0.0, 0.0, 1.0)
                image = cv2.line(
                    image, _as_point(
                        box[src + 1]), _as_point(
                        box[dst + 1]), color, 1)
        image = th.as_tensor(image.transpose(2, 0, 1))

        outputs[self.opts.key_out] = image
        return outputs


class DrawKeypointMap:
    """Render a colorized version of the keypoint heatmap."""

    @dataclass
    class Settings(Serializable):
        sigma: float = 0.1
        key_in: str = ''
        key_out: str = ''
        as_displacement: bool = True

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
        """Provide visualization of keypoint map ..."""

        # Parse inputs.
        if self.opts.key_in and isinstance(inputs, dict):
            kpt_map = inputs[self.opts.key_in]
        else:
            kpt_map = inputs

        if self.opts.as_displacement:
            displacement_map = kpt_map

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

            # NOTE(ycho) exponentiated distance ~ heatmap
            kpt_heatmap = th.where(
                mask, th.exp(-distance_map / denom),
                th.as_tensor(0.0, device=mask.device)
            )
        else:
            kpt_heatmap = kpt_map

        # NOTE(ycho): Produce per-keypoint colored heatmap
        # into a single (slightly ambiguous) visualization.
        # print(kpt_heatmap.shape)  # (1,64,64...??)
        # print(self.colors.shape)
        colored_heatmap = th.einsum(
            '...khw, kc -> ...chw',
            kpt_heatmap,
            self.colors.to(
                kpt_heatmap.device))

        # Format outputs.
        if self.opts.key_out and isinstance(inputs, dict):
            outputs = inputs.copy()
            outputs[self.opts.key_out] = colored_heatmap
        else:
            outputs = colored_heatmap
        return outputs
