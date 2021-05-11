#!/usr/bin/env python3
"""Set of transforms related to keypoints."""

__all__ = ['DenseMapsMobilePose', 'BoxHeatmap']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.data.schema import Schema


class DenseMapsMobilePose:
    """Create dense heatmaps and displacement fields from.

    projected object keypoints - in the style of `MobilePose`.

    `heatmap`: an estimate of the object distribution with a
    gaussian kernel applied near the centroid.

    `displacement_map`: normalized pixel-space displacement.
    Unknown values (e.g. due to kernel size) is set to +inf.
    TODO(ycho): determine if +inf is preferable to other alternatives.
    """
    @dataclass
    class Settings(Serializable):
        # kernel_size: Tuple[int, int] = (5, 5)
        kernel_size: int = 9
        # NOTE(ycho): This `sigma` is defined w.r.t pixel(kernel) units rather
        # than normalized image coordinates, which can be problematic.
        sigma: float = 3.0
        # NOTE: in_place still results in a shallow copy.
        in_place: bool = True
        num_class: int = 9  # bikes, books, etc.
        # [h,w] -> [h//d,w//d]. Note that kernel_size is unaffected.
        downsample: int = 4

        use_displacement: bool = False

    def __init__(self, opts: Settings, device: th.device):
        self.opts = opts
        self.device = device

        # Precompute relevant kernels.
        self.gauss_kernel = self._compute_gaussian_kernel(self.opts,
                                                          self.device)
        self.displacement_kernel = None
        if opts.use_displacement:
            self.displacement_kernel = self._compute_displacement_kernel(
                self.opts, self.device)

    @staticmethod
    def _compute_gaussian_kernel(opts: Settings, device: th.device):
        """Compute a gaussian kernel with the given kernel size and standard
        deviation."""
        delta = th.arange(
            opts.kernel_size,
            device=device) - opts.kernel_size // 2

        gauss_1d = th.exp_(-th.square_(delta) / (2.0 * opts.sigma**2))
        # NOTE(ycho): Don't divide by sum...
        # gauss_1d /= gauss_1d.sum()
        out = gauss_1d[None, :] * gauss_1d[:, None]
        return out

    @staticmethod
    def _compute_displacement_kernel(opts: Settings, device: th.device):
        """Compute a displacement kernel with the given kernel size.

        Note that the output ordering is (major,minor) == (i,j), with
        shape (2,k,k). We generally try to stick with this convention.
        """
        delta = th.arange(
            opts.kernel_size,
            device=device) - opts.kernel_size // 2
        return th.stack(th.meshgrid(delta, delta))

    @staticmethod
    def _compute_adjusted_roi(x: int, y: int, h: int, w: int, r: int):
        """Compute parameters of a sub-window within bounds of the specified
        symmetric rectangle {y+-r,x+-r} that is within range {0-h, 0-w}."""

        # Compute symmetric box extents.
        roi_i00 = y - r
        roi_i01 = y + r + 1
        roi_i10 = x - r
        roi_i11 = x + r + 1

        # Compute intersection of this box with image bounds.
        # ixn_i00 = max(roi_i00, 0)
        # ixn_i01 = min(roi_i01, h)
        # ixn_i10 = max(roi_i10, 0)
        # ixn_i00 = min(roi_i11, w)

        # Compute out-of-bounds offsets.
        # This part effectively computes the "adjustment"
        off_i00 = 0 - min(roi_i00, 0)
        off_i01 = h - max(roi_i01, h)
        off_i10 = 0 - min(roi_i10, 0)
        off_i11 = w - max(roi_i11, w)

        roi = (roi_i00, roi_i01, roi_i10, roi_i11)
        off = (off_i00, off_i01, off_i10, off_i11)
        return (roi, off)

    def __call__(self, inputs: dict):
        # FIXME(ycho): This code is probably incorrect
        # due to shallow copies on tensors.
        opts = self.opts
        if opts.in_place:
            outputs = inputs
        else:
            outputs = inputs.copy()

        # Parse inputs...
        image = inputs[Schema.IMAGE]  # (C,H,W)
        class_index = inputs[Schema.CLASS]
        keypoints_2d_uv = inputs[Schema.KEYPOINT_2D]  # (O, 9, 2|3)

        h, w = image.shape[-2:]
        # NOTE(ycho): number of distinct keypoint classes.
        # Asssumes `keypoints_2d_uv` is formatted as [..., NUM_KPT, DIM_KPT].
        num_vertices = keypoints_2d_uv.shape[-2]

        # NOTE(ycho): Apply downsampling relative to original shape.
        # From this point onwards, we change the value of `h` / `w`.
        h = h // self.opts.downsample
        w = w // self.opts.downsample

        keypoints_2d = (th.as_tensor(keypoints_2d_uv) *
                        th.as_tensor([w, h, 1.0], device=keypoints_2d_uv.device))

        # TODO(ycho): Resolve ambiguous naming convention.
        num_inst = inputs[Schema.INSTANCE_NUM]

        # FIXME(ycho): Because we explicitly refer to
        # a scalar `num_inst`, we cannot process batch inputs after this line.
        n = int(num_inst)

        # Heatmaps for per-object centers...
        shape = list(image.shape[:-3]) + [self.opts.num_class, h, w]
        heatmap = th.zeros(shape, dtype=th.float32, device=self.device)

        # NOTE(ycho): In order for `r` to be fair and symmetric,
        # `opts.kernel_size` has to be an odd integer.
        k = self.opts.kernel_size
        r = self.opts.kernel_size // 2

        # Process each object instance in the image.
        # TODO(ycho): This somehow feels very inefficient.
        for i_obj in range(n):
            # NOTE(ycho): hardcoded assumption that
            # in the dataset, keypoints[...,0,:] = center
            # Potentially replace with Box.CENTROID?
            i_cls = class_index[i_obj]
            center = keypoints_2d[i_obj, 0, :2]  # (..., 2)
            cx, cy = int(center[0]), int(center[1])

            # Update heatmap...
            i_roi, i_off = self._compute_adjusted_roi(cx, cy, h, w, r)

            if np.all(np.abs(i_off) <= r):
                (box_i00, box_i01, box_i10, box_i11) = i_roi
                (off_i00, off_i01, off_i10, off_i11) = i_off

                # Update applicable region from relevant parts of the kernel.
                # TODO(ycho): Consider if classwise vs. unified heatmap is the
                # right way to go.
                roi = heatmap[..., i_cls, box_i00 + off_i00:box_i01 + off_i01,
                              box_i10 + off_i10:box_i11 + off_i11]
                ker = self.gauss_kernel[
                    off_i00: k + off_i01,
                    off_i10: k + off_i11]
                roi[...] = roi.maximum(ker)
            outputs[Schema.HEATMAP] = heatmap

        if self.opts.use_displacement:
            # NOTE(ycho): 2X to account for offsets in {+i,+j} directions,
            shape = list(image.shape[:-3]) + [2 * num_vertices, h, w]

            # FIXME(ycho): OK-ish to initialize as ones since normalized?
            displacement_map = th.full(
                shape,
                fill_value=float('inf'),
                dtype=th.float32, device=self.device)

            # Compute on-the-fly normalized displacement/distance kernels.
            # TODO(ycho): Technically, not the efficient way to achieve this.
            image_scale = th.as_tensor([h, w], device=self.device)[
                :, None, None]
            normalized_displacement = self.displacement_kernel / image_scale
            distance = normalized_displacement.square().sum(dim=0)

            for i_obj in range(n):
                # Process vertices ...
                vertices = keypoints_2d[i_obj, :, :2]
                for i, (x, y) in enumerate(vertices):
                    i0, i1 = int(y), int(x)
                    i_roi, i_off = self._compute_adjusted_roi(i1, i0, h, w, r)

                    if np.any(np.abs(i_off) > r):
                        continue
                    (box_i00, box_i01, box_i10, box_i11) = i_roi
                    (off_i00, off_i01, off_i10, off_i11) = i_off

                    # Update displacement map (cwise min, closest)
                    roi = displacement_map[..., i * 2: i * 2 + 2,
                                           box_i00 + off_i00:box_i01 + off_i01,
                                           box_i10 + off_i10:box_i11 + off_i11]
                    ker = normalized_displacement[...,
                                                  off_i00: k + off_i01,
                                                  off_i10: k + off_i11]

                    # Since we need to update both coordinates simultaneously,
                    # select the relevant region based on mask and perform explicit
                    # update.
                    msk = (
                        roi.square().sum(dim=-3) >
                        distance[off_i00: k + off_i01, off_i10: k + off_i11])
                    roi[..., msk] = ker[..., msk]
            outputs[Schema.DISPLACEMENT_MAP] = displacement_map

        shape = list(image.shape[:-3]) + [num_vertices, h, w]
        kpt_heatmap = th.zeros(
            shape,
            dtype=th.float32,
            device=self.device)

        for i_obj in range(n):
            # Process vertices ...
            # TODO(ycho): either `r`, `sigma` or both should be
            # inversely proportional to `depth`,
            # where depth ~ inputs[Schema.TRANSLATION].norm() ...
            vertices = keypoints_2d[i_obj, :, :2]
            for i_kpt, (x, y) in enumerate(vertices):
                i0, i1 = int(y), int(x)
                i_roi, i_off = self._compute_adjusted_roi(i1, i0, h, w, r)

                if np.any(np.abs(i_off) > r):
                    continue
                (box_i00, box_i01, box_i10, box_i11) = i_roi
                (off_i00, off_i01, off_i10, off_i11) = i_off

                # Update displacement map (cwise min, closest)
                roi = kpt_heatmap[..., i_kpt,
                                  box_i00 + off_i00:box_i01 + off_i01,
                                  box_i10 + off_i10:box_i11 + off_i11]
                ker = self.gauss_kernel[...,
                                        off_i00: k + off_i01,
                                        off_i10: k + off_i11]
                roi[...] = roi.maximum(ker)

            outputs[Schema.KEYPOINT_HEATMAP] = kpt_heatmap

        return outputs


class BoxHeatmap:
    """Legacy implementation for heatmap computation from object bounding box.

    NOTE(ycho): Currently left here for archival purposes for referring to the
    explicit stages of the transform that the object parameter
    undergoes in order for the resulting projection to be valid.
    """

    def __init__(self, device: th.device):
        self.device = device
        vertices = list(itertools.product(
            *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))  # 8x3
        # NOTE(ycho): prepending centroid.
        vertices.insert(0, [0.0, 0.0, 0.0])
        vertices = th.as_tensor(vertices, dtype=th.float32, device=self.device)
        colors = (0.5 + 0.5 * vertices)
        self.vertices = vertices
        self.colors = colors

    def __call__(self, inputs):
        # Parse inputs.
        h, w = inputs[Schema.IMAGE].shape[-2:]
        R = th.as_tensor(inputs[Schema.ORIENTATION])
        T = th.as_tensor(inputs[Schema.TRANSLATION])
        S = th.as_tensor(inputs[Schema.SCALE])
        P = th.as_tensor(inputs[Schema.PROJECTION])
        num_inst = inputs[Schema.INSTANCE_NUM]

        heatmaps = []
        n = int(num_inst)
        for i in range(n):
            # NOTE(ycho): Taking an instance-specific slice of batched input.
            irxn = R[i * 9:(i + 1) * 9]
            itxn = T[i * 3:(i + 1) * 3]
            iscale = S[i * 3:(i + 1) * 3]

            T_scale = th.eye(4)
            T_scale[(0, 1, 2), (0, 1, 2)] = iscale.reshape(3)

            # BBOX3D transform
            T_box = th.eye(4)
            T_box[:3, :3] = irxn.reshape(3, 3)
            T_box[:3, -1] = itxn

            # Camera transforms
            T_p = P.reshape(4, 4).float()

            # Compose all transforms
            # NOTE(ycho): Looks like `camera/view` is not needed.
            # Perhaps it's been fused into
            # object/{translation,orientation}.
            T = T_p @ T_box @ T_scale

            # Apply transform.
            # NOTE(ycho): skipping division on the last axis here.
            v = self.vertices @ T[:3, :3].T + T[:3, -1]
            v[..., :-1] /= v[..., -1:]

            # NDC -> Screen coordinates
            v[..., 0] = (1 + v[..., 0]) * (0.5 * h)
            v[..., 1] = (1 + v[..., 1]) * (0.5 * w)
            y, x = v[..., 0], v[..., 1]

            # TODO(ycho): avoid unnecessary use of cv2 here
            heatmap = th.zeros_like(inputs[Schema.IMAGE])
            for j, (px, py) in enumerate(zip(x, y)):
                if px < 0 or px >= w or py < 0 or py >= h:
                    continue
                # cv2.circle(heatmap, (int(px), int(py)), 16, (0, 0, 255), -1)
                heatmap[...,
                        int(py) - 1:int(py) + 1,
                        int(px) - 1:int(px) + 1] = 255.0 * self.colors[j][:,
                                                                          None,
                                                                          None]  # 255
            heatmaps.append(heatmap)

        heatmaps = th.stack(heatmaps, dim=0)  # CHW, where C == num instances
        inputs[Schema.KEYPOINT_MAP] = heatmaps
        return inputs
