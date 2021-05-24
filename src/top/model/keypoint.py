#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import List, Tuple, Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.model.backbone import resnet_fpn_backbone
from top.model.layers import (
    KeypointLayer2D, HeatmapLayer2D, ConvUpsample,
    DisplacementLayer2D, HeatmapCoordinatesLayer)
from top.data.schema import Schema


def _in_place_clipped_sigmoid(x: th.Tensor, eps: float):
    """Clipped in-place sigmoid in range [eps,1-eps].

    # NOTE(ycho): Adopted soft sigmoid from CenterNet Repo
    """
    return th.clamp(x.sigmoid_(), eps, 1.0 - eps)


def _gather_feat(feat: th.Tensor, ind: th.Tensor, mask=None):
    """Gather features:: taken from `CenterNet`.

    `ind` is expected to be formatted with the last dimension
    representing the (i,j)-formatted indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


class KeypointNetwork2D(nn.Module):
    """Simple 2D Keypoint inference network."""

    @dataclass
    class Settings(Serializable):
        backbone_name: str = 'resnet50'
        num_trainable_layers: int = 0
        returned_layers: Tuple[int, ...] = (4,)
        upsample: ConvUpsample.Settings = ConvUpsample.Settings()
        center: HeatmapLayer2D.Settings = HeatmapLayer2D.Settings(
            hidden=(128,))
        keypoint: KeypointLayer2D.Settings = KeypointLayer2D.Settings(
            heatmap=HeatmapLayer2D.Settings(hidden=(128, 128)))
        coord: HeatmapCoordinatesLayer.Settings = HeatmapCoordinatesLayer.Settings()
        # NOTE(ycho): upsample_steps < 5 results in downsampling.
        upsample_steps: Tuple[int, ...] = (128, 64, 16)

        clip_sigmoid_eps: float = 1e-4  # soft sigmoid epsilon

    def __init__(self, opts: Settings):
        super().__init__()
        self.opts = opts

        # NOTE(ycho): Current backbone applies 32x spatial reduction
        self.backbone = resnet_fpn_backbone(
            opts.backbone_name, pretrained=True,
            trainable_layers=opts.num_trainable_layers,
            returned_layers=opts.returned_layers)

        upsample_layers = []
        # NOTE(ycho): hardcoded channel size from
        # known value based on current backbone.
        c_in = 256
        for c_out in self.opts.upsample_steps:
            upsample_layers.append(ConvUpsample(opts.upsample, c_in, c_out))
            c_in = c_out

        self.upsample = nn.Sequential(*upsample_layers)

        self.center = HeatmapLayer2D(opts.center, c_in=c_in)
        self.scale = nn.Conv2d(c_in, 3, 3, 1, 1)
        self.keypoint = KeypointLayer2D(opts.keypoint, c_in=c_in)

        # NOTE(ycho): `coord` is a parameter-free (non-trainable) layer.
        self.coord = HeatmapCoordinatesLayer(opts.coord)

    def forward(self, inputs):
        x = inputs

        # FIXME(ycho): hardcoded feature layer
        x = self.backbone(x)['0']

        # Upsample ...
        x = self.upsample(x)

        # Final outputs ...
        center = self.center(x)
        center = _in_place_clipped_sigmoid(center,
                                           self.opts.clip_sigmoid_eps)
        scale_map = self.scale(x)

        kpt_offset, kpt_heatmap = self.keypoint(x)
        kpt_heatmap = _in_place_clipped_sigmoid(kpt_heatmap,
                                                self.opts.clip_sigmoid_eps)

        # Compute object coordinates, non-max suppression + top-k pruning
        obj_scores, obj_coords = self.coord(center)

        # TODO(ycho): Consider as a transform instead?
        if True:
            # Determine max valid coordinate bounds.
            h, w = scale_map.shape[-2:]
            max_bounds = th.as_tensor([h, w],
                                      dtype=th.int32, device=center_uv.device)

            # Lookup `scale` from `obj_coords`.
            # TODO(ycho): Consider linear interpolation in case
            # `obj_coords` falls to some floating-point precision boundary?
            # For now, assume locally smooth and use the nearest neighbor
            # instead.
            obj_indices = th.round(obj_coord).to(dtype=th.int32)
            flat_index = (obj_indices[..., 0] * w + obj_indices[..., 1])

            # NOTE(ycho): overwrite `invalid` index with 0.
            # Required for `gather()` to NOT crash.
            in_bound = th.all(th.logical_and(obj_indices >= 0,
                                             obj_indices < max_bounds), dim=-1)
            flat_index[~in_bound] = 0

            # Format, permute, etc, etc.
            X = scale_map.reshape(shape[:-2] + (-1,))
            I = flat_index[:, None]
            I = I.expand(*((-1, shape[1]) + tuple(flat_index.shape[1:])))

            # NOTE(ycho): permute required for (B,3,O) -> (B,O,3)
            scale_output = X.gather(-1, I).permute(0, 2, 1)

        output = {
            Schema.HEATMAP: center,
            # NOTE(ycho): Dense heatmap, unlike the labelled counterpart.
            Schema.SCALE_MAP: scale_map,
            Schema.KEYPOINT_OFFSET: kpt_offset,
            Schema.KEYPOINT_HEATMAP: kpt_heatmap,
            Schema.CENTER_2D: obj_coords,
            # NOTE(ycho): Sparse per-object scale heatmap.
            Schema.SCALE = scale_output
        }

        return output


def main():
    model = KeypointNetwork2D(KeypointNetwork2D.Settings())
    dummy = th.empty((1, 3, 128, 128), dtype=th.float32)
    out = model(dummy)
    print(out[Schema.HEATMAP].shape)
    print(out[Schema.SCALE_MAP].shape)  # 1,3,32,32
    print(out[Schema.CENTER_2D].shape)  # 1,9,8,2


if __name__ == '__main__':
    main()
