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
    DisplacementLayer2D)
from top.data.schema import Schema


def _in_place_clipped_sigmoid(x: th.Tensor, eps: float):
    """Clipped in-place sigmoid in range [eps,1-eps].

    # NOTE(ycho): Adopted soft sigmoid from CenterNet Repo
    """
    return th.clamp(x.sigmoid_(), eps, 1.0 - eps)

    # return x.sigmoid_().clip_(eps, 1.0 - eps)


class KeypointNetwork2D(nn.Module):
    """Simple 2D Keypoint inference network."""

    @dataclass
    class Settings(Serializable):
        backbone_name: str = 'resnet50'
        num_trainable_layers: int = 0
        returned_layers: Tuple[int, ...] = (4,)
        upsample: ConvUpsample.Settings = ConvUpsample.Settings()
        # displacement: DisplacementLayer2D.Settings = DisplacementLayer2D.Settings()
        center: HeatmapLayer2D.Settings = HeatmapLayer2D.Settings()
        keypoint: KeypointLayer2D.Settings = KeypointLayer2D.Settings()
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
        self.keypoint = KeypointLayer2D(opts.keypoint, c_in=c_in)
        # self.displacement = DisplacementLayer2D(opts.displacement, c_in=c_in)

    def forward(self, inputs):
        x = inputs

        # FIXME(ycho): hardcoded feature layer
        x = self.backbone(x)['0']

        # Upsample ...
        x = self.upsample(x)

        # Final outputs ...
        center_logits = self.center(x)

        center = _in_place_clipped_sigmoid(center_logits,
                                           self.opts.clip_sigmoid_eps)

        # displacement_map = self.displacement(x)
        kpt_offset, kpt_heatmap = self.keypoint(x)

        kpt_heatmap = _in_place_clipped_sigmoid(kpt_heatmap,
                                                self.opts.clip_sigmoid_eps)

        output = {
            Schema.HEATMAP: center,
            # Schema.HEATMAP_LOGITS: heatmap_logits,
            # Schema.DISPLACEMENT_MAP: displacement_map
            Schema.KEYPOINT_OFFSET: kpt_offset,
            Schema.KEYPOINT_HEATMAP: kpt_heatmap
        }
        return output


def main():
    model = KeypointNetwork2D(KeypointNetwork2D.Settings())
    dummy = th.empty((1, 3, 128, 128), dtype=th.float32)
    out = model(dummy)
    print(out[Schema.HEATMAP].shape)


if __name__ == '__main__':
    main()
