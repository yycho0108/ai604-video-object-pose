#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import List, Tuple, Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.model.backbone import resnet_fpn_backbone
from top.model.layers import KeypointLayer2D, ConvUpsample


class KeypointNetwork2D(nn.Module):
    """
    Simple 2D Keypoint inference network.
    """

    @dataclass
    class Settings(Serializable):
        backbone_name: str = 'resnet50'
        num_trainable_layers: int = 0
        keypoint: KeypointLayer2D.Settings = KeypointLayer2D.Settings()
        returned_layers: Tuple[int] = (4,)
        upsample: ConvUpsample.Settings = ConvUpsample.Settings()

    def __init__(self, opts: Settings):
        super().__init__()
        self.opts = opts

        # NOTE(ycho): Current backbone applies 32x spatial reduction
        self.backbone = resnet_fpn_backbone(
            opts.backbone_name, pretrained=True,
            trainable_layers=opts.num_trainable_layers,
            returned_layers=opts.returned_layers)

        # NOTE(ycho): Hardcoded number of channels
        self.upsample = nn.Sequential(
            ConvUpsample(opts.upsample, 256, 128),
            ConvUpsample(opts.upsample, 128, 64),
            ConvUpsample(opts.upsample, 64, 32),
            ConvUpsample(opts.upsample, 32, 16),
            ConvUpsample(opts.upsample, 16, 16),
        )

        self.keypoint = KeypointLayer2D(opts.keypoint,
                                        c_in=16)

    def forward(self, inputs):
        x = inputs

        # FIXME(ycho): hardcoded feature layer
        x = self.backbone(x)['0']

        # Upsample 32x
        x = self.upsample(x)

        # Keypoint.
        x = self.keypoint(x)
        return x


def main():
    model = KeypointNetwork2D(KeypointNetwork2D.Settings())
    dummy = th.empty((1, 3, 128, 128), dtype=th.float32)
    out = model(dummy)
    print(out.shape)


if __name__ == '__main__':
    main()
