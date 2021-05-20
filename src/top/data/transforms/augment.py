#!/usr/bin/env python3
"""Set of transforms related to data augmentation."""

__all__ = ['PhotometricAugment']

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple, Dict, Hashable

import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

from top.data.schema import Schema


class PhotometricAugment:
    """Apply photometric augmentation."""
    @dataclass
    class Settings(Serializable):
        brightness: Tuple[float, float] = (0.6, 1.6)
        contrast: Tuple[float, float] = (0.6, 1.6)
        saturation: Tuple[float, float] = (0.6, 1.6)
        hue: Tuple[float, float] = (-0.2, 0.2)
        key_in: Schema = Schema.IMAGE
        key_out: Schema = Schema.IMAGE

    def __init__(self, opts: Settings, in_place: bool = True):
        self.opts = opts
        self.in_place = in_place
        self.xfm = transforms.ColorJitter(
            brightness=opts.brightness,
            contrast=opts.contrast,
            saturation=opts.saturation,
            hue=opts.hue)

    def __call__(self, inputs: Dict[Hashable, th.Tensor]):
        if inputs is None:
            return None

        # NOTE(ycho): Shallow copy but pedantically safer
        if self.in_place:
            outputs = inputs
        else:
            outputs = inputs.copy()

        augmented_image = self.xfm(outputs[self.opts.key_in])
        outputs[self.opts.key_out] = augmented_image
        return inputs


def main():
    from top.data.objectron_sequence import (
        ObjectronSequence,
        SampleObjectron,
        DecodeImage,
        ParseFixedLength,
        _skip_none)

    opts = SampleObjectron.Settings()
    xfm = transforms.Compose([
        DecodeImage(size=(480, 640)),
        ParseFixedLength(ParseFixedLength.Settings(seq_len=4)),
        PhotometricAugment(PhotometricAugment.Settings())
    ])
    dataset = SampleObjectron(opts, xfm)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=_skip_none)

    for data in loader:
        context, features = data
        print(features['image'].shape)
        break


if __name__ == '__main__':
    main()
