#!/usr/bin/env python3
"""
Set of transforms related to data augmentation.
"""

__all__ = ['PhotometricAugment']

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple

import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

from top.data.schema import Schema


class PhotometricAugment:
    """
    Apply photometric augmentation.
    """
    @dataclass
    class Settings(Serializable):
        brightness: Tuple[float, float] = (0.4, 1.5)
        contrast: Tuple[float, float] = (0.6, 1.5)
        saturation: Tuple[float, float] = (0.2, 2.0)
        hue: Tuple[float, float] = (-0.3, 0.3)

    def __init__(self, opts: Settings, in_place: bool = True):
        self.opts = opts
        self.in_place = in_place
        self.xfm = transforms.ColorJitter(
            brightness=opts.brightness,
            contrast=opts.contrast,
            saturation=opts.saturation,
            hue=opts.hue)

    def __call__(self, inputs):
        if inputs is None:
            return None
        context, features = inputs
        image = features['image']

        # NOTE(ycho): Shallow copy but pedantically safer
        if not self.in_place:
            features = features.copy()

        augmented_image = self.xfm(features['image'])
        features['image'] = augmented_image
        return (context, features)


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
