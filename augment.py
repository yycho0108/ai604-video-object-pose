#!/usr/bin/env python3

from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

from objectron_dataset import (
    Objectron,
    SampleObjectron,
    DecodeImage,
    ParseFixedLength,
    _skip_none)


class PhotometricAugment:
    """
    Apply photometric augmentation.
    """
    @dataclass
    class Settings:
        brightness: Tuple[float, float] = (0.2, 2.0)
        contrast: Tuple[float, float] = (0.3, 2.0)
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

        features['image'] = self.xfm(features['image'])

        return (context, features)


def main():
    opts = SampleObjectron.Settings()
    xfm = transforms.Compose([
        DecodeImage(size=(480, 640)),
        ParseFixedLength(ParseFixedLength.Settings()),
        PhotometricAugment(PhotometricAugment.Settings())
    ])
    dataset = SampleObjectron(opts, xfm)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=_skip_none)

    for data in loader:
        # print(data)
        break


if __name__ == '__main__':
    main()
