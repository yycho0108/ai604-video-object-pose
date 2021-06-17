#!/usr/bin/env python3

import torch as th

from typing import Hashable
from torchvision.transforms import Compose, Resize, ToTensor

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema
from top.data.transforms import (
    PhotometricAugment,
    InstancePadding,
    Normalize)

from matplotlib import pyplot as plt


def _stack_images(x: th.Tensor):
    # BCHW -> CHBW
    x = x.permute((1, 2, 0, 3))
    # CHBW -> CH(BxW)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def _to_image(x: th.Tensor):
    x = (x * 0.25) + 0.5
    return x.permute((1, 2, 0)).detach().cpu().numpy()

class ToTensorWithDict:
    def __init__(self, key:Hashable):
        self.key=key
        self.xfm = ToTensor()
    def __call__(self, x):
        y=x.copy()
        y[self.key]=self.xfm(y[self.key])
        return y

def main():
    opts = DatasetSettings()
    opts = update_settings(opts)
    key_out = '__aug_img__'  # Try to prevent key collision
    transform = Compose([
        InstancePadding(InstancePadding.Settings()),
        # ToTensorWithDict(key
        PhotometricAugment(PhotometricAugment.Settings(key_out=key_out)),
        Normalize(Normalize.Settings(keys=(Schema.IMAGE, key_out,))),
    ])
    train_loader, test_loader = get_loaders(opts,
                                            device=th.device('cpu'),
                                            batch_size=4,
                                            transform=transform)

    fig, ax = plt.subplots(2, 1)
    for data in train_loader:
        image = _stack_images(data[Schema.IMAGE])
        aug_image = _stack_images(data[key_out])
        print(image.min(), image.max())
        print(aug_image.min(), aug_image.max())
        image = _to_image(image)
        aug_image = _to_image(aug_image)
        print(image.min(), image.max())
        print(aug_image.min(), aug_image.max())
        ax[0].imshow(image)
        ax[1].imshow(aug_image)
        k = plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
