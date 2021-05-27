#!/usr/bin/env python3

import torch as th
from torchvision.transforms import Compose, Resize

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema
from top.data.transforms import PhotometricAugment, InstancePadding

from matplotlib import pyplot as plt


def _stack_images(x: th.Tensor):
    # BCHW -> CHBW
    x = x.permute((1, 2, 0, 3))
    # CHBW -> CH(BxW)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def _to_image(x: th.Tensor):
    return x.permute((1, 2, 0)).detach().cpu().numpy()


def main():
    opts = DatasetSettings()
    opts = update_settings(opts)
    transform = Compose([
        InstancePadding(InstancePadding.Settings()),
        # PhotometricAugment(PhotometricAugment.Settings(key_out=key_out))
    ])
    train_loader, test_loader = get_loaders(opts,
                                            device=th.device('cpu'),
                                            batch_size=4,
                                            transform=transform)

    fig, ax = plt.subplots(1, 1)
    for data in train_loader:
        image = _stack_images(data[Schema.IMAGE])
        image = _to_image(image)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        k = plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
