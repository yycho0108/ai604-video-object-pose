#!/usr/bin/env python3

import torch as th

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema
from top.data.transforms import PhotometricAugment

from matplotlib import pyplot as plt


def main():
    opts = DatasetSettings()
    opts = update_settings(opts)
    transform = PhotometricAugment(
        PhotometricAugment.Settings(
            key_out='aug_img'))
    train_loader, test_loader = get_loaders(opts,
                                            device=th.device('cpu'),
                                            batch_size=1,
                                            transform=transform)

    fig, ax = plt.subplots(2, 1)
    for data in train_loader:
        image = data[Schema.IMAGE][0].permute((1, 2, 0))  # 13HW
        aug_image = data['aug_img'][0].permute((1, 2, 0))  # 13HW
        ax[0].imshow(image.cpu().numpy())
        ax[1].imshow(aug_image.cpu().numpy())
        k = plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
