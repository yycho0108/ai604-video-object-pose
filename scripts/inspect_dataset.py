#!/usr/bin/env python3

import torch as th

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema

from matplotlib import pyplot as plt


def main():
    opts = DatasetSettings()
    opts = update_settings(opts)
    train_loader, test_loader = get_loaders(opts,
                                            device=th.device('cpu'),
                                            batch_size=1,
                                            transform=None)

    for data in train_loader:
        image = data[Schema.IMAGE][0].permute((1, 2, 0))  # 13HW
        plt.imshow(image.cpu().numpy())
        plt.show()


if __name__ == '__main__':
    main()
