#!/usr/bin/env python3

import torch as th
from torchvision.transforms import Compose, Resize

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders, collate_cropped_img)
from top.data.schema import Schema
from top.data.transforms import PhotometricAugment, InstancePadding
from top.data.bbox_reg_util import CropObject

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
        # NOTE(ycho): `FormatLabel` must be applied prior
        # to `CropObject` since it modifies the requisite tensors.
        # FormatLabel(FormatLabel.Settings(), opts.vis_thresh),
        CropObject(CropObject.Settings()),
        # Normalize(Normalize.Settings(keys=(Schema.CROPPED_IMAGE,)))
    ])
    _, test_loader = get_loaders(opts,
                                 device=th.device('cpu'),
                                 batch_size=4,
                                 transform=transform,
                                 collate_fn=collate_cropped_img)

    fig, ax = plt.subplots(1, 1)
    for data in test_loader:
        # -> 4,1,3,640,480
        # -- (batch_size, ? ,3, 640, 480)
        print(data[Schema.IMAGE].shape)
        print(data[Schema.CROPPED_IMAGE].shape)
        print(data[Schema.INDEX])
        print(data[Schema.INSTANCE_NUM])
        print('points_2d')
        print(data[Schema.KEYPOINT_2D])
        print('points_3d')
        print(data[Schema.KEYPOINT_3D])

        # image = _stack_images(data[Schema.IMAGE])
        image = _stack_images(data[Schema.CROPPED_IMAGE])
        image = _to_image(image)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        k = plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
