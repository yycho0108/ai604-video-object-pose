#!/usr/bin/env python3

import torch as th
from torchvision.transforms import Compose, Resize

from top.run.app_util import update_settings
from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema
from top.data.transforms import (PhotometricAugment, GeometricAugment)
from top.data.transforms import DrawKeypoints, DrawKeypointMap, DenseMapsMobilePose, DrawBoundingBoxFromKeypoints

from matplotlib import pyplot as plt


def _to_image(x: th.Tensor):
    return x.permute((1, 2, 0)).detach().cpu().numpy()


def main():
    opts = DatasetSettings()
    opts = update_settings(opts)
    augment = GeometricAugment(
        GeometricAugment.Settings(
            p_flip_lr=1.0), in_place=False)
    key_out = '__keypoint_image__'  # try not to result in key collision

    visualize = Compose([
        DenseMapsMobilePose(
            DenseMapsMobilePose.Settings(),
            device=th.device('cpu')),
        DrawKeypointMap(
            DrawKeypointMap.Settings(
                key_in=Schema.KEYPOINT_HEATMAP,
                key_out=key_out,
                as_displacement=False)),
        DrawBoundingBoxFromKeypoints(DrawBoundingBoxFromKeypoints.Settings(
            key_in=key_out,
            key_out=key_out
        ))
    ])

    # visualize = DrawKeypoints(DrawKeypoints.Settings(key_out=key_out))
    #visualize = DrawKeypointMap(DrawKeypointMap.Settings(key_in=Schema.IMAGE, key_out=key_out,
    #    as_displacement = False))
    train_loader, _ = get_loaders(opts,
                                  device=th.device('cpu'),
                                  batch_size=None,
                                  transform=None)

    fig, ax = plt.subplots(2, 1)
    for data in train_loader:
        resize = Resize(size=data[Schema.IMAGE].shape[-2:])

        aug_data = augment(data)
        # aug_data = data
        print(data[Schema.KEYPOINT_2D])
        print(aug_data[Schema.KEYPOINT_2D])
        v0 = resize(visualize(data)[key_out])
        v1 = resize(visualize(aug_data)[key_out])
        # v0 = 255 * resize(v0) + data[Schema.IMAGE]
        # v1 = 255 * resize(v1) + aug_data[Schema.IMAGE]
        v0 = th.where(v0 <= 0, data[Schema.IMAGE].float(), 255.0 * v0)
        v1 = th.where(v1 <= 0, aug_data[Schema.IMAGE].float(), 255.0 * v1)

        v0 = _to_image(v0) / 255.0
        v1 = _to_image(v1) / 255.0
        #image = data[Schema.IMAGE][0].permute((1, 2, 0))  # 13HW
        #aug_image = data['aug_img'][0].permute((1, 2, 0))  # 13HW
        ax[0].imshow(v0)
        ax[0].set_title('orig')
        ax[1].imshow(v1)
        ax[1].set_title('aug')
        k = plt.waitforbuttonpress()
        # break


if __name__ == '__main__':
    main()
