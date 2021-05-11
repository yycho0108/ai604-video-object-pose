#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th
from torchvision.transforms import Compose
from torchvision.utils import save_image

from top.data.load import (DatasetSettings, get_loaders)
from top.data.objectron_detection import ObjectronDetection, SampleObjectron
from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.transforms import (
    BoxHeatmap,
    DrawKeypoints,
    DrawKeypointMap,
    DenseMapsMobilePose,
    InstancePadding,
)
from top.data.schema import Schema

from top.run.app_util import update_settings
from top.run.torch_util import resolve_device


@dataclass
class AppSettings(Serializable):
    dataset: DatasetSettings = DatasetSettings()


def main():
    opts = AppSettings()
    opts = update_settings(opts)

    device = resolve_device('cpu')

    xfm = Compose([BoxHeatmap(device=device),
                   DrawKeypoints(DrawKeypoints.Settings()),
                   DenseMapsMobilePose(DenseMapsMobilePose.Settings(), device),
                   InstancePadding(InstancePadding.Settings()),
                   #DrawDisplacementMap(DrawDisplacementMap.Settings(
                   #    key_in=Schema.DISPLACEMENT_MAP,
                   #    key_out='dmap_vis'
                   #))
                   ])

    dataset, _ = get_loaders(opts.dataset, device, None, xfm)

    for data in dataset:
        save_image(data[Schema.IMAGE] / 255.0, F'/tmp/img.png')
        # save_image(data['rendered_keypoints'] / 255.0, F'/tmp/rkpts.png')

        for i, img in enumerate(data[Schema.KEYPOINT_MAP]):
            save_image(img / 255.0, F'/tmp/kpt-{i}.png')

        for i, img in enumerate(data[Schema.HEATMAP]):
            save_image(img, F'/tmp/heatmap-{i}.png',
                       normalize=True)

        #for i, img in enumerate(data[Schema.DISPLACEMENT_MAP]):
        #    save_image(
        #        th.where(th.isfinite(img), img.abs(), th.as_tensor(0.0)),
        #        F'/tmp/displacement-{i}.png',
        #        normalize=True)
        # save_image(data['dmap_vis'], '/tmp/dmap-vis.png')
        break


if __name__ == '__main__':
    main()
