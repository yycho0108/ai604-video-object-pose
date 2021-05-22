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
    BoxPoints2D,
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

    xfm = Compose([
        BoxPoints2D(device=device),
        DrawKeypoints(DrawKeypoints.Settings()),
        DenseMapsMobilePose(DenseMapsMobilePose.Settings(), device),
        InstancePadding(InstancePadding.Settings()),
    ])

    dataset, _ = get_loaders(opts.dataset, device, None, xfm)

    for data in dataset:
        print(data['points_2d_debug'])
        print(data[Schema.KEYPOINT_2D])
        save_image(data[Schema.IMAGE] / 255.0, F'/tmp/img.png')
        for i, img in enumerate(data[Schema.HEATMAP]):
            save_image(img, F'/tmp/heatmap-{i}.png',
                       normalize=True)

        break


if __name__ == '__main__':
    main()
