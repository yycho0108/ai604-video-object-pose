#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th
from torchvision.transforms import Compose

from top.run.app_util import update_settings
from top.run.torch_util import resolve_device
from top.data.load import (DatasetSettings, get_loaders)
from top.data.transforms import (
    DenseMapsMobilePose,
    Normalize,
    InstancePadding)


@dataclass
class Settings(Serializable):
    dataset: DatasetSettings = DatasetSettings()
    padding: InstancePadding.Settings = InstancePadding.Settings()
    batch_size: int = 8
    device: str = ''
    num_samples: int = 8


def main():
    opts = Settings()
    opts = update_settings(opts)
    device = resolve_device(opts.device)
    transform = Compose([
        DenseMapsMobilePose(DenseMapsMobilePose.Settings(), device),
        Normalize(Normalize.Settings()),
        InstancePadding(InstancePadding.Settings())
    ])
    data_loader, _ = get_loaders(
        opts.dataset, device, opts.batch_size, transform)

    for i, data in enumerate(data_loader):
        for k, v in data.items():
            if isinstance(v, th.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        if i >= opts.num_samples:
            break


if __name__ == '__main__':
    main()
