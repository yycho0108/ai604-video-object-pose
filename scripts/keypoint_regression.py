#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th

from top.train.trainer import Trainer
from top.train.callback import (
    Callbacks, EvalCallback, SaveModelCallback)
from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device

from top.model.keypoint import KeypointNetwork2D
from top.model.loss import KeypointCrossEntropyLoss
# from top.data.objectron_dataset_detection import Objectron
from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.transforms import BoxHeatmap


@dataclass
class AppSettings(Serializable):
    model: KeypointNetwork2D.Settings = KeypointNetwork2D.Settings()
    data: ColoredCubeDataset.Settings = ColoredCubeDataset.Settings()
    # NOTE(ycho): root run path is set to tmp dir y default.
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    train: Trainer.Settings = Trainer.Settings()
    device: str = ''


def load_data(opts: AppSettings):
    """ Fetch pair of (train,test) loaders for MNIST data """
    # FIXME(ycho): Incorrect data loading scheme, fix.
    # Configure loaders.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(opts.data_dir, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(opts.data_dir, train=False, download=False,
                                  transform=transform)

    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size)
    test_loader = th.utils.data.DataLoader(
        test_dataset, batch_size=opts.batch_size)

    return (train_loader, test_loader)


def main():
    opts = AppSettings()
    opts = update_settings(opts)
    # opts.save('/tmp/test.yaml')
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = KeypointNetwork2D(opts.model).to(device)

    dataset = ColoredCubeDataset(opts.data, device=device)
    # NOTE(ycho): `ColoredCubeDataset` natively supports batch size,
    # thus skip collation in DataLoader.
    data_loader = th.utils.data.DataLoader(dataset, batch_size=None)

    callbacks = Callbacks([])

    loss = KeypointCrossEntropyLoss()

    # FIXME(ycho): Still drafting this section.

    #def loss_fn(model: nn.Module, data):
    #    image = data['image']
    #    target = data['keypoint_map']
    #    # points = data['points']
    #    # target = compute_keypoint_map(data['object/orientation'], ...)
    #    output = model(inputs)
    #    return loss(output, target)

    ## Trainer
    #trainer = Trainer(
    #    opts.train,
    #    model,
    #    optimizer,
    #    loss_fn,
    #    callbacks,
    #    data_loader)

    #trainer.train()


if __name__ == '__main__':
    main()
