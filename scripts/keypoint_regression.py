#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK

import enum
import logging
from dataclasses import dataclass, replace
from simple_parsing import Serializable

import torch as th
from torchvision.transforms import Compose

from top.train.trainer import Trainer
from top.train.callback import (
    Callbacks, EvalCallback, SaveModelCallback)
from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device

from top.model.keypoint import KeypointNetwork2D
from top.model.loss import ObjectHeatmapLoss, KeypointDisplacementLoss
from top.data.objectron_dataset_detection import Objectron, SampleObjectron
from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.transforms import DenseMapsMobilePose, Normalize
from top.data.schema import Schema


class DatasetOptions(enum.Enum):
    CUBE = "CUBE"
    OBJECTRON = "OBJECTRON"
    SAMPLE_OBJECTRON = "SAMPLE_OBJECTRON"


@dataclass
class AppSettings(Serializable):
    model: KeypointNetwork2D.Settings = KeypointNetwork2D.Settings()

    # Select dataset among different options, by name.
    dataset: DatasetOptions = DatasetOptions.OBJECTRON

    cube: ColoredCubeDataset.Settings = ColoredCubeDataset.Settings()
    objectron: Objectron.Settings = Objectron.Settings()
    sample_objectron: SampleObjectron.Settings = SampleObjectron.Settings()

    # NOTE(ycho): root run path is set to tmp dir y default.
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    train: Trainer.Settings = Trainer.Settings()
    batch_size: int = 8
    device: str = ''


def load_data(opts: AppSettings, device: th.device):
    """ Fetch pair of (train,test) loaders for MNIST data """
    # TODO(ycho): Consider scripted compositions?
    transform = Compose([
        DenseMapsMobilePose(DenseMapsMobilePose.Settings(), device),
        Normalize(Normalize.Settings())
    ])

    # TODO(ycho): Prefer unified interface across dataset instances.
    # FIXME(ycho): Maybe not the most elegant idea.
    # The code looks complex due to the nested replacement.
    # In general, we consider `dataclass` instances to be immutable,
    # which is why such a workaround is necessary.
    # TODO(ycho): Consider alternatives.
    if opts.dataset == DatasetOptions.CUBE:
        data_opts = opts.cube
        train_dataset = ColoredCubeDataset(
            data_opts, device, transform)
        test_dataset = ColoredCubeDataset(
            data_opts, device, transform)
    elif opts.dataset == DatasetOptions.OBJECTRON:
        data_opts = opts.objectron

        data_opts = replace(data_opts, train=True)
        train_dataset = Objectron(data_opts, transform)

        data_opts = replace(data_opts, train=False)
        test_dataset = Objectron(data_opts, transform)
    elif opts.dataset == DatasetOptions.SAMPLE_OBJECTRON:
        objectron_opts = replace(
            opts.sample_objectron.objectron, train=True)
        data_opts = replace(
            opts.sample_objectron, objectron=objectron_opts)
        train_dataset = SampleObjectron(data_opts, transform)

        objectron_opts = replace(
            opts.sample_objectron.objectron, train=False)
        data_opts = replace(
            opts.sample_objectron, objectron=objectron_opts)
        test_dataset = SampleObjectron(data_opts, transform)
    else:
        raise ValueError(F'Invalid dataset choice : {opts.dataset}')

    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size)

    test_loader = th.utils.data.DataLoader(
        test_dataset, batch_size=opts.batch_size)

    return (train_loader, test_loader)


def main():
    # logging.basicConfig(level=logging.DEBUG)
    opts = AppSettings()
    opts = update_settings(opts)
    # opts.save('/tmp/test.yaml')
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = KeypointNetwork2D(opts.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    # NOTE(ycho): Force data loading on the CPU.
    # train_loader, test_loader = load_data(opts, device=device)
    train_loader, test_loader = load_data(opts, device=th.device('cpu:0'))

    callbacks = Callbacks([])

    # TODO(ycho): weight the losses with some constant ??
    losses = {
        Schema.DISPLACEMENT_MAP: KeypointDisplacementLoss(),
        Schema.HEATMAP: ObjectHeatmapLoss()
    }

    def loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        data = {k: v.to(device) for (k, v) in data.items()}

        image = data[Schema.IMAGE]
        outputs = model(image)
        displacement_loss = losses[Schema.DISPLACEMENT_MAP](outputs, data)
        heatmap_loss = losses[Schema.HEATMAP](outputs, data)
        return (displacement_loss + heatmap_loss)

    ## Trainer
    trainer = Trainer(
        opts.train,
        model,
        optimizer,
        loss_fn,
        callbacks,
        train_loader)

    trainer.train()


if __name__ == '__main__':
    main()
