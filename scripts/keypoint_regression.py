#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK

import enum
import logging
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import Dict, Any
from tqdm.auto import tqdm

import torch as th
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

from top.train.saver import Saver
from top.train.trainer import Trainer

from top.train.event.hub import Hub
from top.train.event.topics import Topic
from top.train.event.helpers import (Collect, Periodic, Evaluator)
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

# NOTE(ycho): Required for dealing with our `enum`
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class DatasetOptions(enum.Enum):
    CUBE = "CUBE"
    OBJECTRON = "OBJECTRON"
    SAMPLE_OBJECTRON = "SAMPLE_OBJECTRON"


# NOTE(ycho): Register encoder-decoder pair for `DatasetOptions` enum.
# NOTE(ycho): Parsing from type annotations: only available for python>=3.7.
@encode.register(DatasetOptions)
def encode_dataset_options(obj: DatasetOptions) -> str:
    """Encode the enum with the underlying `str` representation. """
    return str(obj.value)


register_decoding_fn(DatasetOptions, DatasetOptions.__getitem__)


@dataclass
class AppSettings(Serializable):
    model: KeypointNetwork2D.Settings = KeypointNetwork2D.Settings()

    # Select dataset among different options, by name.
    dataset: DatasetOptions = DatasetOptions.OBJECTRON

    cube: ColoredCubeDataset.Settings = ColoredCubeDataset.Settings()
    objectron: Objectron.Settings = Objectron.Settings()
    # TODO(ycho): SampleObjectron should be deprecated in favor of just
    # using CachedDataset() directly. (Confusing settings)
    sample_objectron: SampleObjectron.Settings = SampleObjectron.Settings()

    # NOTE(ycho): root run path is set to tmp dir y default.
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    train: Trainer.Settings = Trainer.Settings()
    batch_size: int = 8
    device: str = ''

    # Logging interval / every N train steps
    log_period: int = int(32)

    # Checkpointing interval / every N train steps
    save_period: int = int(1e3)

    # Evaluation interval / every N train steps
    eval_period: int = int(1e3)


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


class TrainLogger:
    """
    Logging during training - specifically, tqdm-based logging to the shell and tensorboard.
    """

    def __init__(self, hub: Hub, writer: th.utils.tensorboard.SummaryWriter,
                 period: int):
        self.step = None
        self.hub = hub
        self.writer = writer
        self.tqdm = tqdm()
        self.period = period
        self._subscribe()

    def _on_loss(self, loss):
        """ log training loss. """
        loss = loss.detach().cpu()

        # Update tensorboard ...
        self.writer.add_scalar('train_loss', loss,
                               global_step=self.step)

        # Update tqdm logger bar.
        self.tqdm.set_postfix(loss=loss)
        self.tqdm.update()

    def _on_train_out(self, inputs, outputs):
        """ log training outputs. """

        # Fetch inputs ...
        input_image = (inputs[Schema.IMAGE].detach())
        out_heatmap = (th.sigmoid(outputs[Schema.HEATMAP_LOGITS]).detach())
        target_heatmap = inputs[Schema.HEATMAP].detach()

        # TODO(ycho): denormalize input image.
        self.writer.add_image(
            'train_images',
            input_image[0].cpu(),
            global_step=self.step)
        self.writer.add_images('out_heatmap',
                               out_heatmap[0, :, None].cpu(),
                               global_step=self.step)
        self.writer.add_images('target_heatmap',
                               target_heatmap[0, :, None].cpu(),
                               global_step=self.step)

    def _on_step(self, step):
        """ save current step """
        self.step = step

    def _subscribe(self):
        self.hub.subscribe(Topic.STEP, self._on_step)
        # NOTE(ycho): Log loss only periodically.
        self.hub.subscribe(Topic.TRAIN_LOSS,
                           Periodic(self.period, self._on_loss))
        self.hub.subscribe(Topic.TRAIN_OUT,
                           Periodic(self.period, self._on_train_out))

    def __del__(self):
        self.tqdm.close()


def main():
    # logging.basicConfig(level=logging.DEBUG)
    opts = AppSettings()
    opts = update_settings(opts)
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = KeypointNetwork2D(opts.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    writer = th.utils.tensorboard.SummaryWriter(path.log)

    # NOTE(ycho): Force data loading on the CPU.
    # train_loader, test_loader = load_data(opts, device=device)
    train_loader, test_loader = load_data(opts, device=th.device('cpu:0'))

    # NOTE(ycho): Synchronous event hub.
    hub = Hub()

    # Save meta-parameters.
    def _save_params():
        opts.save(path.dir / 'opts.yaml')
        # NOTE(ycho): Only works with a modified version of the
        # main SimpleParsing repository.
        # opts.load(path.dir / 'opts.yaml')
    hub.subscribe(
        Topic.TRAIN_BEGIN, _save_params)

    # Periodically log training statistics.
    # FIXME(ycho): hardcoded logging period.
    # NOTE(ycho): Currently only plots `loss`.
    collect = Collect(hub, Topic.METRICS, [])
    train_logger = TrainLogger(hub, writer, opts.log_period)

    # Periodically save model, per epoch.
    # TODO(ycho): Consider folding this callback inside Trainer().
    hub.subscribe(
        Topic.EPOCH,
        lambda epoch: Saver(
            model,
            optimizer).save(
            path.ckpt /
            F'epoch-{epoch}.zip'))

    # Periodically save model, per N training steps.
    # TODO(ycho): Consider folding this callback inside Trainer()
    # and adding {save_period} args to Trainer instead.
    hub.subscribe(
        Topic.STEP,
        Periodic(opts.save_period, lambda step: Saver(
            model,
            optimizer).save(
            path.ckpt /
            F'step-{step}.zip')))

    # Periodically evaluate model, per N training steps.
    # NOTE(ycho): Load and process test data ...
    # TODO(ycho): Consider folding this callback inside Trainer()
    # and adding {test_loader, eval_fn} args to Trainer instead.
    def _eval_fn(model, data):
        return model(data[Schema.IMAGE].to(device))
    evaluator = Evaluator(
        Evaluator.Settings(period=opts.eval_period),
        hub, model, test_loader, _eval_fn)

    # TODO(ycho):
    # All metrics evaluation should reset stats at eval_begin(),
    # aggregate stats at eval_step(),
    # and output stats at eval_end(). These signals are all implemented.
    # What are the appropriate metrics to implement for keypoint regression?
    # - keypoint matching F1 score(?)
    # - loss_fn() but for the evaluation datasets
    def _on_eval_step(inputs, outputs):
        pass
    hub.subscribe(Topic.EVAL_STEP, _on_eval_step)

    collect = Collect(hub, Topic.METRICS, [])

    def _log_all(metrics: Dict[Topic, Any]):
        pass
    hub.subscribe(Topic.METRICS, _log_all)

    # TODO(ycho): weight the losses with some constant ??
    losses = {
        Schema.DISPLACEMENT_MAP: KeypointDisplacementLoss(),
        Schema.HEATMAP: ObjectHeatmapLoss()
    }

    def _loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        data = {k: v.to(device) for (k, v) in data.items()}
        image = data[Schema.IMAGE]
        outputs = model(image)
        # Also make input/output pair from training
        # iterations available to the event bus.
        hub.publish(Topic.TRAIN_OUT,
                    inputs=data,
                    outputs=outputs)
        displacement_loss = losses[Schema.DISPLACEMENT_MAP](outputs, data)
        heatmap_loss = losses[Schema.HEATMAP](outputs, data)
        return (displacement_loss + heatmap_loss)

    ## Trainer
    trainer = Trainer(
        opts.train,
        model,
        optimizer,
        _loss_fn,
        hub,
        train_loader)

    trainer.train()


if __name__ == '__main__':
    main()
