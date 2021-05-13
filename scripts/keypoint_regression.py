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

from top.model.keypoint import KeypointNetwork2D
from top.model.loss import ObjectHeatmapLoss, KeypointDisplacementLoss

from top.data.transforms import (
    DenseMapsMobilePose,
    Normalize,
    InstancePadding,
    DrawKeypointMap
)
from top.data.schema import Schema
from top.data.load import (DatasetSettings, get_loaders)

from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device


@dataclass
class AppSettings(Serializable):
    model: KeypointNetwork2D.Settings = KeypointNetwork2D.Settings()

    # Dataset selection options.
    dataset: DatasetSettings = DatasetSettings()

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

    padding: InstancePadding.Settings = InstancePadding.Settings()
    maps: DenseMapsMobilePose.Settings = DenseMapsMobilePose.Settings()


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

        self.draw_kpt_map = DrawKeypointMap(
            DrawKeypointMap.Settings(
                as_displacement=False))

    def _on_loss(self, loss):
        """log training loss."""
        loss = loss.detach().cpu()

        # Update tensorboard ...
        self.writer.add_scalar('train_loss', loss,
                               global_step=self.step)

        # Update tqdm logger bar.
        self.tqdm.set_postfix(loss=loss)
        self.tqdm.update()

    def _on_train_out(self, inputs, outputs):
        """log training outputs."""

        # Fetch inputs ...
        with th.no_grad():
            input_image = inputs[Schema.IMAGE].detach()
            out_heatmap = outputs[Schema.HEATMAP].detach()
            target_heatmap = inputs[Schema.HEATMAP].detach()
            # NOTE(ycho): Only show for first image
            # feels a bit wasteful? consider better alternatives...
            out_kpt_map = self.draw_kpt_map(
                outputs[Schema.KEYPOINT_HEATMAP][0]).detach()
            target_kpt_map = self.draw_kpt_map(
                inputs[Schema.KEYPOINT_HEATMAP][0]).detach()

        # TODO(ycho): denormalize input image.
        image = th.clip(0.5 + (input_image[0] * 0.25), 0.0, 1.0)
        self.writer.add_image(
            'train_images',
            image.cpu(),
            global_step=self.step)

        for i_cls in range(out_heatmap.shape[1]):
            self.writer.add_image(F'out_heatmap/{i_cls}',
                                  out_heatmap[0, i_cls, None].cpu(),
                                  global_step=self.step)
            self.writer.add_image(F'target_heatmap/{i_cls}',
                                  target_heatmap[0, i_cls, None].cpu(),
                                  global_step=self.step)
        self.writer.add_image('out_kpt_map',
                              out_kpt_map.cpu(),
                              global_step=self.step)
        self.writer.add_image('target_kpt_map',
                              target_kpt_map.cpu(),
                              global_step=self.step)

    def _on_step(self, step):
        """save current step."""
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


class ModelAsTuple(th.nn.Module):
    """Workaround to avoid tracing bugs in add_graph from rejecting outputs of
    form Dict[Schema,Any]."""

    def __init__(self, model: th.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return tuple(v for (k, v) in self.model(inputs).items())


def main():
    # logging.basicConfig(level=logging.DEBUG)
    opts = AppSettings()
    opts = update_settings(opts)
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = KeypointNetwork2D(opts.model).to(device)
    # FIXME(ycho): Hardcoded lr == 1e-3
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    writer = th.utils.tensorboard.SummaryWriter(path.log)

    # NOTE(ycho): Forcing data loading on the CPU.
    # TODO(ycho): Consider scripted compositions?
    transform = Compose([
        DenseMapsMobilePose(opts.maps, th.device(device)),
        Normalize(Normalize.Settings()),
        InstancePadding(opts.padding)
    ])
    train_loader, test_loader = get_loaders(opts.dataset,
                                            device=th.device(device),
                                            batch_size=opts.batch_size,
                                            transform=transform)

    # NOTE(ycho): Synchronous event hub.
    hub = Hub()

    def _on_train_begin():

        # Save meta-parameters.
        opts.save(path.dir / 'opts.yaml')
        # NOTE(ycho): Currently `load` only works with a modified version of the
        # main SimpleParsing repository.
        # opts.load(path.dir / 'opts.yaml')

        # Generate tensorboard graph.
        data = next(iter(test_loader))
        dummy = data[Schema.IMAGE].to(device).detach()
        # NOTE(ycho): No need to set model to `eval`,
        # eval mode is set internally within add_graph().
        writer.add_graph(ModelAsTuple(model), dummy)

    hub.subscribe(
        Topic.TRAIN_BEGIN, _on_train_begin)

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
        Schema.HEATMAP: ObjectHeatmapLoss(key=Schema.HEATMAP),
        # Schema.DISPLACEMENT_MAP: KeypointDisplacementLoss(),
        Schema.KEYPOINT_HEATMAP: ObjectHeatmapLoss(
            key=Schema.KEYPOINT_HEATMAP)
    }

    def _loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        data = {k: (v.to(device) if isinstance(v, th.Tensor) else v)
                for (k, v) in data.items()}
        image = data[Schema.IMAGE]
        outputs = model(image)
        # Also make input/output pair from training
        # iterations available to the event bus.
        hub.publish(Topic.TRAIN_OUT,
                    inputs=data,
                    outputs=outputs)
        kpt_heatmap_loss = losses[Schema.KEYPOINT_HEATMAP](outputs, data)
        heatmap_loss = losses[Schema.HEATMAP](outputs, data)
        return (kpt_heatmap_loss + heatmap_loss)

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
