"""
(Like YOLO v3) 2D object detector -> 2D Bounding Box -> crop the images
feature map -> FC(confidence/scale/orientation) -> project 2D to 3D bounding box of cropped images

Reference:
    3D Bounding Box Estimation Using Deep Learning and Geometry(https://arxiv.org/abs/1612.00496)
    https://github.com/skhadem/3D-BoundingBox
"""

import logging
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import Tuple, Dict, Any
import torch.autograd.profiler as profiler
from tqdm.auto import tqdm

import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from top.data.transforms.common import InstancePadding, Normalize

from top.train.trainer import Trainer
from top.train.trainer import Saver
from top.train.event.hub import Hub
from top.train.event.topics import Topic
from top.train.event.helpers import (Collect, Periodic, Evaluator)

from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device
from top.run.draw_regressed_bbox import plot_regressed_3d_bbox

from top.model.bbox_3d import BoundingBoxRegressionModel

from top.data.load import (DatasetSettings, collate_cropped_img, get_loaders)
from top.data.schema import Schema
from top.data.bbox_reg_util import CropObject


@dataclass
class AppSettings(Serializable):
    model: BoundingBoxRegressionModel.Settings = BoundingBoxRegressionModel.Settings()
    
    # Dataset selection options.
    dataset: DatasetSettings = DatasetSettings()
    padding: InstancePadding.Settings = InstancePadding.Settings()
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-box')
    train: Trainer.Settings = Trainer.Settings()
    # FIXME(Jiyong): need to test padding for batch
    batch_size: int = 8
    alpha: float = 0.5
    device: str = 'cuda'
    log_period: int = 32
    save_period: int = 1000
    eval_period: int = 1000

    profile: bool = False
    load_ckpt: str = ''


class TrainLogger:
    """
    Logging during training - specifically, tqdm-based logging to the shell and tensorboard.
    """
    def __init__(self, hub: Hub, writer: SummaryWriter, period: int):
        self.step = None
        self.hub = hub
        self.writer = writer
        self.tqdm = tqdm()
        self.period = period
        self._subscribe()

    def _on_loss(self, loss):
        """log training loss."""
        loss_total = loss["total"].detach().cpu()
        loss_scale = loss["scale"].detach().cpu()
        loss_orient = loss["orientation"].detach().cpu()

        # Update tensorboard ...
        self.writer.add_scalar('train_loss_total', loss["total"], global_step=self.step)
        self.writer.add_scalar('train_loss_scale', loss["scale"], global_step=self.step)
        self.writer.add_scalar('train_loss_orientation', loss["orientation"], global_step=self.step)

        # update tqdm logger bar.
        self.tqdm.set_postfix(loss=loss)
        self.tqdm.update()

    def _on_train_out(self, inputs, outputs):
        """log tranining outputs."""
        # Fetch inputs ...
        with th.no_grad():
            input_image = inputs[Schema.IMAGE].detach().cpu()
            proj_matrix = inputs[Schema.PROJECTION].detach().cpu()
            keypoints_2d = inputs[Schema.KEYPOINT_2D].detach().cpu()
            translations = inputs[Schema.TRANSLATION].detach().cpu()

            dimensions = outputs[Schema.SCALE].detach().cpu()
            quaternion = outputs[Schema.QUATERNION].detach().cpu()

        self.writer.add_image('train_images', input_image[0].cpu(), global_step=self.step)

        image_with_box = plot_regressed_3d_bbox(input_image, keypoints_2d, proj_matrix, dimensions, quaternion, translations)
        
        self.writer.add_image('train_result_images', image_with_box[0:3], global_step=self.step)

        print(inputs[Schema.INSTANCE_NUM])
        print(inputs[Schema.INDEX])
        print(inputs[Schema.INDEX].shape)

    def _on_step(self, step):
        """save current step"""
        self.step = step
    
    def _subscribe(self):
        self.hub.subscribe(Topic.STEP, self._on_step)
        # NOTE(ycho): Log loss only periodically.
        self.hub.subscribe(Topic.TRAIN_LOSS, Periodic(self.period, self._on_loss))
        self.hub.subscribe(Topic.TRAIN_OUT, Periodic(self.period, self._on_train_out))
    
    def __del__(self):
        self.tqdm.close()
               

def main():
    logging.basicConfig(level=logging.WARN)
    opts = AppSettings()
    opts = update_settings(opts)
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = BoundingBoxRegressionModel(opts.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-5)
    writer = SummaryWriter(path.log)

    transform = Compose([CropObject(CropObject.Settings()),
                         Normalize(Normalize.Settings(keys=(Schema.CROPPED_IMAGE,)))])
    train_loader, test_loader = get_loaders(opts.dataset,
                                            device=th.device('cpu'),
                                            batch_size=opts.batch_size,
                                            transform=transform,
                                            collate_fn = collate_cropped_img)

    # NOTE(ycho): Synchronous event hub.
    hub = Hub()

    # Save meta-parameters.
    def _save_params():
        opts.save(path.dir / 'opts.yaml')
    hub.subscribe(Topic.TRAIN_BEGIN, _save_params)

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
        # TODO(Jiyong): hardcode for cropped image size
        crop_img = data[Schema.CROPPED_IMAGE].view(-1, 3, 224, 224)
        return model(crop_img.to(device))
    evaluator = Evaluator(
        Evaluator.Settings(period=opts.eval_period),
        hub, model, test_loader, _eval_fn)

    # TODO(Jiyong):
    # All metrics evaluation should reset stats at eval_begin(),
    # aggregate stats at eval_step(),
    # and output stats at eval_end(). These signals are all implemented.
    # What are the appropriate metrics to implement for bounding box regression?
    def _on_eval_step(inputs, outputs):
        pass
    hub.subscribe(Topic.EVAL_STEP, _on_eval_step)

    collect = Collect(hub, Topic.METRICS, [])

    def _log_all(metrics: Dict[Topic, Any]):
        pass
    hub.subscribe(Topic.METRICS, _log_all)

    orientation_loss_func = nn.L1Loss().to(device)
    scale_loss_func = nn.L1Loss().to(device)

    def _loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        image = data[Schema.CROPPED_IMAGE].to(device)
        c, h, w = image.shape[-3:]
        image = image.view(-1, c, h, w)
        truth_quat = data[Schema.QUATERNION].to(device)
        truth_quat = truth_quat.view(-1,4)
        truth_dim = data[Schema.SCALE].to(device)
        truth_dim = truth_dim.view(-1,3)
        truth_trans = data[Schema.TRANSLATION].to(device)
        truth_trans = truth_trans.view(-1,3)

        dim, quat = model(image)

        outputs = {}
        outputs[Schema.SCALE] = dim
        outputs[Schema.QUATERNION] = quat

        # Also make input/output pair from training
        # iterations available to the event bus.
        hub.publish(Topic.TRAIN_OUT,
                    inputs = data,
                    outputs = outputs)

        loss = {}

        scale_loss = scale_loss_func(dim, truth_dim)
        orient_loss = orientation_loss_func(quat, truth_quat)
        total_loss = opts.alpha * scale_loss + orient_loss

        loss["total"] = total_loss
        loss["scale"] = scale_loss
        loss["orientation"] = orient_loss

        return loss

    ## Load from checkpoint
    if opts.load_ckpt:
        logging.info(F'Loading checkpoint {opts.load_ckpt} ...')
        Saver(model, optimizer).load(opts.load_ckpt)

    ## Trainer
    trainer = Trainer(opts.train,
                      model,
                      optimizer,
                      _loss_fn,
                      hub,
                      train_loader)

    # Train, optionally profile
    if opts.profile:
        try:
            with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                trainer.train()
        finally:
            print(
                prof.key_averages().table(
                    sort_by='cpu_time_total',
                    row_limit=16))
            prof.export_chrome_trace("/tmp/trace.json")
    else:
        trainer.train()


if __name__ == '__main__':
    main()
