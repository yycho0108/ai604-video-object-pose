"""
(Like YOLO v3) 2D object detector -> 2D Bounding Box -> crop the images
feature map -> FC(confidence/scale/orientation) -> project 2D to 3D bounding box of cropped images

Reference:
    3D Bounding Box Estimation Using Deep Learning and Geometry(https://arxiv.org/abs/1612.00496)
    https://github.com/skhadem/3D-BoundingBox
"""

from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import Tuple, Dict, Any
from tqdm.auto import tqdm

import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from top.data.transforms.common import InstancePadding

from top.train.trainer import Trainer
from top.train.trainer import Saver
from top.train.event.hub import Hub
from top.train.event.topics import Topic
from top.train.event.helpers import (Collect, Periodic, Evaluator)

from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device

from top.model.bbox_3d import BoundingBoxRegressionModel

from top.data.load import (DatasetSettings, get_loaders)
from top.data.schema import Schema
from top.data.bbox_reg_util import CropObject


@dataclass
class AppSettings(Serializable):
    model: BoundingBoxRegressionModel.Settings = BoundingBoxRegressionModel.Settings()
    
    # Dataset selection options.
    dataset: DatasetSettings = DatasetSettings()
    padding: InstancePadding.Settings = InstancePadding.Settings()
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    train: Trainer.Settings = Trainer.Settings(train_steps=1)
    # FIXME(Jiyong): need to test padding for batch
    batch_size: int = 2
    alpha: float = 0.5
    device: str = 'cpu'
    log_period: int = 32
    save_period: int = 100
    eval_period: int = 100


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
        loss = loss.detach().cpu()

        # Update tensorboard ...
        self.writer.add_scalar('train_loss', loss, global_step=self.step)

        # update tqdm logger bar.
        self.tqdm.set_postfix(loss=loss)
        self.tqdm.update()

    def _on_train_out(self, inputs, outputs):
        """log tranining outputs."""
        # Fetch inputs ...
        with th.no_grad():
            input_image = (inputs(Schema.IMAGE).detach())
        
        self.writer.add_image('train_images', input_image[0].cpu(), global_step=self.step)

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

    opts = AppSettings()
    opts = update_settings(opts)
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = BoundingBoxRegressionModel(opts.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(path.log)

    transform = Compose([CropObject(CropObject.Settings()),])
    train_loader, test_loader = get_loaders(opts.dataset,
                                            device=th.device('cpu'),
                                            batch_size=opts.batch_size,
                                            transform=transform)

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
        return model(data[Schema.IMAGE].to(device))
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

    orientation_loss_func = nn.MSELoss().to(device)
    scale_loss_func = nn.MSELoss().to(device)

    def _loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        # TODO(Jiyong): chage to collate_fn
        image = data[Schema.CROPPED_IMAGE].to(device)
        _, _, c, h, w = image.shape
        image = image.view(-1, c, h, w)
        truth_orient = data[Schema.ORIENTATION].to(device)
        truth_orient = truth_orient.view(-1,4)
        truth_dim = data[Schema.SCALE].to(device)
        truth_dim = truth_dim.view(-1,3)

        dim, quat = model(image)

        scale_loss = scale_loss_func(dim, truth_dim)
        orient_loss = orientation_loss_func(quat, truth_orient)
        loss = opts.alpha * scale_loss + orient_loss

        return loss

    trainer = Trainer(opts.train,
                      model,
                      optimizer,
                      _loss_fn,
                      hub,
                      train_loader)

    print('======Training Start======')
    trainer.train()
    print('======Training End======')


if __name__ == '__main__':
    main()
