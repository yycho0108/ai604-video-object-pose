"""
(Like YOLO v3) 2D object detector -> 2D Bounding Box -> crop the images
feature map -> FC(confidence/scale/orientation) -> project 2D to 3D bounding box of cropped images

Reference:
    3D Bounding Box Estimation Using Deep Learning and Geometry(https://arxiv.org/abs/1612.00496)
    https://github.com/skhadem/3D-BoundingBox
"""

from dataclasses import dataclass, replace
from simple_parsing import Serializable

import torch as th
import torch.nn as nn
from torchvision.transforms import Compose

from top.train.trainer import Trainer
from top.train.callback import Callbacks, EvalCallback, SaveModelCallback

from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device

from top.model.bbox_3d import BoundingBoxRegressionModel
from top.model.loss_util import orientation_loss, generate_bins

from top.data.objectron_dataset_detection import Objectron
from top.data.schema import Schema


@dataclass
class AppSettings(Serializable):
    model: BoundingBoxRegressionModel.Settings = BoundingBoxRegressionModel.Settings()
    dataset: Objectron.Settings = Objectron.Settings()
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    train: Trainer.Settings = Trainer.Settings()
    batch_size: int = 8
    alpha: float = 0.6
    w: float = 0.4
    device: str = ''


def load_data(opts: AppSettings, device: th.device):
    # FIXME(Jiyong): data preprocessing for 3D bouning box regression
    transform = None
    data_opts = opts.dataset
    
    # For train data
    data_opts = replace(data_opts, train=True)
    train_dataset = Objectron(data_opts, transform)
    # For test data
    data_opts = replace(data_opts, train=False)
    test_dataset = Objectron(data_opts, transform)

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size)

    return train_dataset, test_dataset

def main():

    opts = AppSettings()
    opts = update_settings(opts)
    path = RunPath(opts.path)

    device = resolve_device(opts.device)
    model = BoundingBoxRegressionModel(opts.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader = load_data(opts, device=device)

    callbacks = Callbacks([])

    conf_loss_func = nn.CrossEntropyLoss().to(device)
    scale_loss_func = nn.MSELoss().to(device)
    orient_loss_func = orientation_loss

    def loss_fn(model: th.nn.Module, data):
        # Now that we're here, convert all inputs to the device.
        data = {k: v.to(device) for (k, v) in data.items()}

        image = data[Schema.IMAGE]
        orient, conf, dim = model(image)
        
        # FIXME(Jiyong): modify to Objectron
        truth_orient = data['Orientation'].float().to(device)
        truth_conf = data['Confidence'].long().to(device)
        truth_dim = data['scale'].float().to(device)

        scale_loss = scale_loss_func(dim, truth_dim)
        orient_loss = orient_loss_func(orient, truth_orient, truth_conf)

        truth_conf = th.max(truth_conf, dim=1)[1]
        conf_loss = conf_loss_func(conf, truth_conf)

        loss_theta = conf_loss + opts.w * orient_loss
        loss = opts.alpha * scale_loss + loss_theta

        return loss

    trainer = Trainer(opts.train, model, optimizer, loss_fn, callbacks, train_loader)

    trainer.train()

if __name__=='__main__':
    main()