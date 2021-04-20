#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from top.train.callback import (
    Callbacks, EvalCallback, SaveModelCallback)
from top.train.saver import Saver
from top.train.trainer import Trainer

from top.run.path_util import RunPath
from top.run.torch_util import resolve_device


@dataclass
class AppSettings(Serializable):
    device: str = ''
    data_dir: str = '~/.cache/ai604/mnist/MNIST_data/'
    batch_size: int = 32
    train: Trainer.Settings = Trainer.Settings(num_epochs=4)
    run: RunPath.Settings = RunPath.Settings(root='/tmp/mnist')


class ConvBlock(nn.Module):
    """ Simple Conv + BatnNorm + Relu block """

    def __init__(self, c_in: int, c_out: int, k: int = 3, *args, **kwargs):
        super().__init__()
        # NOTE(ycho): bias=False since followed by BatchNorm anyway.
        self.conv = nn.Conv2d(c_in, c_out, k, bias=False, *args, **kwargs)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Model(nn.Module):
    """
    Simple convolutional MNIST digit classification model.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 16)
        self.conv2 = ConvBlock(16, 32, stride=2)
        self.conv3 = ConvBlock(32, 64, stride=2)

        # inference
        self.flat = nn.Flatten()
        self.fc = nn.Linear(1600, 10)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def load_data(opts: AppSettings):
    """ Fetch pair of (train,test) loaders for MNIST data """
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


class Evaluator:
    """
    Aggregate statistics over multiple batch and log statistics.
    Currently only computes accuracy, potentially might be generalizable.
    """

    def __init__(self,
                 writer: th.utils.tensorboard.SummaryWriter,
                 device: th.device):
        self.writer = writer
        self.device = device

        self.prev_step = None
        self.num_samples = 0
        self.num_correct = 0.0

    def __call__(self, step: int, model: nn.Module, data):
        # Logging at the end of aggregation loop.
        if (step != self.prev_step):
            if self.prev_step is not None:
                # Log derived stat (accuracy).
                print(F'Acc = {self.num_correct}/{self.num_samples}')
                accuracy = (self.num_correct / self.num_samples)
                print(F'@{step} accuracy={accuracy} ')
                self.writer.add_scalar('accuracy', accuracy, step)

            # Reset stats.
            self.num_samples = 0
            self.num_correct = 0.0
            self.prev_step = step

        # Run evaluation ...
        inputs, target = data
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        output = model(inputs)

        # Aggregate statistics.
        num_correct = (target == th.argmax(output, dim=1)).sum()
        self.num_samples += inputs.shape[0]
        self.num_correct += num_correct


def main():
    # Settings ...
    opts = AppSettings()

    # Path configuration ...
    path = RunPath(opts.run)

    # Device resolution ...
    device = resolve_device(opts.device)

    # Data
    train_loader, test_loader = load_data(opts)

    # Model, loss
    model = Model().to(device)
    xs_loss = nn.CrossEntropyLoss()

    def loss_fn(model: nn.Module, data):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)
        return xs_loss(output, target)

    # Optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    # Callbacks, logging, ...
    writer = th.utils.tensorboard.SummaryWriter(path.log)
    evaluator = Evaluator(writer=writer, device=device)
    callbacks = Callbacks([
        EvalCallback(
            EvalCallback.Settings(
                num_samples=float('inf')
            ), model, test_loader, eval_fn=evaluator)
    ])

    # Trainer
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
