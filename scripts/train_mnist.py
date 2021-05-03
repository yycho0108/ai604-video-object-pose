#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from top.train.event.hub import Hub
from top.train.event.topics import Topic
from top.train.event.helpers import (Collect, Periodic, Evaluator)

from top.train.saver import Saver
from top.train.trainer import Trainer

from top.run.app_util import update_settings
from top.run.path_util import RunPath
from top.run.torch_util import resolve_device


@dataclass
class AppSettings(Serializable):
    device: str = ''
    data_dir: str = '~/.cache/ai604/mnist/MNIST_data/'
    batch_size: int = 32
    train: Trainer.Settings = Trainer.Settings(num_epochs=4)
    run: RunPath.Settings = RunPath.Settings(root='/tmp/mnist')
    eval_period: int = int(1e3)


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


class Accuracy:
    """
    Node that computes and produces accuracy metric.
    """

    def __init__(self, hub: Hub, topic: str):
        self.hub = hub
        self.num_samples = 0
        self.num_correct = 0.0
        self.topic = topic
        self._subscribe()

    def _on_eval_begin(self):
        self.num_samples = 0
        self.num_correct = 0.0

    def _on_eval_step(self, inputs, outputs):
        _, target = inputs
        target = target.to(outputs.get_device())
        num_correct = (target == th.argmax(outputs, dim=1)).sum()
        self.num_samples += target.shape[0]  # Assume batch dim == 0.
        self.num_correct += num_correct

    def _on_eval_end(self):
        accuracy = (self.num_correct / self.num_samples)
        self.hub.publish(self.topic, accuracy)

    def _subscribe(self):
        self.hub.subscribe(Topic.EVAL_BEGIN, self._on_eval_begin)
        self.hub.subscribe(Topic.EVAL_STEP, self._on_eval_step)
        self.hub.subscribe(Topic.EVAL_END, self._on_eval_end)


def main():
    # Settings ...
    opts = AppSettings()
    opts = update_settings(opts)

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

    def _eval_fn(model, data):
        inputs, _ = data
        output = model(inputs.to(device))
        return output

    hub = Hub()

    # TODO(ycho): The default behavior of evaluator (num_samples==1)
    # might be confusing and unintuitive - prefer more reasonable default?
    evaluator = Evaluator(
        Evaluator.Settings(period=opts.eval_period, num_samples=128),
        hub, model, test_loader, _eval_fn)

    accuracy = Accuracy(hub, 'accuracy')
    metrics = Collect(hub, Topic.METRICS,
                      (Topic.STEP, 'accuracy'))

    def _on_metrics(data):
        # TODO(ycho): Fix clunky syntax with `Collect`.
        step_arg, _ = data[Topic.STEP]
        step = step_arg[0]
        acc_arg, _ = data['accuracy']
        accuracy = acc_arg[0]

        # Print to stdout ...
        print(F'@{step} accuracy={accuracy} ')

        # Tensorboard logging ...
        writer.add_scalar('accuracy', accuracy, step)

    hub.subscribe(Topic.METRICS, _on_metrics)

    # Trainer
    trainer = Trainer(
        opts.train,
        model,
        optimizer,
        loss_fn,
        hub,
        train_loader)

    trainer.train()


if __name__ == '__main__':
    main()
