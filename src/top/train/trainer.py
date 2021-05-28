#!/usr/bin/env python3

__all__ = ["Trainer"]

from dataclasses import dataclass
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from simple_parsing import Serializable
import torch as th
import logging
from tqdm.auto import tqdm

from top.train.saver import Saver
from top.train.event.hub import Hub
from top.train.event.topics import Topic


class Trainer(object):
    """Generic trainer for a pytorch nn.Module.

    Intended to be flexible, modify as needed.
    """

    @dataclass
    class Settings(Serializable):
        train_steps: int = int(1e4)
        # NOTE(ycho): Large # of epochs by default,
        # Such that the tranining would *generally* terminate
        # due to `train_steps`.
        num_epochs: int = int(100)

    def __init__(self,
                 opts: Settings,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer,
                 loss_fn: Callable[[th.nn.Module, Any], th.Tensor],
                 hub: Hub,
                 loader: th.utils.data.DataLoader
                 ):
        """
        Args:
            opts: Trainer options.
            model: The model to train.
            optimizer: Optimizer, e.g. `Adam`.
            loss_fn: The function that maps (model, next(iter(loader))) -> cost.
            loader: Iterable data loader.
        """
        self.opts = opts
        self.model = model
        self.optim = optimizer

        # NOTE(ycho): loss_fn != th.nn.*Loss
        # since it directly takes `model` as argument.
        # This is (for now) intended to support maximum flexibility.
        # The simplest such version might be something like:
        # def loss_fn(model, data):
        #     x, y = data
        #     y_ = model(x)
        #     return th.square(y_-y).mean()

        self.loss_fn = loss_fn
        self.hub = hub
        self.loader = loader

    def _train(self):
        """Internal function for dealing with the inner training loop."""
        step = 0
        for epoch in range(self.opts.num_epochs):
            self.hub.publish(Topic.EPOCH, epoch)
            for i, data in enumerate(self.loader):
                # Compute loss ...
                # NOTE(ycho): if one of the callbacks require training loss,
                # e.g. for logging, simply register a hook to the loss module
                # rather than trying to extract them here.
                loss = self.loss_fn(self.model, data)
                self.hub.publish(Topic.TRAIN_LOSS, loss)

                # Backprop + Optimize ...
                self.optim.zero_grad()
                loss["total"].backward()
                self.optim.step()

                # Emit `step` event.
                # == logging, saving, evaluation
                self.hub.publish(Topic.STEP, step)
                step += 1

                if step >= self.opts.train_steps:
                    return

    def train(self):
        self.model.train()

        # TODO(ycho): Consider registering more hooks.
        self.hub.publish(Topic.TRAIN_BEGIN)
        try:
            self._train()
        except KeyboardInterrupt:
            logging.info('Terminating training due to SIGINT')
        finally:
            # TODO(ycho): When an interrupt occurs, the current state
            # will ALWAYS be saved to a hardcoded file in a temporary directory.
            # Maybe this is a bad idea.
            Saver(self.model, self.optim).save('/tmp/model-backup.zip')


def main():
    """Simplest possible trainer setup."""
    model = th.nn.Linear(1, 1)
    optim = th.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(model, data):
        x, target = data
        output = model(x)
        loss = th.mean(th.square(output - target))
        return loss

    def get_loader():
        while True:
            # NOTE(ycho): `32` here is a dummy fictitious batch size.
            x = th.empty((32, 1), dtype=th.float32)
            y = th.empty((32, 1), dtype=th.float32)
            yield (x, y)

    trainer = Trainer(
        Trainer.Settings(train_steps=1),
        model,
        optim,
        loss_fn,
        Hub(),
        get_loader())

    trainer.train()


if __name__ == '__main__':
    main()
