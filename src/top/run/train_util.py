#!/usr/bin/env python3

from dataclasses import dataclass
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from simple_parsing import Serializable
import torch as th
import logging


def _resolve_device(device: Union[None, str, th.device]):
    """ Resolve torch device. """
    if device:
        device = th.device(device)
    else:
        if th.cuda.is_available():
            # NOTE(ycho): does NOT work for multi-gpu settings.
            device = th.device('cuda:0')
            th.cuda.set_device(device)
        else:
            device = th.device('cpu')
    return device


class Saver(object):

    # Reserved keys
    KEY_MODEL = '__model__'
    KEY_OPTIM = '__optim__'

    def __init__(self,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer = None):
        self.model = model
        self.optim = optimizer

    def load(self, path: str):
        ckpt = th.load(path)

        # Load parameters from the checkpoint ...
        self.model.load_state_dict(ckpt.pop(self.KEY_MODEL))
        if self.optim:
            self.optim.load_state_dict(ckpt.pop(self.KEY_OPTIM))

        # Any remainder will be returned.
        return ckpt

    def save(self, path: str, **kwargs):
        # Model
        save_dict = {self.KEY_MODEL: self.model.state_dict()}

        # Optimizer
        if self.optim is not None:
            save_dict[self.KEY_OPTIM] = self.optim.state_dict()
        else:
            msg = F'Invalid optimizer={self.optim} on Saver.save()'
            logging.warn(msg)

        # Additional information
        save_dict.update(kwargs)
        th.save(save_dict, path)


class Callback(object):
    """
    Base class for periodic callbacks.
    TODO(ycho): Consider migration to e.g. fast.ai API
    when our custom solution cannot handle some sophisticated function.
    """

    def __init__(self, period: int, callback: Callable[[int], None]):
        self.period = period
        self.callback = callback
        self.last_call = 0

    def on_step(self, step: int):
        # NOTE(ycho): Don't use (step % self.period) which is common,
        # since if `num_env` != 0, lcm(period, num_env) may not be period.
        if step < self.last_call + self.period:
            return
        self.callback(step)
        self.last_call = step


class SaveModelCallback(Callback):
    """ Callback for saving models. """
    @dataclass
    class Settings(Serializable):
        pattern: str = 'save-{:06d}.zip'
        period: int = int(1e3)  # NOTE(ycho): by default, every 1000 steps

    def __init__(self, opts: Settings, path: str, saver: Saver):
        self.opts = opts
        self.path = Path(path)
        self.saver = saver
        super().__init__(self.opts.period, self._save)

    def _save(self, step: int):
        filename = self.path / self.opts.pattern.format(step)
        self.saver.save(filename)


class EvalCallback(Callback):
    """
    Callback for periodic model evaluation.
    Only sets up the boilerplate skeleton code, so `eval_fn` should do the heavy lifting.
    The expected `eval` operations might include:

    * computing relevant metrics (accuracy, recall, ...)
    * logging the relevant metrics (to e.g. TensorBoard, Terminal, ...)
    """

    @dataclass
    class Settings(Serializable):
        period: int = int(1e3)
        num_samples: int = 1

    def __init__(self, opts: Settings, model: th.nn.Module,
                 loader: th.utils.data.DataLoader, eval_fn=None):
        self.opts = opts
        self.model = model
        self.loader = loader
        self.eval_fn = eval_fn
        super().__init__(self.opts.period, self._eval)

    def _eval(self, step: int):
        # NOTE(ycho): Set `model` to `eval` mode.
        prev_mode = self.model.training
        self.model.eval()

        # Run evaluation loop ...
        count = 0
        for data in self.loader:

            # Eval step...
            with th.no_grad():
                self.eval_fn(self.model, data)

            # Increment eval counts
            count += 1
            if count >= self.opts.num_samples:
                break

        # NOTE(ycho): Restore previous training mode.
        self.model.train(prev_mode)


class Callbacks(Callback):
    """ Class to deal with a bunch of callbacks """

    def __init__(self, callbacks: Optional[List[Callback]] = []):
        self.callbacks = callbacks

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def on_step(self, step: int):
        for cb in self.callbacks:
            cb.on_step(step)


class Trainer(object):
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """

    @dataclass
    class Settings(Serializable):
        train_steps: int(1e4)
        eval_period: int(1e3)
        num_epochs: 1

    def __init__(self,
                 opts: Settings,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer,
                 loss_fn: Callable[[th.nn.Module, Any], th.Tensor],
                 callbacks: Callbacks,
                 loader: th.utils.data.DataLoader,
                 ):
        self.opts = opts
        self.model = model
        self.optim = optim

        # NOTE(ycho): loss_fn != th.nn.*Loss
        # since it directly takes `model` as argument.
        # This is (for now) intended to support maximum flexibility.
        # The simplest such version might be something like:
        # def loss_fn(model, data):
        #     x, y = data
        #     y_ = model(x)
        #     return th.square(y_-y).mean()

        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.loader = loader

    def train(self):
        self.model.train()

        step = 0
        # TODO(ycho): Consider registering more hooks.
        #self.callbacks.on_train(train)
        try:
            for epoch in self.opts.num_epochs:
                # self.callbacks.on_epoch(epoch)
                for i, data in self.loader:

                    # Compute loss ...
                    # NOTE(ycho): if one of the callbacks require training loss,
                    # e.g. for logging, simply register a hook to the loss module
                    # rather than trying to extract them here.
                    loss = self.loss_fn(self.model, data)

                    # Backprop + Optimize ...
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # Deal with callbacks ...
                    # == logging, saving, evaluation
                    self.callbacks.on_step(step)

        except KeyboardInterrupt:
            logging.info('Terminating training due to SIGINT')
        finally:
            # TODO(ycho): When an interrupt occurs, the current state
            # will ALWAYS be saved to a hardcoded file in a temporary directory.
            # Maybe this is a bad idea.
            Saver(self.model, self.optim).save('/tmp/model-backup.zip')


def main():
    # NOTE(ycho): Currently missing `loss_fn`
    #trainer = Trainer(Trainer.Settings(),
    #                  model, optimizer, callbacks, train_loader)


if __name__ == '__main__':
    main()
