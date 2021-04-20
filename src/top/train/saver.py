#!/usr/bin/env python3

from dataclasses import dataclass
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from simple_parsing import Serializable
import torch as th
import logging


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
