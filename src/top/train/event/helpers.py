#!/usr/bin/env python3

from enum import Enum
from typing import (Callable, List, Any, Tuple, Union, Dict, Hashable)
from functools import partial
from dataclasses import dataclass
from simple_parsing import Serializable

import torch as th

from top.train.event.hub import Hub
from top.train.event.topics import Topic


class Periodic:
    """
    Invoke `callback` every N events.
    """

    class Keep(Enum):
        """
        Strategy for dealing with N events.
        NOTE(ycho): Currently, only `LAST` is supported.
        """
        FIRST = "FIRST"
        ALL = "ALL"
        LAST = "LAST"

    def __init__(self, period: int, callback: Callable[[Any], None],
                 keep: Keep = Keep.LAST):
        """

        """
        self.index = 0
        self.period = period
        self.callback = callback
        # NOTE(ycho): Currently, `keep` argument is ignored.
        self.keep = keep

    def __call__(self, *args, **kwargs):
        self.index += 1
        if self.index < self.period:
            return
        self.callback(*args, **kwargs)
        # NOTE(ycho): Reset index count.
        self.index = 0


class Collect:
    """
    Subscriber pattern for collecting a set of events
    and passing through all aggregated data at once.

    TODO(ycho): Consider synchronization threshold.
    """

    def __init__(self, hub: Hub, topic: Hashable,
                 keys: Tuple[Hashable, ...]):
        self.hub = hub
        self.topic = topic
        self.keys = keys
        self.data = {}

        self._subscribe()

    def _on_event(self, topic: Hashable, *args, **kwargs):
        self.data[topic] = (args, kwargs)

        # NOTE(ycho): shallow check for `collect` condition being met.
        if len(self.data) == len(self.keys):
            self._publish()

    def _subscribe(self):
        for key in self.keys:
            self.hub.subscribe(key, partial(self._on_event, key))

    def _publish(self):
        # TODO(ycho): Check for validity of `data`.
        self.hub.publish(self.topic, self.data)
        # NOTE(ycho): clearing data...
        self.data = {}


class Evaluator:
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

    def __init__(self,
                 opts: Settings,
                 hub: Hub,
                 model: th.nn.Module,
                 loader: th.utils.data.DataLoader,
                 eval_fn: Callable[[th.nn.Module, Any], None]
                 ):
        self.hub = hub
        self.opts = opts
        self.model = model
        self.loader = loader
        self.eval_fn = eval_fn

        self._subscribe()

    def _subscribe(self):
        self.hub.subscribe(Topic.STEP,
                           Periodic(self.opts.period, self._eval))

    def _eval(self, step: int):
        # NOTE(ycho): Set `model` to `eval` mode, but
        # cache the previous model cfg for restoration.
        prev_mode = self.model.training
        self.model.eval()

        # Run evaluation loop ...
        self.hub.publish(Topic.EVAL_BEGIN)
        # NOTE(ycho): is no_grad() necessary?
        # FIXME(ycho): What if the model requires gradient (e.g. saliency?)
        with th.no_grad():
            for (count, data) in enumerate(self.loader):
                outputs = self.eval_fn(self.model, data)
                self.hub.publish(Topic.EVAL_STEP,
                                 inputs=data,
                                 outputs=outputs)
                # Increment eval counts.
                if count >= self.opts.num_samples:
                    break
        self.hub.publish(Topic.EVAL_END)

        # NOTE(ycho): Restore previous training mode.
        self.model.train(prev_mode)
