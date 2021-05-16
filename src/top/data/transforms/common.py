#!/usr/bin/env python3
"""
Set of commonly transforms for which there is no clear alternative location.
"""

__all__ = ['Normalize', 'InstancePadding']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable
import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import Compose
from pytorch3d.transforms.transform3d import Transform3d

from top.run.torch_util import resolve_device
from top.run.app_util import update_settings
from top.data.schema import Schema


class Normalize:
    """
    Lightweight wrapper around torchvision.transforms.Normalize
    to reason with dictionary-valued inputs.

    NOTE(ycho): Expects image in UINT8 form!
    """

    @dataclass
    class Settings(Serializable):
        mean: float = 0.5
        std: float = 0.25
        in_place: bool = False
        keys: Tuple[Schema, ...] = (Schema.IMAGE,)

    def __init__(self, opts: Settings):
        self.opts = opts
        self.xfm = transforms.Normalize(opts.mean, opts.std)

    def __call__(self, inputs: dict):
        if not self.opts.in_place:
            outputs = inputs.copy()
        else:
            outputs = inputs

        for key in self.opts.keys:
            # NOTE(ycho): uint8 --> float32
            image = inputs[key].to(th.float32) / 255.0
            # Now apply the whitening transformation.
            outputs[key] = self.xfm(image)

        return outputs


class InstancePadding:
    """
    Class for padding variable number of object instances in a frame to some
    pre-defined maximum allowable number of instances, for collation.
    """
    @dataclass
    class Settings(Serializable):
        max_num_inst: int = 4
        pad_keys: Tuple[Schema, ...] = (
            Schema.KEYPOINT_2D,
            Schema.KEYPOINT_3D,
            Schema.TRANSLATION,
            Schema.ORIENTATION,
            Schema.SCALE,
            Schema.CLASS,
            # Schema.KEYPOINT_MAP,
            # Schema.DISPLACEMENT_MAP,
            Schema.KEYPOINT_NUM,  # I guess this is needed too...
            Schema.CROPPED_IMAGE
        )

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs: Dict[Schema, th.Tensor]):
        outputs = inputs.copy()
        for k in self.opts.pad_keys:
            if k not in inputs:
                continue
            pad = [0] * (2 * len(inputs[k].shape))
            pad[-2] = int(self.opts.max_num_inst - inputs[Schema.INSTANCE_NUM])
            x = th.as_tensor(inputs[k])
            outputs[k] = F.pad(x, pad)
            # logging.debug(F'PAD {k} {x.shape} -> {outputs[k].shape}')
        return outputs
