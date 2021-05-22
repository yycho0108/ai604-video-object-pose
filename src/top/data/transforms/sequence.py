#!/usr/bin/env python3
"""Set of transforms related to sequences."""

__all__ = ['CropSequence']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as thio
from torchvision import transforms

from top.data.schema import Schema


class CropSequence:
    """Convert variable-length sequence to fixed-length representation.
    Intended to be used immediately after the raw output from `Objectron`.

    FIXME(ycho): Isn't it a bit wasteful to slice up a full video once
    and discard the reset?? Discuss strategies on overcoming correlated-samples issue
    """

    @dataclass
    class Settings(Serializable):
        seq_len: int = 8  # length of sub-sequence to extract
        max_num_inst: int = 4  # max number of instances in the frame
        stride: int = 4  # number of intermediate frames to skip

        # NOTE(ycho): Variable-length fields to apply padding for
        # `max_num_inst` ... needed for collating to batch
        pad_keys: Tuple[Schema, ...] = (
            Schema.IMAGE,
            Schema.KEYPOINT_2D,
            Schema.INSTANCE_NUM,
            Schema.TRANSLATION,
            Schema.ORIENTATION,
            Schema.SCALE,
            Schema.PROJECTION,
            Schema.KEYPOINT_NUM,
            Schema.VISIBILITY,
            Schema.CLASS,
            Schema.INTRINSIC_MATRIX
        )

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs: Dict[Hashable, th.Tensor]):
        if inputs is None:
            return None

        # Reject data with less than `seq_len` frames.
        n = inputs[Schema.IMAGE].shape[0]
        if n < self.opts.seq_len:
            return None

        # Extract a slice of length `seq_len` from the sequence.
        max_stride = n // self.opts.seq_len
        stride = min(max_stride, self.opts.stride)
        strided_len = stride * self.opts.seq_len

        i0 = np.random.randint(0, n - strided_len + 1)
        i1 = i0 + strided_len

        outputs = {k: (v[i0:i1:stride] if isinstance(v, th.Tensor) else v)
                   for (k, v) in inputs.items()}
        return outputs
