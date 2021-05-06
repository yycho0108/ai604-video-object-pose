#!/usr/bin/env python3
"""
Set of transforms related to sequences.
"""

__all__ = ['DecodeImage', 'ParseFixedLength']

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple, Dict, Hashable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.data.schema import Schema


class DecodeImage:
    """
    Decode serialized image bytes, and resize them to a fixed size.
    FIXME(ycho): Modify camera intrinsics according to the resize transform.
    """

    def __init__(self, size: Tuple[int, int], in_place: bool = True):
        self.size = size
        self.in_place = in_place
        self.resize = transforms.Resize(self.size)

    def __call__(self, inputs: Tuple[dict, dict]):
        if inputs is None:
            return None
        context, features = inputs

        # NOTE(ycho): Shallow copy but pedantically safer
        if not self.in_place:
            features = features.copy()

        # Replace `image/encoded` with `image`
        features['image'] = th.stack([
            self.resize(thio.decode_image(th.from_numpy(img_bytes)))
            for img_bytes in features['image/encoded']], dim=0)

        del features['image/encoded']

        # Dimension features are no longer valid.
        # We can also rewrite them, but this information is already
        # implicitly included in features["image"].
        del features['image/width']
        del features['image/height']

        return (context, features)


class ParseFixedLength:
    """
    Convert variable-length sequence to fixed-length representation.
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
        pad_keys: Tuple[str, ...] = (
            'object/translation', 'object/orientation', 'object/scale')

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(
            self, inputs: Tuple[Dict[str, th.Tensor],
                                Dict[str, th.Tensor]]):
        if inputs is None:
            return None
        context, features = inputs

        # Reject data with less than `seq_len` frames.
        # NOTE(ycho): Unsqueeze `count` which is a scalar.
        n = context['count'][0]
        if n < self.opts.seq_len:
            return None

        # Extract a slice of length `seq_len` from the sequence.
        max_stride = n // self.opts.seq_len
        stride = min(max_stride, self.opts.stride)
        strided_len = stride * self.opts.seq_len

        i0 = np.random.randint(0, n - strided_len + 1)
        i1 = i0 + strided_len

        # FIXME(ycho):
        # sequence_id is currently a bytes list instead of `string`.
        # This makes it somewhat clunky to collate, so we'll drop this field
        # for now.
        # TODO(ycho): Consider adding `stride` to `new_ctx`
        new_ctx = dict(count=self.opts.seq_len)
        new_feat = {k: v[i0:i1:stride] for (k, v) in features.items()}

        # Some property are variable-length, since they are dependent on `instance_num`.
        # Pad the sequences to account for this.
        for key in self.opts.pad_keys:
            num_inst = new_feat['instance_num']
            x = new_feat[key]

            # single reference instance
            ref = th.as_tensor(x[0])
            # Compute single-feature dimension
            # TODO(ycho): consider assert for ref.shape[0] % num_inst[0] == 0
            dim = int(ref.shape[0] // num_inst[0])

            x_pad = ref.new_empty(
                (self.opts.seq_len, dim * self.opts.max_num_inst))
            if np.all(num_inst == num_inst[0]):
                # TODO(ycho): ensure this still works if
                # `instance_num` > `max_num_inst`
                m = min(dim * self.opts.max_num_inst, ref.shape[0])
                x_pad[:, :m] = th.as_tensor(x)[:, :m]
            else:
                max_n = min(self.opts.max_num_inst, num_ints)
                for i, n in enumerate(max_n):
                    x_pad[i, :n * dim] = th.as_tensor(x[i])
            new_feat[key] = x_pad

        return (new_ctx, new_feat)
