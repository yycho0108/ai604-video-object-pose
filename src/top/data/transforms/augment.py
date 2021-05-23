#!/usr/bin/env python3
"""Set of transforms related to data augmentation."""

__all__ = ['PhotometricAugment', 'GeometricAugment']

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple, Dict, Hashable
import itertools
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as thio
from torchvision import transforms

from top.data.schema import Schema


class PhotometricAugment:
    """Apply photometric augmentation."""
    @dataclass
    class Settings(Serializable):
        brightness: Tuple[float, float] = (0.6, 1.6)
        contrast: Tuple[float, float] = (0.6, 1.6)
        saturation: Tuple[float, float] = (0.6, 1.6)
        hue: Tuple[float, float] = (-0.2, 0.2)
        key_in: Schema = Schema.IMAGE
        key_out: Schema = Schema.IMAGE

    def __init__(self, opts: Settings, in_place: bool = True):
        self.opts = opts
        self.in_place = in_place
        self.xfm = transforms.ColorJitter(
            brightness=opts.brightness,
            contrast=opts.contrast,
            saturation=opts.saturation,
            hue=opts.hue)

    def __call__(self, inputs: Dict[Hashable, th.Tensor]):
        if inputs is None:
            return None

        # NOTE(ycho): Shallow copy but pedantically safer
        if self.in_place:
            outputs = inputs
        else:
            outputs = inputs.copy()

        augmented_image = self.xfm(outputs[self.opts.key_in])
        outputs[self.opts.key_out] = augmented_image
        return inputs


class GeometricAugment:

    @dataclass
    class Settings(Serializable):
        p_flip_lr: float = 0.5

    def __init__(self, opts: Settings, in_place: bool = False):
        self.opts = opts
        self.in_place = in_place
        self.flip = transforms.RandomHorizontalFlip(self.opts.p_flip_lr)

        # Figure out keypoint permutation.
        # FIXME(ycho): Probably not the brightest idea to keep this hardcoded.
        v_in = list(itertools.product(
            *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
        v_in = np.insert(v_in, 0, [0, 0, 0], axis=0)

        v_out = list(itertools.product(
            *zip([0.5, -0.5, -0.5], [-0.5, 0.5, 0.5])))
        v_out = np.insert(v_out, 0, [0, 0, 0], axis=0)

        delta = np.square(v_out[None, :, ...] - v_in[:, None, ...])  # I,O,...
        delta = delta.reshape(delta.shape[0], delta.shape[1], -1)
        delta = np.einsum('iok, iok -> io', delta, delta)  # I,O
        self.perm = np.argmin(delta, axis=0)  # best-match output per input

    def __call__(self, inputs: Dict[Hashable, th.Tensor]):
        if self.in_place:
            outputs = inputs
        else:
            print('copy')
            outputs = inputs.copy()

        # Conditional No-op based on random sample probability
        if th.rand(1) >= self.opts.p_flip_lr:
            return output

        # Flip for all image-like inputs in the schema.
        for key in [Schema.IMAGE, Schema.CROPPED_IMAGE,
                    Schema.KEYPOINT_MAP, Schema.HEATMAP,
                    Schema.HEATMAP_LOGITS, Schema.DISPLACEMENT_MAP,
                    Schema.KEYPOINT_HEATMAP]:
            if key not in inputs:
                continue

            # NOTE(ycho): Taken from `transforms.RandomHorizontalFlip`.
            outputs[key] = transforms.functional.hflip(inputs[key])

        # NOTE(ycho): `keypoint` supplied in X-Y order
        if Schema.KEYPOINT_2D in outputs:
            # Clone to prevent overwriting input ...
            outputs[Schema.KEYPOINT_2D] = (
                inputs[Schema.KEYPOINT_2D].clone()
            )

            # Transform coordinates to account for flipping.
            outputs[Schema.KEYPOINT_2D][..., 0] = (
                1.0 - inputs[Schema.KEYPOINT_2D][..., 0]
            )

            # Apply permutations to preserve "correct" keypoint order.
            outputs[Schema.KEYPOINT_2D] = outputs[Schema.KEYPOINT_2D][..., self.perm, :]

        # Translation vector
        if Schema.TRANSLATION in outputs:
            outputs[Schema.TRANSLATION][..., 1] *= -1

        # Rotation matrix
        if Schema.ORIENTATION in outputs:
            orientation = outputs[Schema.ORIENTATION]
            s = list(orientation.shape)
            if orientation.shape[-1] == 9:
                s_new = s[:-1] + [3, 3]
                orientation = orientation.reshape(s_new)

            # Apply reflection across "Y"-axis
            # TODO(ycho): either this or [..., :, 1]
            # TODO(ycho): might be a different axis. jeez...
            orientation[..., 1, :] *= -1

            # Restore previous shape.
            orientation = orientation.reshape(s)

        # TODO(ycho): Currently unhandled fields:
        # KEYPOINT_3D = "point_3d"
        # QUATERNION = "quaternion"
        # PROJECTION = "camera/projection"
        # INTRINSIC_MATRIX = "camera/intrinsics"
        # BOX_2D = "box_2d"

        return outputs


def main():
    from top.data.objectron_sequence import (
        ObjectronSequence,
        SampleObjectron,
        DecodeImage,
        ParseFixedLength,
        _skip_none)

    opts = SampleObjectron.Settings()
    xfm = transforms.Compose([
        DecodeImage(size=(480, 640)),
        ParseFixedLength(ParseFixedLength.Settings(seq_len=4)),
        PhotometricAugment(PhotometricAugment.Settings())
    ])
    dataset = SampleObjectron(opts, xfm)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=_skip_none)

    for data in loader:
        context, features = data
        print(features['image'].shape)
        break


if __name__ == '__main__':
    main()
