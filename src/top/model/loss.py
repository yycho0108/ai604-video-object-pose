#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from top.data.schema import Schema

from top.model.loss_util import FocalLoss


class ObjectHeatmapLoss(nn.Module):
    def __init__(self, key: Schema = Schema.HEATMAP):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.key = key

    def forward(
            self, output: Dict[str, th.Tensor],
            targets: Dict[str, th.Tensor]) -> float:

        # Extract relevant tensors from arguments.
        pred = output[self.key]
        target = targets[self.key]

        # FIXME(ycho): Hardcoded batch_size inference
        batch_size = target.shape[0]

        # NOTE(ycho): deprecated for now ...
        if False:
            diff = pred - target
            mask = th.ones_like(diff, dtype=th.bool)
            # Ignore padded labels after `num_instance`.
            #inums = target[Schema.INSTANCE_NUM]
            #for batch_i in range(batch_size):
            #    num_instance = inums[batch_i]
            #    mask[batch_i, num_instance:] = False

            diff[~mask] = 0.0
            numer = th.sum(th.square(diff))
            denom = th.sum(mask)
            return numer / denom

        out = self.focal_loss(pred, target)
        return out


class KeypointDisplacementLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Dict[str, th.Tensor],
                target: Dict[str, th.Tensor]) -> float:
        pred = output[Schema.DISPLACEMENT_MAP]

        mask = th.isfinite(target[Schema.DISPLACEMENT_MAP])

        diff = pred - target[Schema.DISPLACEMENT_MAP]
        # NOTE(ycho): during inference, this mask is approximated
        # by the heatmaps.
        # NOTE(ycho): We MUST this use form since inf * 0 = NaN.
        diff[~mask] = 0.0
        # NOTE(ycho): Using abs here, which results in L1 loss.
        numer = th.sum(th.abs(diff))
        denom = th.sum(mask)
        return numer / denom


class KeypointHeatmapLoss(nn.Module):
    def __init__(self):
        return NotImplemented

    def forward(
            self, output: Dict[str, th.Tensor],
            target: Dict[str, th.Tensor]) -> float:
        return NotImplemented


class KeypointCrossEntropyLoss(nn.Module):
    """Given a keypoint heatmap of logits, compute the loss against integer-
    valued target map of keypoints.

    TODO(ycho): Perhaps not the best idea, especially if the number of keypoints are very sparse.
    """

    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output: th.Tensor, target: th.Tensor) -> float:
        return self.loss(output, target)
