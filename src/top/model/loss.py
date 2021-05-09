#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from top.data.schema import Schema


class ObjectHeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
            self, output: Dict[str, th.Tensor],
            target: Dict[str, th.Tensor]) -> float:
        if Schema.HEATMAP in output:
            pred = output[Schema.HEATMAP]
        elif Schema.HEATMAP_LOGITS in output:
            pred = th.sigmoid(output[Schema.HEATMAP_LOGITS])
        else:
            raise KeyError(
                F'Could not find heatmap key in {list(output.keys())}')
        return self.loss(pred, target[Schema.HEATMAP])


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
        diff[~mask] = 0.0
        # NOTE(ycho): Using abs here, which results in L1 loss.
        numer = th.sum(th.abs(diff))
        denom = th.sum(mask)
        return numer / denom


class KeypointCrossEntropyLoss(nn.Module):
    """
    Given a keypoint heatmap of logits,
    compute the loss against integer-valued target map of keypoints.
    TODO(ycho): Perhaps not the best idea, especially if the number of keypoints are very sparse.
    """

    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output: th.Tensor, target: th.Tensor) -> float:
        return self.loss(output, target)
