#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
