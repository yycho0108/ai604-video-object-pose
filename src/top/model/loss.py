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


class KeypointScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, output: Dict[str, th.Tensor],
                target: Dict[str, th.Tensor]) -> float:

        # We extract the center index from the input.
        # TODO(ycho): Consider adding a data-processing `transform` instead.
        # H, W = inputs[Schema.IMAGE].shape[-2:]
        h, w = output[Schema.SCALE_MAP].shape[-2:]

        # FIXME(ycho): `visibility` mask should ultimately account for
        # out-of-range behavior ... (fingers crossed)
        visibility = target[Schema.VISIBILITY].to(dtype=th.bool)[..., 0]
        keypoints_2d_uv = target[Schema.KEYPOINT_2D]
        center_uv = keypoints_2d_uv[..., 0, :2]
        scale_xy = th.as_tensor(
            [w, h], dtype=th.int32, device=center_uv.device)
        center_xy = th.round(center_uv * scale_xy).to(dtype=th.int64)
        # NOTE(ycho): Explicitly writing out (i,j) since the `Objectron`
        # keypoint order is # unconventional.
        j = center_xy[..., 0]  # (B, O)
        i = center_xy[..., 1]  # (B, O)
        flat_index = (i * w + j)

        in_bound = th.all(th.logical_and(center_xy >= 0,
                                         center_xy < scale_xy), dim=-1)
        visibility = th.logical_and(visibility, in_bound)

        # NOTE(ycho): Overwrite invalid(invisible) index with 0
        # in order to prevent errors during gather().
        # Here, we explicitly check for not only the dataset visibility,
        # but also the validity of the resulting indexes within image bounds as
        # well.
        flat_index[~visibility] = 0

        shape = output[Schema.SCALE_MAP].shape

        X = output[Schema.SCALE_MAP].reshape(shape[:-2] + (-1,))
        I = flat_index[:, None]
        I = I.expand(*((-1, shape[1]) + tuple(flat_index.shape[1:])))
        V = visibility

        # NOTE(ycho): permute required for (B,3,O) -> (B,O,3)
        scale_output = X.gather(-1, I).permute(0, 2, 1)
        scale_target = target[Schema.SCALE]

        return self.loss(scale_output[V], scale_target[V])
