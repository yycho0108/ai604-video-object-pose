#!/usr/bin/env python3

from typing import Dict, List, Tuple, Hashable
from scipy.optimize import linear_sum_assignment

import torch as th
from torchvision.transforms import Compose
from top.data.transforms import (
    DenseMapsMobilePose,
    Normalize,
    InstancePadding,
)

from top.model.layers import (
    KeypointLayer2D, HeatmapLayer2D, ConvUpsample,
    DisplacementLayer2D, HeatmapCoordinatesLayer)

from top.data.schema import Schema
from top.data.load import (DatasetSettings, get_loaders)

from top.run.app_util import update_settings
from top.run.torch_util import resolve_device


def associate_keypoints_euclidean(
        centers: th.Tensor,
        center_scores: th.Tensor,
        keypoints: th.Tensor,
        keypoint_scores: th.Tensor):
    """Associate keypoints to their respective objects. Work-around in the
    absence of offset outputs from the model (for now).

    centers: [..., N, 2] Set of object centers.
    keypoints: [..., M, 2] Set of keypoints.
    """

    O = centers[..., :, None, :]  # N12
    K = keypoints[..., None, :, :]  # 1M2

    #S_O = center_scores[..., :, None, :]
    #S_K = keypoint_scores[..., :, None, :]

    # ...,NM
    cost_matrix = th.norm(O - K, dim=-1)  # * (S_O * S_K)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    i, j = linear_sum_assignment(cost_matrix)
    return (i, j)


class GroundTruthDecoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord = HeatmapCoordinatesLayer(
            HeatmapCoordinatesLayer.Settings())

    def forward(self, inputs: Dict[Hashable, th.Tensor]):
        scores, indices = self.coord(inputs[Schema.HEATMAP])
        kpt_scores, kpt_indices = self.coord(inputs[Schema.KEYPOINT_HEATMAP])

        # associations
        i, j = associate_keypoints_euclidean(
            indices, scores,
            kpt_indices, kpt_scores)

        # BATCH_SIZE x NUM_CLASS X NUM_INSTANCES
        print(scores.shape)
        print(indices.shape)
        print(kpt_scores.shape)
        print(kpt_indices.shape)

        return None


def main():
    model = GroundTruthDecoder()
    device = resolve_device('cpu')

    transform = Compose([
        DenseMapsMobilePose(
            DenseMapsMobilePose.Settings(),
            th.device('cpu:0')),
        InstancePadding(InstancePadding.Settings())])

    _, test_loader = get_loaders(DatasetSettings(), device, 1,
                                 transform=transform)

    for data in test_loader:
        outputs = model(data)
        break


if __name__ == '__main__':
    main()
