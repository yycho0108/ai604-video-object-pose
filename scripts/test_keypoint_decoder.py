#!/usr/bin/env python3

import numpy as np
import itertools
from typing import Dict, List, Tuple, Hashable
from scipy.optimize import linear_sum_assignment

import torch as th
from torchvision.transforms import Compose
from pytorch3d.ops.perspective_n_points import efficient_pnp
from pytorch3d.transforms import so3_relative_angle

from top.model.layers import (
    KeypointLayer2D, HeatmapLayer2D, ConvUpsample,
    DisplacementLayer2D, HeatmapCoordinatesLayer)

from top.data.schema import Schema
from top.data.load import (DatasetSettings, get_loaders)
from top.data.transforms import (
    DenseMapsMobilePose,
    Normalize,
    InstancePadding,
    BoxPoints2D
)

from top.run.app_util import update_settings
from top.run.torch_util import resolve_device


def associate_keypoints_euclidean(
        centers: th.Tensor,
        center_scores: th.Tensor,
        keypoints: th.Tensor,
        keypoint_scores: th.Tensor
):
    """Associate keypoints to their respective objects. Work-around in the
    absence of offset outputs from the model (for now).

    centers: [..., N, 2] Set of object centers.
    keypoints: [..., M, 2] Set of keypoints.
    """

    O = centers[..., :, None, :]  # N12
    K = keypoints[..., None, :, :]  # 1M2

    O_S = center_scores[..., :, None]
    K_S = keypoint_scores[..., None, :]

    # ...,NM
    cost_matrix = th.norm((O - K).float(), dim=-1) * O_S * K_S
    cost_matrix = cost_matrix.detach().cpu().numpy()

    i, j = linear_sum_assignment(cost_matrix)
    return (i, j)


def get_cube_points() -> th.Tensor:
    """Get cube points, sorted in descending order by axes and coordinates."""
    # TODO(ycho): remove duplicated code everywhere regarding these vertex
    # definitions.
    points_3d = list(itertools.product(
        *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
    points_3d = np.insert(points_3d, 0, [0, 0, 0], axis=0)
    points_3d = th.as_tensor(points_3d, dtype=th.float32).reshape(-1, 3)
    return points_3d


def compute_transforms(
        points_2d: th.Tensor,
        box_scale: th.Tensor,
        projection_matrix: th.Tensor,
        point_ids: th.Tensor = None):
    # Compute target 3d bounding box with scale applied.
    cube_points = get_cube_points().to(device=points_2d.device)

    # Extract relevant fields from the dataset.
    # NOTE(ycho): Cloning here, to avoid overwriting input data.
    # print(cube_points.shape) # 9,3 --> 1,9,3
    # print(box_scale.shape)   # ...,3
    points_2d = points_2d.clone()
    # points_3d = cube_points[None, ...] * box_scale[:, None, :]
    points_3d = cube_points * box_scale[None, :]
    P = projection_matrix.reshape(4, 4)

    # Preprocess `points_2d` to comform to `efficient_pnp` convention.
    # points_2d is in X-Y order (i.e. minor-major), normalized to range (0.0,
    # 1.0).
    points_2d -= 0.5
    # points_2d *= 2.0 / P[None, None, (0, 1), (0, 1)]
    points_2d *= 2.0 / P[None, (0, 1), (0, 1)]

    # Compute PNP solution ...
    try:
        # NOTE(ycho): Only apply PnP on the points that were found.
        if point_ids is not None:
            points_3d = points_3d[point_ids, ...]
        solution = efficient_pnp(points_3d[None], points_2d[None],
                                 skip_quadratic_eq=True)
    except RuntimeError as e:
        print('Encountered error during PnP : {}'.format(e))
        return None

    R, T = solution.R[0], solution.T[0]

    # NOTE(ycho): **IMPORTANT**: this is the post-processing step
    # that accounts for the difference in the conventions used
    # within the `objectron` dataset vs. pytorch3d.

    # NOTE(ycho): In the PNP solution convention,
    # y[i] = Proj(x[i] R[i] + T[i])
    # In the objectron convention,
    # y[i] = Proj(R[i] x[i] + T[i])
    R = th.transpose(R, -2, -1)

    # NOTE(ycho): additional correction to account for coordinate flipping.
    DR = th.as_tensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ], dtype=th.float32, device=R.device)
    R = th.einsum('ab,...bc->...ac', DR, R)
    T = th.einsum('ab,...b->...a', DR, T)

    return (R, T)


def compute_deltas(R_src, T_src, R_dst, T_dst):
    err_R = so3_relative_angle(R_src, R_dst)
    # NOTE(ycho): Account for occasional (...,3,1) inputs.
    Tdim = (-2, -1) if (T_src.shape[-1] == 1) else -1
    err_T = th.norm(T_src - T_dst, dim=Tdim)
    return (err_R, err_T)


class GroundTruthDecoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord = HeatmapCoordinatesLayer(
            HeatmapCoordinatesLayer.Settings())

    def forward(self, inputs: Dict[Hashable, th.Tensor]):
        scores, indices = self.coord(inputs[Schema.HEATMAP])
        kpt_scores, kpt_indices = self.coord(inputs[Schema.KEYPOINT_HEATMAP])
        h, w = inputs[Schema.KEYPOINT_HEATMAP].shape[-2:]

        # Let us ignore class assignments (for now), and
        # treat all instances as instances irrespective of class.
        # {batch_size, num_class, num_objects}
        indices = indices.reshape(
            (indices.shape[0], indices.shape[1] * indices.shape[2]) +
            (indices.shape[3:]))
        scores = scores.reshape(
            (scores.shape[0], scores.shape[1] * scores.shape[2]) +
            (scores.shape[3:]))

        # Convert scores (high is better) to costs (low is better)
        # TODO(ycho): Improve this ad-hoc cost conversion.
        costs = th.exp(-scores)
        kpt_costs = th.exp(-kpt_scores)

        # Generate associations between keypoints and object(center) predictions.
        # NOTE(ycho): workaround until we incorporate learned offsets estimation.
        # TODO(ycho): Batched LAP would REALLY be preferable to this loop.
        batch_obj_kpts = []
        for i_batch in range(kpt_indices.shape[0]):
            kpt_coords = [[] for _ in range(indices.shape[1])]
            kpt_ids = [[] for _ in range(indices.shape[1])]
            for i_kpt in range(kpt_indices.shape[1]):
                # Index into current quantities of interest.
                # Basically, we're trying to figure out,
                # per-keypoint, which object would be ideal
                # for placement(s).
                O = indices[i_batch]
                O_S = costs[i_batch]
                K = kpt_indices[i_batch, i_kpt]
                K_S = kpt_costs[i_batch, i_kpt]
                ii, jj = associate_keypoints_euclidean(O, O_S, K, K_S)

                # Aggregate keypoint assignments over best-matched objects.
                # FIXME(ycho): I feel like we can do this more efficiently ...
                for i, j in zip(ii, jj):
                    # Object acceptance criterion ...
                    # FIXME(ycho): ad-hoc value
                    if O_S[i] > np.exp(-0.25):
                        continue
                    # Keypoint acceptance criterion ...
                    # FIXME(ycho): ad-hoc value
                    if K_S[j] > np.exp(-0.25):
                        continue
                    kpt_coords[i].append(K[j])
                    # print('should really be an integer or something ...')
                    # print(K[j])
                    kpt_ids[i].append(i_kpt)

            # Gather all non-trivial objects with N>0 keypoint assignments.
            # TODO(ycho): N>4 would probably be required, now that I think
            # about it, rather than the current criterion of N>0.
            out_batch = []
            for i_obj, k in enumerate(kpt_coords):
                if not k:
                    continue
                entry = (i_obj, kpt_coords[i_obj], kpt_ids[i_obj])
                out_batch.append(entry)
            batch_obj_kpts.append(out_batch)

        batch_obj_boxes = []
        for i_batch, entries in enumerate(batch_obj_kpts):
            se3s = []
            for entry in entries:
                i_obj, kpt_coords, kpt_ids = entry
                obj_coords = indices[i_batch][i_obj]

                # FIXME(ycho): In the actual network output, we would have
                # corresponding `scale` inferences that correspond to `i_obj`.
                # However, since we didn't generate SCALE_MAP (which we can...),
                # let us look it up by *closest-match heuristic* for now.
                target_centers = inputs[Schema.KEYPOINT_2D][i_batch, ..., 0, :2]
                j = th.argmin(th.norm(target_centers - obj_coords, dim=-1))
                scale = inputs[Schema.SCALE][i_batch][j]

                # Actual indices
                kpt_coords = th.stack(kpt_coords, axis=0).reshape(-1, 2)

                # Convert to normalized coordinates in range (0, 1).
                points_2d = kpt_coords / th.as_tensor(
                    [h, w], device=kpt_coords.device)
                points_2d = th.flip(points_2d, dims=(-1,))  # i,j --> x,y

                P = inputs[Schema.PROJECTION][i_batch].reshape(4, 4)
                R_gt = inputs[Schema.ORIENTATION][i_batch, j].reshape(3, 3)
                T_gt = inputs[Schema.TRANSLATION][i_batch, j].reshape(3)

                sol = compute_transforms(
                    points_2d,
                    scale,
                    P,
                    th.as_tensor(
                        kpt_ids, dtype=th.int64, device=points_2d.device))

                if sol is not None:
                    R, T = sol
                    se3s.append((P, T, R, scale))

                #for p2 in [points_2d,
                #           inputs[Schema.KEYPOINT_2D]
                #           [i_batch, j, kpt_ids, : 2]]:
                #    sol = compute_transforms(
                #        # points_2d,
                #        # inputs[Schema.KEYPOINT_2D][i_batch, j, kpt_ids, :2],
                #        p2,
                #        scale,
                #        inputs[Schema.PROJECTION][i_batch].reshape(4, 4),
                #        th.as_tensor(
                #            kpt_ids, dtype=th.int64, device=points_2d.device))

                #    if sol is not None:
                #        R, T = sol
                #        print(R.shape, T.shape)
                #        print(R_gt.shape, T_gt.shape)
                #        print('solution ...')
                #        print(R)
                #        print(T)

                #        print('target ...')
                #        print(inputs[Schema.VISIBILITY][i_batch, j])
                #        print(R_gt)
                #        print(T_gt)

                #        dR, dT = compute_deltas(R, T, R_gt[None], T_gt[None])
                #        print(
                #            'err_R = {} ({} deg.)'.format(
                #                dR, th.rad2deg(dR)))
                #        print('err_T = {}'.format(dT))

            box_points = get_cube_points()  # 9,3
            obj_boxes = []
            for se3 in se3s:
                P, T, R, S = se3
                # transform to camera coordinates.
                box_3d = th.einsum('ab,kb->ka', R, box_points * S) + T
                # project.
                box_2d = th.einsum('ab,kb->ka', P[:3, :3], box_3d)
                box_2d /= box_2d[..., 2:]
                # (-1,1) -> (0,1)
                box_2d[..., :2] = 0.5 + 0.5 * box_2d[..., :2]
                # ij -> xy
                box_2d[..., :2] = th.flip(box_2d[..., :2], dims=(-1,))

                # Take off the last dimension
                box_2d = box_2d[..., :2]

                # Convert to Numpy (for now... at least)
                box_2d = box_2d.detach().cpu().numpy()
                box_3d = box_3d.detach().cpu().numpy()

                obj_boxes.append((box_2d, box_3d))
            batch_obj_boxes.append(obj_boxes)
        return batch_obj_boxes


def main():
    model = GroundTruthDecoder()
    device = resolve_device('cpu')

    transform = Compose([
        DenseMapsMobilePose(
            DenseMapsMobilePose.Settings(),
            th.device('cpu:0')),
        BoxPoints2D(device, key_out='p2d-debug'),
        InstancePadding(InstancePadding.Settings()),
    ])

    _, test_loader = get_loaders(DatasetSettings(), device, 1,
                                 transform=transform)

    for data in test_loader:
        outputs = model(data)
        break


if __name__ == '__main__':
    main()
