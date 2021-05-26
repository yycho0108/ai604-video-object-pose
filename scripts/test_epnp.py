#!/usr/bin/env python3

"""Test script for testing the EPnP implementation from pytorch3d.

Mostly intended to verify the internal conventions used within the
library, and to check for the validity of the output in terms of
accuracy/precision.
"""

import numpy as np
import itertools
from typing import Dict, Hashable, Tuple
import torch as th
from pytorch3d.ops.perspective_n_points import efficient_pnp

from top.data.objectron_detection import ObjectronDetection
from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.schema import Schema


def get_cube_points() -> th.Tensor:
    points_3d = list(itertools.product(
        *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
    points_3d = np.insert(points_3d, 0, [0, 0, 0], axis=0)
    points_3d = th.as_tensor(points_3d, dtype=th.float32).reshape(-1, 3)
    return points_3d


def main():
    device = th.device('cpu:0')
    #dataset = ColoredCubeDataset(
    #    ColoredCubeDataset.Settings(
    #        batch_size=1),
    #    device,
    #    transform=None)
    dataset = ObjectronDetection(
        ObjectronDetection.Settings(), False)

    p0 = get_cube_points()  # 9,3
    for data in dataset:
        points_2d = data[Schema.KEYPOINT_2D][..., :2]  # O,9,2

        K = data[Schema.INTRINSIC_MATRIX].reshape(3, 3)  # 3,3
        P = data[Schema.PROJECTION].reshape(4, 4)

        print('P', P)

        # Restore NDC convention + scaling + account for intrinsic matrix
        # FIXME(ycho): Maybe a more principled unprojection would be needed
        # in case of nontrivial camer matrices.
        points_2d -= 0.5  # [0,1] -> [-0.5, 0.5]
        # print(points_2d)

        # -2.0/project
        # print(dataset.tan_half_fov)
        # print(P[(1,0),(0,1)])
        # points_2d *= -2.0 * dataset.tan_half_fov  # [x/z,y/z,1.0]
        # points_2d *= -2.0 / P[None, None, (1, 0), (0, 1)]
        # points_2d *= -2.0 / P[None, None, (0, 1), (0, 1)]
        points_2d *= 2.0 / P[None, None, (1, 0), (1, 0)]

        print('p2d')
        print(points_2d)
        print(data[Schema.SCALE].shape)

        points_2d = th.flip(points_2d, (-1,))
        points_3d = p0[None] * data[Schema.SCALE][:, None, :]
        solution = efficient_pnp(points_3d, points_2d)

        R_gt = (data[Schema.ORIENTATION].reshape(-1, 3, 3))
        T_gt = (data[Schema.TRANSLATION].reshape(-1, 3, 1))

        print('GT')
        print(R_gt.T)  # NOTE(ycho): transposed due to mismatch in convention
        print(T_gt)

        print('PNP')
        print(solution.R)
        print(solution.T)

        print('ERROR')
        print(solution.err_2d)
        print(solution.err_3d)
        break


if __name__ == '__main__':
    main()
