#!/usr/bin/env python3

import torch as th
from pytorch3d.ops.perspective_n_points import efficient_pnp

from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.schema import Schema


def main():
    device = th.device('cpu:0')
    dataset = ColoredCubeDataset(ColoredCubeDataset.Settings(), device)

    for data in dataset:
        points_2d = data[Schema.KEYPOINT_2D][..., :2]  # 1,9,2

        # Restore NDC convention + permutation
        # NOTE(ycho): This is NOT consistent with
        points_2d -= 0.5  # [0,1] -> [-0.5, 0.5]
        points_2d *= -2.0 * dataset.tan_half_fov

        points_3d = dataset.cloud.points_padded()  # 1,9,3
        solution = efficient_pnp(points_3d, points_2d)

        R_gt = (data[Schema.ORIENTATION].reshape(3, 3))
        T_gt = (data[Schema.TRANSLATION].reshape(3, 1))

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
