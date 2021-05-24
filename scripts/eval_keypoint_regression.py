#!/usr/bin/env python3

import sys
import logging
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import Dict, Any
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
from typing import Tuple
import itertools
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch as th
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

from pytorch3d.ops.perspective_n_points import efficient_pnp

from top.train.saver import Saver
from top.train.trainer import Trainer
from top.train.event.hub import Hub
from top.train.event.topics import Topic
from top.train.event.helpers import (Collect, Periodic, Evaluator)

from top.model.keypoint import KeypointNetwork2D
from top.model.loss import ObjectHeatmapLoss, KeypointDisplacementLoss

from top.data.transforms import (
    DenseMapsMobilePose,
    Normalize,
    InstancePadding,
)
from top.data.transforms.visualize import (
    DrawKeypointMap
)
from top.data.schema import Schema
from top.data.load import (DatasetSettings, get_loaders)

from top.run.app_util import update_settings
from top.run.path_util import RunPath, get_latest_file
from top.run.torch_util import resolve_device

# FIXME(ycho): OK, probably not the brightest idea
# from .keypoint_regression import AppSettings


@dataclass
class AppSettings(Serializable):
    model: KeypointNetwork2D.Settings = KeypointNetwork2D.Settings()

    # Dataset selection options.
    dataset: DatasetSettings = DatasetSettings()

    # NOTE(ycho): root run path is set to tmp dir y default.
    path: RunPath.Settings = RunPath.Settings(root='/tmp/ai604-kpt')
    batch_size: int = 8
    device: str = ''

    padding: InstancePadding.Settings = InstancePadding.Settings()
    maps: DenseMapsMobilePose.Settings = DenseMapsMobilePose.Settings()


def spatial_nms(x, kernel_size: int = 3):
    # alternatively, kernel_size ~ 3 * sigma
    pad = (kernel_size - 1) // 2
    local_max = th.nn.functional.max_pool2d(
        x, (kernel_size, kernel_size), stride=1, padding=pad)
    peak_mask = (local_max == x).float()
    return x * peak_mask


def top_k(scores, k=40):
    batch_size, types, h, w = scores.size()
    scores, indices = th.topk(scores.view(batch_size, types, -1), k)
    indices = indices % (h * w)
    i = (indices / w)
    j = (indices % w)
    return (scores, indices, i, j)


def decode_kpt_heatmap(heatmap: th.Tensor,
                       max_num_instance: int = 4):
    # TODO(ycho): Incorporate:
    # - cropping by `scale` <?
    # - pruning/gathering by `offset` from heatmap center(s)
    batch, cat, height, width = heatmap.size()
    num_points = heatmap.shape[1]
    heatmap = spatial_nms(heatmap)
    scores, inds, i, j = top_k(heatmap, k=max_num_instance)
    # scores = (32,9,4)
    # (i,j) = (32,2,9,4)
    return scores, th.stack([i, j], axis=-1)


def associate_keypoints_euclidean(
        centers: th.Tensor,
        keypoints: th.Tensor):
    """Associate keypoints to their respective objects. Work-around in the
    absence of offset outputs from the model (for now).

    centers: [..., N, 2] Set of object centers.
    keypoints: [..., M, 2] Set of keypoints.
    """
    O = centers[..., :, None, :]  # N12
    K = keypoints[..., None, :, :]  # 1M2

    # ...,NM
    # TODO(ycho): Consider incorporating object confidences to cost matrix
    # i.e. (O - K) * score(O) * score(K)
    cost_matrix = th.norm(O - K, dim=-1)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    i, j = linear_sum_assignment(cost_matrix)


def compute_pose_epnp(intrinsic_matrix: th.Tensor,
                      scale: Tuple[float, float, float],
                      points_2d: th.Tensor  # ,kpt_index:th.Tensor
                      ):
    intrinsic_matrix = intrinsic_matrix.reshape(4, 4)
    # Expected 3D bounding box vertices
    points_3d = list(itertools.product(
        *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
    points_3d = np.insert(points_3d, 0, [0, 0, 0], axis=0)
    points_3d = th.as_tensor(points_3d, dtype=th.float32)
    points_3d = (points_3d[:, None] * th.as_tensor(scale,
                                                   device=points_3d.device))

    # Restore NDC convention + scaling + account for intrinsic matrix
    # FIXME(ycho): Maybe a more principled unprojection would be needed
    # in case of nontrivial camer matrices.
    points_2d = points_2d.clone()
    points_2d -= 0.5  # [0,1] -> [-0.5, 0.5]
    points_2d *= -2.0 / intrinsic_matrix[(0, 1), (1, 0)]

    # NOTE(ycho): PNP occasionally (actually quite often)? fails.
    try:
        solution = efficient_pnp(
            points_3d.transpose(
                0, 1).to(
                points_2d.device), points_2d.transpose(
                0, 1))
        return (solution.R, solution.T)
    except RuntimeError:
        return None


def main():
    # logging.basicConfig(level=logging.DEBUG)

    # Initial parsing looking for `RunPath` ...
    opts = AppSettings()
    opts = update_settings(opts)
    if not opts.path.key:
        raise ValueError('opts.path.key required for evaluation (For now)')
    path = RunPath(opts.path)

    # Re-parse full args with `base_opts` as default instead
    # TODO(ycho): Verify if this works.
    base_opts = update_settings(
        opts, argv=['--config_file', str(path.dir / 'opts.yaml')])
    opts = update_settings(base_opts)

    # Instantiation ...
    device = resolve_device(opts.device)
    model = KeypointNetwork2D(opts.model).to(device)

    # Load checkpoint.
    ckpt_file = get_latest_file(path.ckpt)
    print('ckpt = {}'.format(ckpt_file))
    Saver(model, None).load(ckpt_file)

    # NOTE(ycho): Forcing data loading on the CPU.
    # TODO(ycho): Consider scripted compositions?
    transform = Compose([
        DenseMapsMobilePose(opts.maps, th.device('cpu:0')),
        Normalize(Normalize.Settings()),
        InstancePadding(opts.padding)
    ])
    _, test_loader = get_loaders(opts.dataset,
                                 device=th.device('cpu:0'),
                                 batch_size=opts.batch_size,
                                 transform=transform)

    model.eval()
    for data in test_loader:
        # Now that we're here, convert all inputs to the device.
        data = {k: (v.to(device) if isinstance(v, th.Tensor) else v)
                for (k, v) in data.items()}
        image = data[Schema.IMAGE]
        image_scale = th.as_tensor(image.shape[-2:])  # (h,w) order
        print('# instances = {}'.format(data[Schema.INSTANCE_NUM]))
        with th.no_grad():
            outputs = model(image)

            heatmap = outputs[Schema.HEATMAP]
            kpt_heatmap = outputs[Schema.KEYPOINT_HEATMAP]

            # FIXME(ycho): hardcoded obj==1 assumption
            scores, indices = decode_kpt_heatmap(
                kpt_heatmap, max_num_instance=4)

            # hmm...
            upsample_ratio = th.as_tensor(
                image_scale / th.as_tensor(heatmap.shape[-2:]),
                device=indices.device)
            upsample_ratio = upsample_ratio[None, None, None, :]

        scaled_indices = indices * upsample_ratio

        # Visualize inferred keypoints ...
        if False:
            # FIXME(ycho): Pedantically incorrect!!
            heatmap_vis = DrawKeypointMap(
                DrawKeypointMap.Settings(
                    as_displacement=False))(heatmap)
            kpt_heatmap_vis = DrawKeypointMap(
                DrawKeypointMap.Settings(
                    as_displacement=False))(kpt_heatmap)

            fig, ax = plt.subplots(3, 1)
            hv_cpu = heatmap_vis[0].detach().cpu().numpy().transpose(1, 2, 0)
            khv_cpu = kpt_heatmap_vis[0].detach(
            ).cpu().numpy().transpose(1, 2, 0)
            img_cpu = th.clip(
                0.5 + (image[0] * 0.25),
                0.0, 1.0).detach().cpu().numpy().transpose(
                1, 2, 0)
            ax[0].imshow(hv_cpu)
            ax[1].imshow(khv_cpu / khv_cpu.max())
            ax[2].imshow(img_cpu)
            plt.show()

        # scores = (32,9,4)
        # (i,j)  = (32,2,9,4)
        for i_batch in range(scores.shape[0]):
            # GROUND_TRUTH
            kpt_in = data[Schema.KEYPOINT_2D][i_batch, ..., :2]
            kpt_in = kpt_in * image_scale.to(kpt_in.device)
            # X-Y order (J-I order)
            # print(kpt_in)

            # print(scaled_indices[i_batch])  # Y-X order (I-J order)
            print('scale.shape')  # 32,4,3
            print(data[Schema.SCALE].shape)
            sol = compute_pose_epnp(
                data[Schema.PROJECTION][i_batch],
                # not estimating scale info for now ...,
                data[Schema.SCALE][i_batch],
                th.flip(scaled_indices[i_batch], dims=(-1,)
                        ) / image_scale.to(scaled_indices.device)
            )
            if sol is None:
                continue
            R, T = sol
            print(R, data[Schema.ORIENTATION][i_batch])
            print(T, data[Schema.TRANSLATION][i_batch])
            break

        np.save(F'/tmp/heatmap.npy', heatmap.cpu().numpy())
        np.save(F'/tmp/kpt_heatmap.npy', kpt_heatmap.cpu().numpy())
        break


if __name__ == '__main__':
    main()
