#!/usr/bin/env python3

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from torchvision.transforms import Compose
from pytorch3d.transforms.transform3d import Transform3d

from top.run.torch_util import resolve_device
from top.run.app_util import update_settings


class TakeFirst:
    def __init__(self):
        pass

    def __call__(self, inputs):
        return {k: v[0] for (k, v) in inputs.items()}


class DrawKeypoints:
    """
    Draw keypoints (as inputs['points']) on an image as-is.
    Mostly intended for debugging.
    """

    @dataclass
    class Settings(Serializable):
        kernel_size: int = 5

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs):
        outputs = inputs
        image = inputs['image']

        # Points in UV-coordinates
        # consistent with the objectron format.
        h, w = image.shape[-2:]
        points_uv = th.as_tensor(inputs['points'])

        # NOTE(ycho): The Objectron dataset flipped their convention
        # so that the point is ordered in a minor-major axis order.
        points = points_uv * th.as_tensor([w, h, 1.0])
        out = th.zeros_like(inputs['image'])

        # TODO(ycho): Resolve ambiguous naming convention.
        if 'instance_num' in inputs:
            num_inst = inputs['instance_num']
        elif 'num_instances' in inputs:
            num_inst = inputs['num_instances']
        else:
            raise KeyError(F'Instance number not found : {inputs.keys()}')
        n = int(num_inst)

        r = self.opts.kernel_size // 2
        for i in range(n):
            for x, y, _ in points[i]:
                i0, i1 = int(y), int(x)
                out[..., i0 - r: i0 + r, i1 - r:i1 + r] = 255
        outputs['rendered_keypoints'] = out
        return outputs


class DenseMapsMobilePose:
    """
    Create dense heatmaps and displacement fields from
    projected object keypoints - in the style of `MobilePose`.
    """
    @dataclass
    class Settings(Serializable):
        # kernel_size: Tuple[int, int] = (5, 5)
        kernel_size: int = 5
        sigma: float = 1.0
        num_class: int = 10
        in_place: bool = True

    def __init__(self, opts: Settings, device: th.device):
        self.opts = opts
        self.device = device

        # Precompute relevant kernels.
        self.gauss_kernel = self._compute_gaussian_kernel(self.opts)
        self.displacement_kernel = self._compute_displacement_kernel(self.opts)

    @staticmethod
    def _compute_gaussian_kernel(opts: Settings, device: th.device):
        delta = th.arange(
            opts.kernel_size,
            device=device) - opts.kernel_size // 2
        delta = th.exp_(-th.square_(delta) / (2.0 * opts.sigma**2))
        delta /= delta.sum()
        return delta[None, :] * delta[:, None]

    @staticmethod
    def _compute_displacement_kernel(opts: Settings, device: th.device):
        delta = th.arange(
            opts.kernel_size,
            device=device) - opts.kernel_size // 2
        return th.stack(th.meshgrid(delta, delta))

    def __call__(self, inputs: dict):
        # FIXME(ycho): This code is probably incorrect
        # due to shallow copies on tensors.
        opts = self.opts
        if opts.in_place:
            outputs = inputs
        else:
            outputs = inputs.copy()

        # Parse inputs...
        image = inputs['image']  # (C,H,W)
        keypoints_2d_uv = inputs['points']  # (O, 9, 2|3)
        keypoints_2d = keypoints_2d_uv * image.shape[-2:]

        # TODO(ycho): Resolve ambiguous naming convention.
        if 'instance_num' in inputs:
            num_inst = inputs['instance_num']
        elif 'num_instances' in inputs:
            num_inst = inputs['num_instances']
        else:
            raise KeyError(F'Instance number not found : {inputs.keys()}')

        # FIXME(ycho): CANNOT deal with batch inputs after this line.
        n = int(num_inst)
        for i in range(n):
            center = keypoints_2d[i, 0, :2]

        # HEATMAPS for PER_OBJECT CENTERS
        center = keypoints_2d[..., 0, :2]  # (..., 2)
        shape = image.shape
        shape[-3] = 1
        heatmap = th.zeros(shape, dtype=th.float32, device=self.device)

        # KEYPOINTS DEFINED WRT SINGLE OBJECT - HOW?
        # Allocate heatmap.
        points = keypoints_2d[..., 1:, :2]

        shape[-3] = 1
        detection_map = th.zeros(
            shape, dtype=th.float32, device=self.device)

        inputs['displacement_map'] = displacement_map


class BoxHeatmap:

    def __init__(self, device: th.device):
        self.device = device
        vertices = list(itertools.product(
            *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))  # 8x3
        # NOTE(ycho): prepending
        vertices = np.insert(vertices, 0, [0, 0, 0], axis=0)
        # vertices = np.r_[[[0, 0, 0]], vertices]
        vertices = th.as_tensor(vertices, dtype=th.float32, device=self.device)
        colors = (0.5 + 0.5 * vertices)
        self.vertices = vertices
        self.colors = colors

    def __call__(self, inputs):
        # Parse inputs.
        h, w = inputs['image'].shape[-2:]
        R = inputs['object/orientation']
        T = inputs['object/translation']
        S = inputs['object/scale']
        P = inputs['camera/projection']

        # TODO(ycho): Resolve ambiguous naming convention.
        if 'instance_num' in inputs:
            num_inst = inputs['instance_num']
        elif 'num_instances' in inputs:
            num_inst = inputs['num_instances']
        else:
            raise KeyError(F'Instance number not found : {inputs.keys()}')

        heatmaps = []
        n = int(num_inst)
        for i in range(n):
            # NOTE(ycho): Taking an instance-specific slice of batched input.
            irxn = R[i * 9:(i + 1) * 9]
            itxn = T[i * 3:(i + 1) * 3]
            iscale = S[i * 3:(i + 1) * 3]

            # T_scale = np.diag(np.r_[iscale.cpu().numpy(), 1.0])
            T_scale = Transform3d().scale(
                iscale.reshape(1, 3)).get_matrix()[0].float()

            # BBOX3D transform
            T_box = th.eye(4)
            T_box[:3, :3] = irxn.reshape(3, 3)
            T_box[:3, -1] = itxn

            # Camera transforms
            T_p = P.reshape(4, 4).float()

            # Compose all transforms
            # NOTE(ycho): Looks like `camera/view` is not needed.
            # Perhaps it's been fused into
            # object/{translation,orientation}.
            T = T_p @ T_box @ T_scale

            # Apply transform.
            # NOTE(ycho): skipping division on the last axis here.
            v = self.vertices @ T[:3, :3].T + T[:3, -1]
            v[..., :-1] /= v[..., -1:]

            # NDC -> Screen coordinates
            v[..., 0] = (1 + v[..., 0]) * (0.5 * h)
            v[..., 1] = (1 + v[..., 1]) * (0.5 * w)
            y, x = v[..., 0], v[..., 1]

            # TODO(ycho): avoid unnecessary use of cv2 here
            heatmap = th.zeros_like(inputs['image'])
            for j, (px, py) in enumerate(zip(x, y)):
                if px < 0 or px >= w or py < 0 or py >= h:
                    continue
                # cv2.circle(heatmap, (int(px), int(py)), 16, (0, 0, 255), -1)
                heatmap[...,
                        int(py) - 1:int(py) + 1,
                        int(px) - 1:int(px) + 1] = 255.0 * self.colors[j][:,
                                                                          None,
                                                                          None]  # 255
            heatmaps.append(heatmap)

        heatmaps = th.stack(heatmaps, dim=0)  # CHW, where C == num instances
        inputs['keypoint_heatmap'] = heatmaps
        return inputs


def main():
    from top.data.objectron_dataset_detection import Objectron, SampleObjectron
    from top.data.colored_cube_dataset import ColoredCubeDataset

    dataset_cls = ColoredCubeDataset
    # dataset_cls = SampleObjectron

    opts = dataset_cls.Settings()
    opts = update_settings(opts)
    device = resolve_device('cpu:0')

    if dataset_cls is SampleObjectron:
        xfm = Compose([BoxHeatmap(device=device),
                       DrawKeypoints(DrawKeypoints.Settings())])
        dataset = dataset_cls(opts, transform=xfm)
    else:
        xfm = Compose([TakeFirst(), BoxHeatmap(device=device),
                       DrawKeypoints(DrawKeypoints.Settings())])
        dataset = dataset_cls(opts, device, transform=xfm)

    for data in dataset:
        save_image(data['image'] / 255.0, F'/tmp/img.png')
        save_image(data['rendered_keypoints'] / 255.0, F'/tmp/rkpts.png')
        for i, img in enumerate(data['keypoint_heatmap']):
            save_image(img / 255.0, F'/tmp/kpt-{i}.png')
        break


if __name__ == '__main__':
    main()
