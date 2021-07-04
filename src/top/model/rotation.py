#!/usr/bin/env python3
"""Layer(s) for 3D rotation parameterization.

Reference:

https://github.com/papagina/RotationContinuity

@inproceedings{Zhou_2019_CVPR,
title={On the Continuity of Rotation Representations in Neural Networks},
author={Zhou, Yi and Barnes, Connelly and Jingwan, Lu and Jimei, Yang and Hao, Li},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month={June},
year={2019}
}
"""

from typing import Optional

import torch as th
import torch.nn as nn
from dataclasses import dataclass
from simple_parsing import Serializable


def _normalize(x: th.Tensor, dim: int):
    """Normalize a tensor in-place.

    Also deal with degeneracies (e.g. zero-tensor). ||x||_2 == 1.
    """
    mag = th.linalg.norm(x, dim=dim, keepdim=True)
    # mag = th.max(mag, th.as_tensor(1e-8).to(mag.device))
    return th.div(x, mag)


class RotationOrtho6D(nn.Module):
    """Convert 6DoF vector into 9DoF(3x3) rotation matrix."""

    @dataclass
    class Settings(Serializable):
        c_dim: int = 1
        normalize: bool = True

    def __init__(self, cfg: Settings):
        super().__init__()
        # FIXME(ycho): Might interfere with torch.jit scripting
        self.cfg = cfg

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        cfg = self.cfg
        if inputs.shape[cfg.c_dim] != 6:
            msg = F'Invalid input shape : {inputs.shape}[{cfg.c_dim}]; must be 6DoF!'
            raise ValueError(msg)

        # Populate output ...
        x, y = th.chunk(inputs, 2, dim=cfg.c_dim)
        if cfg.normalize:
            x = _normalize(x, dim=cfg.c_dim)
        z = th.cross(x, y, dim=cfg.c_dim)
        z = _normalize(z, dim=cfg.c_dim)
        y = th.cross(z, x, dim=cfg.c_dim)

        # NOTE(ycho): As a result of this stack, the tensor becomes
        # (..., i=3, ...) --> (..., i=3, j=3, ...)
        stack_dim = cfg.c_dim + 1 if cfg.c_dim >= 0 else cfg.c_dim
        outputs = th.stack([x, y, z], dim=stack_dim)
        return outputs


class GeodesicLoss(nn.Module):
    """"""

    def __init__(self, cos: bool = True):
        super().__init__()
        self.cos = cos

    def forward(self, x: th.Tensor, y: th.Tensor) -> float:
        """
        Expected format = (...,3,3) where x'=Rx.
        """
        if self.cos:
            # Map to 0~1 range, -0.5*cos(Y)+0.5
            theta = (th.einsum('...ij,...ij->...', x, y)
                     .mul_(-0.25).add_(0.75))
        else:
            theta = (th.einsum('...ij,...ij->...', x, y)
                     .sub_(1.0).mul_(0.5)
                     .clamp_(-1.0, 1.0)
                     .acos_())
        error = theta.mean()
        # NOTE(ycho): Beware of numerical instabilities!
        return error


def main():
    from pytorch3d.transforms import random_rotations, so3_relative_angle
    from matplotlib import pyplot as plt
    x = random_rotations(n=64, dtype=th.float32)
    y = random_rotations(n=64, dtype=th.float32)
    loss = GeodesicLoss()
    z1 = (loss(x, y))
    z2 = (so3_relative_angle(x, y, cos_angle=False))
    print(z1)
    print(z2)

    plt.plot(z1, z2, '+')
    plt.show()


def train():
    th.autograd.set_detect_anomaly(True)
    from pytorch3d.transforms import random_rotations, so3_relative_angle
    from matplotlib import pyplot as plt

    n: int = 1

    y = random_rotations(n=n, dtype=th.float32)

    class Model(th.nn.Module):
        def __init__(self, n: int = n):
            super().__init__()
            x = random_rotations(n=n, dtype=th.float32)
            # NOTE(ycho): permute() to pass in row vectors.
            x = x[..., :, :2].permute(0, -1, -2).reshape(-1, 6)
            self.R6 = RotationOrtho6D(RotationOrtho6D.Settings())
            self.model = th.nn.Parameter(x, requires_grad=True)

        def forward(self):
            return self.R6(self.model)

    model = Model()
    optim = th.optim.Adam(model.parameters(), 1e-2)
    loss = GeodesicLoss()

    for _ in range(1024):
        optim.zero_grad()
        l = loss(model(), y)

        if not th.isfinite(l).all():
            print('non finite l!')
            break
        l.backward()
        optim.step()

        if not th.isfinite(model.model).all():
            print('loss')
            print(l)
            print('param')
            print(model.model)
            print('rotmat')
            print(model())
            print('NaN!')
            break
        print(l)
    print('relative angle')
    print(so3_relative_angle(model(), y))


if __name__ == '__main__':
    train()
