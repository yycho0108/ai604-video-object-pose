#!/usr/bin/env python3

"""
Test script for differentiable spatial to numerical (DSNT) layer in kornia.

FIXME(ycho): Currently does NOT work.

@article{nibali2018numerical,
  title={Numerical Coordinate Regression with Convolutional Neural Networks},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1801.07372},
  year={2018}
}
"""

import torch as th
from kornia.geometry.subpix import render_gaussian2d
from top.model.layers import DsntLayer2D

from matplotlib import pyplot as plt


def main():
    w = 64
    h = 64

    mean = th.as_tensor([0.2, 0.4], dtype=th.float32).reshape(1, 2)
    std = th.as_tensor([0.001, 0.001], dtype=th.float32).reshape(1, 2)
    c0 = render_gaussian2d(
        mean=mean, std=std, size=(h, w),
        normalized_coordinates=True)
    # Divide by max (potentially invalid in the actual model?).
    c0 /= c0.max()

    mean = th.as_tensor([-0.5, -0.5], dtype=th.float32).reshape(1, 2)
    std = th.as_tensor([0.005, 0.005], dtype=th.float32).reshape(1, 2)
    c1 = render_gaussian2d(
        mean=mean, std=std, size=(h, w),
        normalized_coordinates=True)
    # Divide by max (potentially invalid in the actual model?).
    c1 /= c1.max()

    inputs = th.stack([c0, c1], dim=1)  # 1,2,64,64

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(inputs[0, 0])
    ax[1].imshow(inputs[0, 1])
    plt.show()

    # NOTE(ycho): the combined effect of dividing by the maximum
    # and adjusting the temperature is sharpening the logit values
    # such that the output from the DSNT layer can remain precise.
    model = DsntLayer2D(DsntLayer2D.Settings(temperature=10.0))
    prob, kpts = model(inputs)
    print(kpts)


if __name__ == '__main__':
    main()
