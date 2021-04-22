"""
Compute priorbox coordinates in center-offset(2D) form for each source feature map.
Attempt to modify SSD anchor to 3D(scale:w,h,d). Create two kind of cube(small,large) in every feature map cell.
Reference: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/prior_box.py
"""

from typing import List
from dataclasses import dataclass
from simple_parsing import Serializable
from math import sqrt as sqrt
from itertools import product as product

import torch


# |TODO(Jiyong)|: change for Objectron setting
@dataclass
class AnchorSetting:
    image_size: int = 300
    feature_maps: List[int] = (7,)
    min_sizes: List[int] = (100,)
    max_sizes: List[int] = (200,)


class Anchor(object):
    """Compute priorbox coordinates in center-offset form for each source feature map."""
    def __init__(self, cfg: AnchorSetting):
        super(Anchor, self).__init__()
        self.image_size = cfg.image_size
        self.feature_maps = cfg.feature_maps
        self.min_sizes = cfg.min_sizes
        self.max_sizes = cfg.max_sizes

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / self.image_size
                cy = (i + 0.5) / self.image_size

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k, s_k]

                # aspect_ratio: 1
                # rel size: max_size
                s_k_prime = self.max_sizes[k]/self.image_size
                mean += [cx, cy, s_k_prime, s_k_prime, s_k_prime]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 5)

        return output


def main():
    cfg = AnchorSetting()
    anchor = Anchor(cfg)
    print(anchor.forward())

if __name__ == '__main__':
    main()