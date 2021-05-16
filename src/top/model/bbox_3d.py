
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import List, Tuple, Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from top.model.backbone import resnet_fpn_backbone
from top.model.loss_util import *
from top.data.schema import Schema


# TODO(Jiyong): compare various method to regress orientation,
# such as quaternions, rotation vector, euler anger, multibin, etc
class BoundingBoxRegressionModel(nn.Module):
    """
    regress parameters for projection from 2D to 3D bounding box 
    """

    @dataclass
    class Settings(Serializable):
        backbone_name: str = 'resnet50'
        num_trainable_layers: int = 0
        returned_layers: Tuple[int] = (4,)
        num_bins: int = 2
        w: float = 0.4

    def __init__(self, opts: Settings):
        
        super(BoundingBoxRegressionModel, self).__init__()
        self.opts = opts

        self.features = resnet_fpn_backbone(opts.backbone_name,
                                            pretrained=True,
                                            trainable_layers=opts.num_trainable_layers,
                                            returned_layers=opts.returned_layers)

        # # NOTE(Jiyong): This module is used for Multibin regression
        # # FIXME(Jiyong): hardcode for input size
        # self.confidence = nn.Sequential(
        #             nn.Linear(512 * 7 * 7, 256),
        #             nn.ReLU(True),
        #             nn.Dropout(),
        #             nn.Linear(256, 256),
        #             nn.ReLU(True),
        #             nn.Dropout(),
        #             nn.Linear(256, opts.num_bins),
        #             nn.Softmax()
        #         )

        # NOTE(Jiyong): This module is used for regressing quaternions
        # FIXME(Jiyong): hardcode for input size
        self.quaternions = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 256),
                    nn.ReLU(True),
                    # nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    # nn.Dropout(),
                    nn.Linear(256, 4)
                )

        # FIXME(Jiyong): hardcode for input size
        self.dimension = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 256),
                    nn.ReLU(True),
                    # nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    # nn.Dropout(),
                    nn.Linear(256, 3)
                )

    def forward(self, x):
        # FIXME(Jiyong): hardcoded feature layer
        x = self.features(x)['0']
        # confidence = self.confidence(x)
        dimension = th.squeeze(self.dimension(x))
        quaternions = th.squeeze(self.quaternions(x))

        # # NOTE(Jiyong): for multibin
        # # valid cos and sin values are obtained by applying an L2 norm.
        # tri_orientation = self.orientation(x)
        # tri_orientation = tri_orientation.view(-1, self.bins, 2)
        # tri_orientation = F.normalize(tri_orientation, dim=2)

        return dimension, quaternions