"""
Reference:
    https://github.com/skhadem/3D-BoundingBox/blob/master/torch_lib/ClassAverages.py
    https://github.com/skhadem/3D-BoundingBox/blob/master/torch_lib/Dataset.py
"""


from dataclasses import dataclass
from typing import Tuple
import numpy as np
import os
import json
import copy
from simple_parsing import Serializable

import torch as th
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from pytorch3d.transforms import matrix_to_quaternion

from top.data.schema import Schema
from top.run.box_generator import Box

import pickle

"""
Enables writing json with numpy arrays to file
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)


class ClassAverages:
    """
    Class will hold the average dimension for a class, regressed value is the residual
    """
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.json'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        class_ = class_.lower()
        return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map


class CropObject(object):
    """
    Crop object from image.
    2D keypoint -> 2D box(min/max) -> crop
    """

    @dataclass
    class Settings(Serializable):
        crop_img_size: Tuple[int, int] = (224, 224)

    def __init__(self, opts: Settings):
        self.opts = opts

    # FIXME(Jiyon): make batch with cropped image, np,scipy -> th
    def __call__(self, inputs: dict):
        # Parse inputs
        image = inputs[Schema.IMAGE]
        num_object = inputs[Schema.INSTANCE_NUM]
        translation = inputs[Schema.TRANSLATION]
        orientation = inputs[Schema.ORIENTATION]
        scale = inputs[Schema.SCALE]
        visibility = inputs[Schema.VISIBILITY]

        c, h, w = image.shape[:]
        keypoints_2d_uv = inputs[Schema.KEYPOINT_2D]

        
        keypoints_2d = th.as_tensor(keypoints_2d_uv) * th.as_tensor([w, h, 1.0])
        num_vertices = keypoints_2d.shape[-2]
        keypoints_2d = keypoints_2d.reshape(-1,num_vertices,3)

        # clamp for the case that keypoints is in out of image
        keypoints_2d_clamp = th.clamp(th.as_tensor(keypoints_2d_uv), min=0, max=1)
        keypoints_2d_clamp = th.as_tensor(keypoints_2d_clamp) * th.as_tensor([w, h, 1.0])
        keypoints_2d_clamp = keypoints_2d_clamp.reshape(-1,num_vertices,3)

        orientation = th.as_tensor(orientation)
        orientation = orientation.reshape(-1,3,3)
        quaternions = matrix_to_quaternion(orientation)

        scale = th.as_tensor(scale)
        scale = scale.reshape(-1,3)

        translation = th.as_tensor(translation)
        translation = translation.reshape(-1,3)

        point_min = th.min(keypoints_2d_clamp, dim=1).values
        point_max = th.max(keypoints_2d_clamp, dim=1).values
        
        vis_img = th.tensor([])
        vis_crop_img = th.tensor([])
        vis_point_2d = th.tensor([])
        vis_trans = th.tensor([])
        vis_orient = th.tensor([])
        vis_scale = th.tensor([])
        
        for obj in range(num_object):
            # NOTE(Jiyion): If visiblity is false, cropping that object is impossible
            if not visibility[obj]:
                continue

            crop_tmp = image[:, int(point_min[obj][1]):int(point_max[obj][1]), int(point_min[obj][0]):int(point_max[obj][0])]
            crop_tmp = resize(crop_tmp, size=self.opts.crop_img_size)

            vis_img = th.cat((vis_img, image))
            vis_crop_img = th.cat((vis_crop_img, crop_tmp))
            vis_point_2d = th.cat((vis_point_2d, keypoints_2d[obj]))
            vis_trans = th.cat((vis_trans, translation[obj]))
            vis_orient = th.cat((vis_orient, quaternions[obj]))
            vis_scale = th.cat((vis_scale, scale[obj]))
            
        # shallow copy
        outputs = inputs.copy()
        outputs[Schema.IMAGE] = vis_img.reshape(-1, c, w, h)
        outputs[Schema.CROPPED_IMAGE] = vis_crop_img.reshape(-1, c, self.opts.crop_img_size[0], self.opts.crop_img_size[1])
        outputs[Schema.KEYPOINT_2D] = vis_point_2d.reshape(-1, 9, 3)
        outputs[Schema.TRANSLATION] = vis_trans.reshape(-1, 3)
        outputs[Schema.ORIENTATION] = vis_orient.reshape(-1, 4)
        outputs[Schema.SCALE] = vis_scale.reshape(-1, 3)

        print(inputs[Schema.INSTANCE_NUM])
        print(inputs[Schema.VISIBILITY])
        print(outputs[Schema.IMAGE].shape)
        print(outputs[Schema.CROPPED_IMAGE].shape)
        print(outputs[Schema.KEYPOINT_2D].shape)
        print(outputs[Schema.TRANSLATION].shape)
        print(outputs[Schema.ORIENTATION].shape)
        print(outputs[Schema.SCALE].shape)

        return outputs

