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
from torchvision.transforms.functional import resized_crop
from pytorch3d.transforms import matrix_to_quaternion

from top.data.schema import Schema
from top.run.box_generator import Box



class NumpyEncoder(json.JSONEncoder):
    """
    Enables writing json with numpy arrays to file
    """
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

        keypoint_2d_min = th.min(keypoints_2d_clamp, dim=1).values
        keypoint_2d_max = th.max(keypoints_2d_clamp, dim=1).values
        
        visible_crop_img = []
        visible_point_2d = []
        visible_trans = []
        visible_quat = []
        visible_scale = []
        visible_box_2d = []
        
        for obj in range(num_object):
            # NOTE(Jiyion): If visiblity is false, cropping that object is impossible
            if not visibility[obj]:
                continue

            top = int(keypoint_2d_min[obj][1])
            left = int(keypoint_2d_min[obj][0])
            height = int(keypoint_2d_max[obj][1]) - int(keypoint_2d_min[obj][1])
            width = int(keypoint_2d_max[obj][0]) - int(keypoint_2d_min[obj][0])

            # Occasionally we get an empty ROI due to floating-point precision
            if width<=0 or height<=0:
                continue
            crop_tmp = resized_crop(image, top, left, height, width, size=self.opts.crop_img_size)

            visible_crop_img.append(crop_tmp)
            visible_point_2d.append(keypoints_2d[obj])
            visible_trans.append(translation[obj])
            visible_quat.append(quaternions[obj])
            visible_scale.append(scale[obj])
            visible_box_2d.append(th.as_tensor([top, left, height, width]))
        
        # shallow copy
        outputs = inputs.copy()
        # case that all objects in image are not visible
        # TODO(Jiyong): If all objects in batch are not visible, how to handle?
        if all(vis == 0 for vis in visibility):
            outputs[Schema.CROPPED_IMAGE] = th.tensor(visible_crop_img).reshape(-1, c, self.opts.crop_img_size[0], self.opts.crop_img_size[1])
            outputs[Schema.KEYPOINT_2D] = th.tensor(visible_point_2d).reshape(-1, num_vertices, 3)
            outputs[Schema.TRANSLATION] = th.tensor(visible_trans).reshape(-1, 3)
            outputs[Schema.QUATERNION] = th.tensor(visible_quat).reshape(-1, 4)
            outputs[Schema.SCALE] = th.tensor(visible_scale).reshape(-1, 3)
            outputs[Schema.BOX_2D] = th.tensor(visible_box_2d).reshape(-1, 4)

            return outputs
        
        outputs[Schema.CROPPED_IMAGE] = th.stack(visible_crop_img).reshape(-1, c, self.opts.crop_img_size[0], self.opts.crop_img_size[1])
        outputs[Schema.KEYPOINT_2D] = th.stack(visible_point_2d).reshape(-1, num_vertices, 3)
        outputs[Schema.TRANSLATION] = th.stack(visible_trans).reshape(-1, 3)
        outputs[Schema.QUATERNION] = th.stack(visible_quat).reshape(-1, 4)
        outputs[Schema.SCALE] = th.stack(visible_scale).reshape(-1, 3)
        outputs[Schema.BOX_2D] = th.stack(visible_box_2d).reshape(-1, 4)

        return outputs

