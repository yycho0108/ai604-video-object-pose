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
from simple_parsing import Serializable

import torch as th
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion

from top.data.schema import Schema
from top.run.box_generator import Box

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

        h, w = image.shape[-2:]
        keypoints_2d_uv = inputs[Schema.KEYPOINT_2D]
        keypoints_2d = th.as_tensor(keypoints_2d_uv) * th.as_tensor([w, h, 1.0])
        num_vertices = keypoints_2d.shape[-2]
        keypoints_2d = keypoints_2d.reshape(-1,num_vertices,3)
        
        orientation = th.as_tensor(orientation)
        orientation = orientation.reshape(-1,3,3)
        quaternions = matrix_to_quaternion(orientation)

        scale = th.as_tensor(scale)
        scale = scale.reshape(-1,3)

        translation = th.as_tensor(translation)
        translation = translation.reshape(-1,3)

        x_min, y_min, _ = th.min(keypoints_2d, dim=1).values
        x_max, y_max, _ = th.max(keypoints_2d, dim=1).values
        
        crop_img = th.empty(num_object, self.opts.crop_img_size, self.opts.crop_img_size)
        for obj in range(num_object):
            crop_tmp = image[:, y_min[obj]:y_max[obj], x_min[obj]:x_max[obj]]
            crop_tmp = th.transpose(crop_tmp, (1,2,0))
            try:
                crop_tmp = F.interpolate(crop_tmp, dsize=self.opts.crop_img_size)
            except Exception as e:
                print(crop_tmp.shape)
                raise
            crop_tmp = th.transpose(crop_tmp, (2,0,1))
            crop_img[obj] = crop_tmp

        # shallow copy
        outputs = inputs.copy()
        outputs[Schema.CROPPED_IMAGE] = crop_img
        outputs[Schema.TRANSLATION] = translation
        outputs[Schema.SCALE] = scale
        outputs[Schema.ORIENTATION] = quaternions
        outputs[Schema.VISIBILITY] = th.as_tensor(inputs[Schema.VISIBILITY]).reshape(-1,1)
        # print([(k, v.shape) if isinstance(v, th.Tensor) else (k,v) for k,v in outputs.items()])

        return outputs