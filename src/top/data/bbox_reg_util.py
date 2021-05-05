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
from simple_parsing.helpers.serialization.serializable import Serializable

import torch as th
import cv2

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


"""
Class will hold the average dimension for a class, regressed value is the residual
"""
class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

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


# FIXME(Jiyong): need to test
class CropObject(object):
    """
    Crop object from image.
    2D keypoint -> 2D box(min/max) -> crop
    """

    @dataclass
    class Settings(Serializable):
        crop_img_size: Tuple[int, int, int] = (3, 224, 224)

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(self, inputs: dict):
        # Parse inputs
        image = inputs[Schema.IMAGE]
        class_index = inputs[Schema.CLASS]
        num_object = inputs[Schema.INSTANCE_NUM]
        num_keypoints = inputs[Schema.KEYPOINT_NUM]
        translation = inputs[Schema.TRANSLATION]
        orientation = inputs[Schema.ORIENTATION]
        scale = inputs[Schema.SCALE]

        h, w = image.shape[-2:]
        keypoints_2d = np.split(th.as_tensor(inputs[Schema.KEYPOINT_2D]), np.array(np.cumsum(num_keypoints)))
        keypoints_2d = [points.reshape(-1,3) for points in keypoints_2d]
        keypoints_2d = [np.multiply(keypoint, np.array([w, h, 1.0], np.float32)).astype(int)
                        for keypoint in keypoints_2d]
        
        crop_img = []
        for object_id in range(num_object):
            x_min, y_min, _ = np.min(keypoints_2d[object_id], axis=0)
            x_max, y_max, _ = np.max(keypoints_2d[object_id], axis=0)
            crop_tmp = image[:, x_min:x_max, y_min:y_max].copy()
            crop_tmp = cv2.resize(crop_tmp, dsize=self.opts.crop_img_size)
            crop_img.append(crop_tmp)

        return crop_img, class_index, translation, orientation, scale