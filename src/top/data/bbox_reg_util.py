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
import cv2
from scipy.spatial.transform import Rotation as R

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
        class_index = inputs[Schema.CLASS]
        num_object = inputs[Schema.INSTANCE_NUM]
        num_keypoints = inputs[Schema.KEYPOINT_NUM]
        translation = inputs[Schema.TRANSLATION]
        orientation = inputs[Schema.ORIENTATION]
        scale = inputs[Schema.SCALE]

        h, w = image.shape[-2:]
        keypoints_2d = np.split(inputs[Schema.KEYPOINT_2D], np.array(np.cumsum(num_keypoints)))
        keypoints_2d = [points.reshape(-1,3) for points in keypoints_2d]
        keypoints_2d = [np.multiply(keypoint, np.array([w, h, 1.0], np.float32)).astype(int)
                        for keypoint in keypoints_2d]
        
        orientation = np.split(orientation, num_object)
        orientation = [rotation.reshape(-1,3,3) for rotation in orientation]

        scale = np.split(scale, num_object)
        scale = [scales.reshape(-1,3) for scales in scale]

        translation = np.split(translation, num_object)
        translation = [translations.reshape(-1,3) for translations in translation]

        crop_img = []
        quaternions = []
        scale_ = []
        translation_ = []
        for object_id in range(num_object):       
            # NOTE(Jiyong): np.split() leaves an empty array at the end of the list.
            if keypoints_2d[object_id].size == 0:
                break

            x_min, y_min, _ = np.min(keypoints_2d[object_id], axis=0)
            x_max, y_max, _ = np.max(keypoints_2d[object_id], axis=0)

            # NOTE(Jiyong): TypeError: Expected cv::UMat for argument 'src'
            # -> cv2.Umat() is functionally equivalent to np.float32() & (H,W,C)
            crop_tmp = np.float32(image[:, y_min:y_max, x_min:x_max])
            crop_tmp = np.transpose(crop_tmp, (1,2,0))
            try:
                crop_tmp = cv2.resize(crop_tmp, dsize=self.opts.crop_img_size)
            except Exception as e:
                print(crop_tmp.shape)
                raise

            crop_tmp = np.transpose(crop_tmp, (2,0,1))
            crop_img.append(crop_tmp)

            # For quaternions regression
            r = R.from_matrix(orientation[object_id])
            quaternions.append(r.as_quat())

            scale_.append(scale[object_id])
            translation_.append(translation[object_id])

        # shallow copy
        outputs = inputs.copy()
        outputs['crop_img'] = np.stack(crop_img, axis=0)
        outputs[Schema.TRANSLATION] = translation_
        outputs[Schema.SCALE] = scale_
        outputs[Schema.ORIENTATION] = quaternions
        outputs[Schema.VISIBILITY] = th.as_tensor(inputs[Schema.VISIBILITY]).reshape(-1,1)
        # print([(k, v.shape) if isinstance(v, th.Tensor) else (k,v) for k,v in outputs.items()])

        return outputs