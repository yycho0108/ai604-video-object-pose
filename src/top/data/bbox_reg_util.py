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

        h, w = image.shape[-2:]
        keypoints_2d_uv = inputs[Schema.KEYPOINT_2D]

        # clamp for the case that keypoints is in out of image
        keypoints_2d_clamp = th.clamp(th.as_tensor(keypoints_2d_uv), min=0, max=1)
        keypoints_2d = th.as_tensor(keypoints_2d_clamp) * th.as_tensor([w, h, 1.0])
        num_vertices = keypoints_2d.shape[-2]
        keypoints_2d = keypoints_2d.reshape(-1,num_vertices,3)
        
        orientation = th.as_tensor(orientation)
        orientation = orientation.reshape(-1,3,3)
        quaternions = matrix_to_quaternion(orientation)

        scale = th.as_tensor(scale)
        scale = scale.reshape(-1,3)

        translation = th.as_tensor(translation)
        translation = translation.reshape(-1,3)

        point_min = th.min(keypoints_2d, dim=1).values
        point_max = th.max(keypoints_2d, dim=1).values
        
        crop_img = th.empty(num_object, 3, self.opts.crop_img_size[0], self.opts.crop_img_size[1])
        for obj in range(num_object):
            # FIXME(Jiyion): Why is there a case where all keypoints are negative?
            crop_w = int(point_max[obj][0]) - int(point_min[obj][0])
            crop_h = int(point_max[obj][1]) - int(point_min[obj][1])
            if crop_w <= 0 or crop_h <= 0:
                print(keypoints_2d[obj])
                print(keypoints_2d_uv)
                print(num_object)
                with open('/tmp/wtf.pkl', 'wb') as fp:
                    dbg = inputs[Schema.IMAGE].cpu().numpy()
                    pickle.dump(dbg, fp)
                continue
            
            crop_tmp = image[:, int(point_min[obj][1]):int(point_max[obj][1]), int(point_min[obj][0]):int(point_max[obj][0])]
            try:
                crop_tmp = resize(crop_tmp, size=self.opts.crop_img_size)
            except RuntimeError:
                print(point_min[obj][1], point_max[obj][1], point_min[obj][0], point_max[obj][0])
                print(keypoints_2d_uv)
                print(keypoints_2d_clamp)
                print(keypoints_2d)
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
