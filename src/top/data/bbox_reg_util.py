"""
Reference:
    https://github.com/skhadem/3D-BoundingBox/blob/master/torch_lib/ClassAverages.py
    https://github.com/skhadem/3D-BoundingBox/blob/master/torch_lib/Dataset.py
"""


import numpy as np
import os
import json
from PIL import Image

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
    project 3D point to 2D -> 2D box(min/max) -> crop
    """

    def _proj_2d_box(self, orientation, translation, scale):
        """Project 3D bouning box to 2D.(min/max of vertices)"""
        bbox_3d = Box.from_transformation(orientation, translation, scale)
        vertices = bbox_3d._vertices
        x_min, y_min, _ = np.min(vertices, axis=0)
        x_max, y_max, _ = np.max(vertices, axis=0)
        return x_min, x_max, y_min, y_max

    def __call__(self, inputs: dict):
        # Parse inputs
        image = inputs[Schema.IMAGE] # (C,H,W)
        class_index = inputs[Schema.CLASS]
        translation = inputs[Schema.TRANSLATION]
        orientation = inputs[Schema.ORIENTATION]
        scale = inputs[Schema.SCALE]
        x_min, x_max, y_min, y_max = self._proj_2d_box(orientation, translation, scale)
        croped_img = image.crop

        return