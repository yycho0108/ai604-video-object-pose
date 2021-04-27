#!/usr/bin/env python3

from enum import Enum


class Schema(Enum):
    """
    Set of fixed constants to refer to contents from our dataset interfaces.
    NOTE(ycho): Generally intended to follow the Objectron schema.
    """
    IMAGE = "image"
    KEYPOINT_2D = "point_2d"
    INSTANCE_NUM = "instance_num"
    TRANSLATION = "object/translation"
    ORIENTATION = "object/orientation"
    SCALE = "object/scale"
    CLASS = "object/class"  # NOTE(ycho): Class index, integer.
    PROJECTION = "camera/projection"
