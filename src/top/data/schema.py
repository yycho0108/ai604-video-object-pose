#!/usr/bin/env python3

from enum import Enum

# NOTE(ycho): Required for dealing with `enum` registration
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class Schema(Enum):
    """
    Set of fixed constants to refer to contents from our dataset interfaces.
    NOTE(ycho): Generally intended to follow the Objectron schema.
    """
    IMAGE = "image"
    KEYPOINT_2D = "point_2d"
    KEYPOINT_3D = "point_3d"
    INSTANCE_NUM = "instance_num"
    TRANSLATION = "object/translation"
    ORIENTATION = "object/orientation"
    SCALE = "object/scale"
    CLASS = "object/class"  # NOTE(ycho): Class index, integer.
    PROJECTION = "camera/projection"
    KEYPOINT_MAP = "keypoint_map"
    HEATMAP = "object/heatmap"
    HEATMAP_LOGITS = "object/heatmap_logits"
    DISPLACEMENT_MAP = "displacement_map"
    KEYPOINT_NUM = "point_num"


@encode.register(Schema)
def encode_schema(obj: Schema) -> str:
    """Encode the enum with the underlying `str` representation. """
    return str(obj.value)


def decode_schema(obj: str) -> Schema:
    """ Decode str into Schema enum """
    return Schema(obj)


register_decoding_fn(Schema, decode_schema)


def main():
    s = encode_schema(Schema.KEYPOINT_2D)
    decode_schema(s)


if __name__ == '__main__':
    main()
