#!/usr/bin/env python3

__all__ = ['Objectron']

import os
import sys
import io
from dataclasses import dataclass
from simple_parsing import Serializable
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import pickle
import tempfile

# torch+torchvision
import torch as th
import torchvision.io as thio
from torchvision import transforms

from google.cloud import storage

from top.data.cached_dataset import CachedDataset
from top.data.schema import Schema
from top.run.app_util import update_settings

from tfrecord.reader import tfrecord_loader

NUM_KEYPOINTS = 9


def _glob_objectron(bucket_name: str, prefix: str):
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]


def decode(example, feature_names: List[str] = []):
    w = example['image/width'].item()
    h = example['image/height'].item()
    points = example['point_2d']
    num_instances = example['instance_num'].item()
    num_keypoints = example['point_num']
    points = points.reshape(num_instances, NUM_KEYPOINTS, 3)

    # NOTE(ycho): Loading directly with torch to avoid warnings with PIL.
    image = thio.decode_image(th.from_numpy(example['image/encoded']))

    translation = example['object/translation'].reshape(num_instances, 3)
    orientation = example['object/orientation'].reshape(num_instances, 9)
    scale = example['object/scale'].reshape(num_instances, 3)
    visibility = example['object/visibility'].reshape(num_instances, 1)

    out = {
        Schema.IMAGE: image,
        Schema.KEYPOINT_2D: points,
        Schema.INSTANCE_NUM: num_instances,
        Schema.TRANSLATION: translation,
        Schema.ORIENTATION: orientation,
        Schema.SCALE: scale,
        Schema.PROJECTION: example['camera/projection'],
        Schema.KEYPOINT_NUM: num_keypoints,
        Schema.VISIBILITY: visibility
    }

    out.update({k: example[k] for k in feature_names})
    return out


class ObjectronDetection(th.utils.data.IterableDataset):
    """
    Objectron dataset.
    TODO(ycho): Rename to `ObjectronDetection`.
    """

    @dataclass
    class Settings(Serializable):
        bucket_name: str = 'objectron'
        classes: Tuple[str, ...] = (
            'bike',
            'book',
            'bottle',
            'camera',
            'cereal_box',
            'chair',
            'cup',
            'laptop',
            'shoe')
        shuffle_shards: bool = True
        # NOTE(ycho): Refer to objectron/schema/features.py
        context: Tuple[str, ...] = ('count', 'sequence_id')
        # NOTE(ycho): Refer to objectron/schema/features.py
        features: Tuple[str, ...] = (
            'instance_num',
            'image/width',
            'image/height',
            'image/channels',
            'image/encoded',
            'object/name',
            'object/translation',
            'object/orientation',
            'object/scale',
            'object/visibility',
            'point_3d',
            'point_2d',
            'point_num',
            'camera/intrinsics',
            'camera/projection',
            'object/visibility'
        )
        cache_dir = '~/.cache/ai604/'

    def __init__(self, opts: Settings, train: bool = True, transform=None):
        self.opts = opts
        self.train = train
        self.shards = self._get_shards()
        self.xfm = transform

        self._index_map = {c: i for i, c in enumerate(self.opts.classes)}
        self._class_map = {i: c for i, c in enumerate(self.opts.classes)}

        # hmm
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(self.opts.bucket_name)

    def _index_from_class(self, cls: str):
        return self._index_map[cls]

    def _class_from_index(self, idx: str):
        return self._class_map[idx]

    def _get_shards(self):
        """ return list of shards, potentially memoized for efficiency """
        prefix = 'train' if self.train else 'test'
        # FIXME(ycho): hardcoded {prefix}-det-shards
        shards_cache = F'{self.opts.cache_dir}/{prefix}-det-shards.pkl'

        shards_path = Path(shards_cache).expanduser()
        if not shards_path.exists():
            shards_path.parent.mkdir(parents=True, exist_ok=True)
            shards = self._glob()
            with open(str(shards_path), 'wb') as f:
                pickle.dump(shards, f)
        else:
            with open(str(shards_path), 'rb') as f:
                shards = pickle.load(f)

        if self.opts.shuffle_shards:
            np.random.shuffle(shards)
        return shards

    def _glob(self) -> List[str]:
        train_type = 'train' if self.train else 'test'
        shards = []

        # Aggregate class shards
        for cls in self.opts.classes:
            prefix = F'v1/records_shuffled/{cls}/{cls}_{train_type}'
            cls_shards = _glob_objectron(self.opts.bucket_name, prefix)
            shards.extend(cls_shards)
        return shards

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            i0 = 0
            i1 = len(self.shards)
        else:
            n = int(np.ceil(len(self.shards) / float(worker_info.num_workers)))
            i0 = worker_info.id * n
            i1 = min(len(self.shards), i0 + n)

        # Contiguous block slice among shards.
        for shard in self.shards[i0:i1]:
            blob = self.bucket.blob(shard)

            # Download blob and parse tfrecord.
            # TODO(ycho): Consider downloading in the background.
            content = blob.download_as_bytes()
            with io.BytesIO(content) as fp:
                reader = tfrecord_loader(fp, None,
                                         self.opts.features, None)

                # Parse class name from the shard name.
                # This is to avoid awkward string conversion later.
                class_name = shard.split('/')[-2]
                class_index = self._index_from_class(class_name)

                for i, example in enumerate(reader):
                    # Decode example into features format ...
                    features = decode(example)

                    # Add class information.
                    features[Schema.CLASS] = th.full(
                        (int(features[Schema.INSTANCE_NUM]),), class_index)

                    # Transform output and return.
                    output = features
                    if self.xfm:
                        output = self.xfm(output)
                    yield output


class SampleObjectron(th.utils.data.IterableDataset):
    """
    Class that behaves exactly like `Objectron`, except
    this class loads from a locally cached data, for convenience.
    Prefer this class for testing / validation / EDA.

    @see CachedDataset, Objectron.
    """
    @dataclass
    class Settings(Serializable):
        cache: CachedDataset.Settings = CachedDataset.Settings()
        objectron: ObjectronDetection.Settings = ObjectronDetection.Settings()

    def __init__(self, opts: Settings, transform=None):
        self.opts = opts
        # NOTE(ycho): delegate most of the heavy lifting
        # to `CachedDataset`.
        prefix = 'train' if self.opts.objectron.train else 'test'
        # FIXME(ycho): hardcoded {prefix}-det-sample
        self.dataset = CachedDataset(
            opts.cache,
            lambda: ObjectronDetection(
                self.opts.objectron),
            F'{prefix}-det-sample',
            transform=transform)
        self.xfm = transform

    def __iter__(self):
        return self.dataset.__iter__()


def main():
    opts = SampleObjectron.Settings()
    opts = update_settings(opts)
    dataset = SampleObjectron(opts)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0,
        collate_fn=None)

    for data in loader:
        print(data.keys())
        # print(data[Schema.KEYPOINT_2D])
        break


if __name__ == '__main__':
    main()
