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
    """ prefix = google cloud storage bucket glob prefix """
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
    image_bytes = example['image/encoded']
    # NOTE(ycho): Workaround to suppress negligible torch warnings
    # image_bytes.setflags(write=True)
    image = thio.decode_image(th.from_numpy(image_bytes))

    translation = example['object/translation'].reshape(num_instances, 3)
    orientation = example['object/orientation'].reshape(num_instances, 9)
    scale = example['object/scale'].reshape(num_instances, 3)
    visibility = example['object/visibility'].reshape(num_instances, 1)

    out = {
        Schema.IMAGE: th.as_tensor(image),
        Schema.KEYPOINT_2D: th.as_tensor(points),
        Schema.INSTANCE_NUM: num_instances,
        Schema.TRANSLATION: th.as_tensor(translation),
        Schema.ORIENTATION: th.as_tensor(orientation),
        Schema.SCALE: th.as_tensor(scale),
        Schema.PROJECTION: th.as_tensor(example['camera/projection']),
        Schema.KEYPOINT_NUM: num_keypoints,
        Schema.VISIBILITY: visibility,
        Schema.CLASS: bytes(example['object/name']).decode(),
        Schema.INTRINSIC_MATRIX: th.as_tensor(example['camera/intrinsics'])
    }

    out.update({k: example[k] for k in feature_names})
    return out


class ObjectronDetection(th.utils.data.IterableDataset):
    """Objectron dataset.

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
        )
        cache_dir: str = '~/.cache/ai604/'
        local: bool = True

    def __init__(self, opts: Settings, train: bool = True, transform=None):
        self.opts = opts
        self.train = train
        self.shards = self._get_shards()
        self.xfm = transform

        self._index_map = {c: i for i, c in enumerate(self.opts.classes)}
        self._class_map = {i: c for i, c in enumerate(self.opts.classes)}

        # Conditionally instantiate google cloud storage bucket interface.
        self.bucket = None
        if not self.opts.local:
            client = storage.Client.create_anonymous_client()
            self.bucket = client.bucket(self.opts.bucket_name)

    def _index_from_class(self, cls: str):
        return self._index_map[cls]

    def _class_from_index(self, idx: str):
        return self._class_map[idx]

    def _get_shards(self):
        """return list of shards, potentially memoized for efficiency."""
        train_type = 'train' if self.train else 'test'
        if self.opts.local:
            # FIXME(ycho): hardcoded {cache_dir}/objectron-{train_type}/...
            # shards
            root = Path(
                self.opts.cache_dir).expanduser() / F'objectron-{train_type}'
            shards = list(root.glob('*'))
        else:
            # FIXME(ycho): hardcoded {train_type}-det-shards
            shards_cache = F'{self.opts.cache_dir}/{train_type}-det-shards.pkl'

            shards_path = Path(shards_cache).expanduser()
            if not shards_path.exists():
                shards_path.parent.mkdir(parents=True, exist_ok=True)
                shards = self._glob()
                with open(str(shards_path), 'wb') as f:
                    pickle.dump(shards, f)
            else:
                with open(str(shards_path), 'rb') as f:
                    shards = pickle.load(f)

        # NOTE(ycho): `numpy` not strictly needed here.
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
        train_type = 'train' if self.train else 'test'

        # Distribute shards among interleaved workers.
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            i0 = 0
            step = 1
        else:
            i0 = worker_info.id
            step = worker_info.num_workers

        for shard in self.shards[i0::step]:
            # Download blob and parse tfrecord.
            # TODO(ycho): Consider downloading in the background.

            fp = None
            try:
                if self.opts.local:
                    # Load local shard.
                    fp = open(str(shard), 'rb')
                else:
                    # Load shard from remote GS bucket.
                    blob = self.bucket.blob(shard)
                    content = blob.download_as_bytes()
                    fp = io.BytesIO(content)

                reader = tfrecord_loader(fp, None,
                                         self.opts.features, None)

                for i, example in enumerate(reader):
                    # Decode example into features format ...
                    features = decode(example)

                    # Broadcast class information.
                    features[Schema.CLASS] = th.full(
                        (int(features[Schema.INSTANCE_NUM]),),
                        self._index_from_class(features[Schema.CLASS]))

                    # Transform output and return.
                    output = features
                    if self.xfm:
                        output = self.xfm(output)
                    yield output
            finally:
                # Ensure that the `fp` resource is properly released.
                if fp is not None:
                    fp.close()


class SampleObjectron(th.utils.data.IterableDataset):
    """Class that behaves exactly like `Objectron`, except this class loads
    from a locally cached data, for convenience. Prefer this class for testing.

    / validation / EDA.

    @see CachedDataset, Objectron.
    """
    @dataclass
    class Settings(Serializable):
        cache: CachedDataset.Settings = CachedDataset.Settings()
        objectron: ObjectronDetection.Settings = ObjectronDetection.Settings()

    def __init__(self, opts: Settings, train: bool = True, transform=None):
        self.opts = opts
        # NOTE(ycho): delegate most of the heavy lifting
        # to `CachedDataset`.
        train_type = 'train' if train else 'test'
        # FIXME(ycho): hardcoded {train_type}-det-sample
        self.dataset = CachedDataset(
            opts.cache,
            lambda: ObjectronDetection(
                self.opts.objectron, train),
            F'{train_type}-det-sample',
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
