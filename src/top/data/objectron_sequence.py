#!/usr/bin/env python3

__all__ = ['ObjectronSequence']

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
from tqdm import tqdm

# torch+torchvision
import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

from tfrecord.reader import sequence_loader
from google.cloud import storage

from top.data.cached_dataset import CachedDataset


def _glob_objectron(bucket_name: str, prefix: str):
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]


class ObjectronSequence(th.utils.data.IterableDataset):
    """
    Objectron dataset (currently configured for loading sequences only).
    """

    @dataclass
    class Settings(Serializable):
        bucket_name: str = 'objectron'
        classes: Tuple[str] = (
            'bike',
            'book',
            'bottle',
            'camera',
            'cereal_box',
            'chair',
            'cup',
            'laptop',
            'shoe')
        train: bool = True
        shuffle: bool = True
        # NOTE(ycho): Refer to objectron/schema/features.py
        context: List[str] = ('count', 'sequence_id')
        # NOTE(ycho): Refer to objectron/schema/features.py
        features: List[str] = (
            'instance_num',
            'image/width',
            'image/height',
            'image/channels',
            'image/encoded',
            'object/name',
            'object/translation',
            'object/orientation',
            'object/scale',
            'camera/intrinsics',
            'camera/extrinsics',
            'camera/projection',
            'camera/view',
        )
        cache_dir = '~/.cache/ai604/'

    def __init__(self, opts: Settings, transform=None):
        self.opts = opts
        self.shards = self._get_shards()
        self.xfm = transform

        self._index_map = {c: i for i, c in enumerate(self.opts.classes)}
        self._class_map = {i: c for i, c in enumerate(self.opts.classes)}

    def _index_from_class(self, cls: str):
        return self._index_map[cls]

    def _class_from_index(self, idx: str):
        return self._class_map[idx]

    def _get_shards(self):
        """ return list of shards, potentially memoized for efficiency """
        prefix = 'train' if self.opts.train else 'test'
        shards_cache = F'{self.opts.cache_dir}/{prefix}-shards.pkl'

        shards_path = Path(shards_cache).expanduser()
        if not shards_path.exists():
            shards_path.parent.mkdir(parents=True, exist_ok=True)
            shards = self._glob()
            with open(str(shards_path), 'wb') as f:
                pickle.dump(shards, f)
        else:
            with open(str(shards_path), 'rb') as f:
                shards = pickle.load(f)

        if self.opts.shuffle:
            np.random.shuffle(shards)
        return shards

    def _glob(self) -> List[str]:
        train_type = 'train' if self.opts.train else 'test'
        shards = []

        # Aggregate class shards
        for cls in self.opts.classes:
            prefix = F'v1/sequences/{cls}/{cls}_{train_type}'
            cls_shards = _glob_objectron(self.opts.bucket_name, prefix)
            shards.extend(cls_shards)
        return shards

    def __iter__(self):
        # Deal with parallelism...
        # NOTE(ycho): harder to assume uncorrelated inputs
        # without multiple workers loading from different shards
        # on a single batch.
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            i0 = 0
            i1 = len(self.shards)
        else:
            n = int(np.ceil(len(self.shards) / float(worker_info.num_workers)))
            i0 = worker_info.id * n
            i1 = min(len(self.shards), i0 + n)

        # Contiguous block slice among shards
        for shard in self.shards[i0:i1]:
            # Resolve shard name relative to bucket.
            shard_name = F'gs://{self.opts.bucket_name}/{shard}'

            # Parse class name from the shard name.
            # This is to avoid awkward string conversion later.
            class_name = shard.split('/')[-2]
            class_index = self._index_from_class(class_name)

            # NOTE(ycho): tfrr doesn't work due to
            # GCS + SequenceExample. Therefore, we use
            # a custom fork of `tfrecord` package.
            # TODO(ycho): Protection on network error cases
            # Which will inevitably arise during long training
            reader = sequence_loader(
                shard_name, None, self.opts.context,
                self.opts.features, None)

            # Wonder how many samples there are per-shard??
            for i, (context, features) in enumerate(reader):
                # Replace object/name to object/class here.
                names = features.pop('object/name')

                # NOTE(ycho): Instead of trying to parse bytes->str->index,
                # Just rewrite object class with the known class index.
                # TODO(ycho): Watch out for heterogeneous frames, if any.
                # FIXME(ycho): ^^ quite likely?
                classes = th.as_tensor(
                    [class_index for c in names],
                    dtype=th.int32)
                features['object/class'] = classes

                out = (context, features)

                # Apply optional transform on the data.
                # NOTE(ycho): This step is mandatory if we'd like to convert the dataset
                # to fixed-length representation.
                if self.xfm is not None:
                    out = self.xfm(out)
                yield out


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
        objectron: ObjectronSequence.Settings = ObjectronSequence.Settings()

    def __init__(self, opts: Settings, transform=None):
        self.opts = opts
        # NOTE(ycho): delegate most of the heavy lifting
        # to `CachedDataset`.
        prefix = 'train' if self.opts.objectron.train else 'test'
        self.dataset = CachedDataset(opts.cache,
                                     lambda: ObjectronSequence(self.opts.objectron),
                                     F'{prefix}-sample',
                                     transform=transform)
        self.xfm = transform

    def __iter__(self):
        return self.dataset.__iter__()


def _skip_none(batch):
    """ Wrapper around default_collate() for skipping `None`. """
    batch = [x for x in batch if (x is not None)]
    return th.utils.data.dataloader.default_collate(batch)


def main():
    opts = SampleObjectron.Settings()
    xfm = transforms.Compose([
        DecodeImage(size=(480, 640)),
        ParseFixedLength(ParseFixedLength.Settings()),
    ])
    dataset = SampleObjectron(opts, xfm)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=_skip_none)

    for data in loader:
        # print(data)
        break


if __name__ == '__main__':
    main()
