#!/usr/bin/env python3

__all__ = ['Objectron']

import os
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import pickle

# torch+torchvision
import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

import torch_xla.utils.tf_record_reader as tfrr
from google.cloud import storage


NUM_KEYPOINTS = 9


def _glob_objectron(bucket_name: str, prefix: str):
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]

def decode(example, feature_names:List[str] = []):
    w = example['image/width'].item()
    h = example['image/height'].item()
    points = example['point_2d'].numpy()
    num_instances = example['instance_num'].item()
    points = points.reshape(num_instances, NUM_KEYPOINTS, 3)
    image_data = example['image/encoded'].numpy().tobytes()
    image = Image.open(io.BytesIO(image_data))
    npa = np.asarray(image)
    out = {
        'image' : th.from_numpy(npa),
        'points' : points,
        'num_instances' : num_instances
    }

    out.update({k:example[k] for k in feature_names})
    return out
    
class Objectron(th.utils.data.IterableDataset):
    """
    Objectron dataset (currently configured for loading sequences only).
    """

    @dataclass
    class Settings:
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
            'POINT_3D',
            'POINT_NUM',
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

        if self.opts.shuffle:
            np.random.shuffle(shards)
        return shards

    def _glob(self) -> List[str]:
        train_type = 'train' if self.opts.train else 'test'
        shards = []

        # Aggregate class shards
        for cls in self.opts.classes:
            prefix = F'v1/records_shuffled/{cls}/{cls}_{train_type}'
            cls_shards = _glob_objectron(self.opts.bucket_name, prefix)
            shards.extend(cls_shards)
        return shards

    def __iter__(self):

        # Contiguous block slice among shards
        for shard in self.shards:
            # Resolve shard name relative to bucket.
            shard_name = F'gs://{self.opts.bucket_name}/{shard}'

            # Parse class name from the shard name.
            # This is to avoid awkward string conversion later.
            class_name = shard.split('/')[-2]
            class_index = self._index_from_class(class_name)
            
            r = tfrr.TfRecordReader(shard_name, compression='', transforms=None)
            while True:
                example = r.read_example()
                if not example: break

                features = decode(example)
                features['class_name'] = class_name
                features['class_index'] = class_index
            
            output = features
            if self.xfm:
                output = self.xfm(output)
            yield output

def main():
    opts = Objectron.Settings()
    # xfm = transforms.Compose([
    #     DecodeImage(size=(480, 640)),
    #     ParseFixedLength(ParseFixedLength.Settings()),
    # ])
    dataset = Objectron(opts)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=None)

    for data in loader:
        print(data)
        break


if __name__ == '__main__':
    main()
