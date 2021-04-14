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
from tqdm import tqdm

# torch+torchvision
import torch as th
import torch.nn as nn
import torchvision.io as thio
from torchvision import transforms

from tfrecord.reader import sequence_loader
from google.cloud import storage


def _glob_objectron(bucket_name: str, prefix: str):
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]


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
                self.opts.features, None, True)

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
    """
    @dataclass
    class Settings:
        cache_dir: str = '~/.cache/ai604/'
        num_samples: int = 8  # how many samples to fetch
        objectron: Objectron.Settings = Objectron.Settings()

    def __init__(self, opts: Settings, transform=None):
        self.opts = opts
        self.objectron = Objectron(self.opts.objectron)
        self.data = self._build()
        self.xfm = transform

    def _build(self):
        # TODO(ycho): Avoid duplicating path resolution code.
        prefix = 'train' if self.opts.objectron.train else 'test'
        samples_cache = F'{self.opts.cache_dir}/{prefix}-sample.pkl'
        samples_path = Path(samples_cache).expanduser()
        # TODO(ycho): Avoid duplicating caching code.
        if not samples_path.exists():
            # Download data from scratch ...
            samples = []
            for i, data in tqdm(enumerate(self.objectron)):
                samples.append(data)
                if i >= self.opts.num_samples:
                    break
            with open(str(samples_path), 'wb') as f:
                pickle.dump(samples, f)
        else:
            with open(str(samples_path), 'rb') as f:
                samples = pickle.load(f)
        return samples

    def __iter__(self):
        index = 0
        while True:
            index += 1
            out = self.data[index % self.opts.num_samples]
            if self.xfm is not None:
                out = self.xfm(out)
            yield out


class DecodeImage:
    """
    Decode serialized image bytes, and resize them to a fixed size.
    FIXME(ycho): Modify camera intrinsics according to the resize transform.
    """

    def __init__(self, size: Tuple[int, int], in_place: bool = True):
        self.size = size
        self.in_place = in_place
        self.resize = transforms.Resize(self.size)

    def __call__(self, inputs: Tuple[dict, dict]):
        if inputs is None:
            return None
        context, features = inputs

        # NOTE(ycho): Shallow copy but pedantically safer
        if not self.in_place:
            features = features.copy()

        # Replace `image/encoded` with `image`
        features['image'] = th.stack([
            self.resize(thio.decode_image(th.from_numpy(img_bytes)))
            for img_bytes in features['image/encoded']], dim=0)

        del features['image/encoded']

        # Dimension features are no longer valid.
        # We can also rewrite them, but this information is already
        # implicitly included in features["image"].
        del features['image/width']
        del features['image/height']

        return (context, features)


class ParseFixedLength:
    """
    Convert variable-length sequence to fixed-length representation.
    Intended to be used immediately after the raw output from `Objectron`.

    FIXME(ycho): Isn't it a bit wasteful to slice up a full video once
    and discard the reset?? Discuss strategies on overcoming correlated-samples issue
    """
    class Settings:
        seq_len: int = 8  # length of sub-sequence to extract
        max_num_inst: int = 4  # max number of instances in the frame
        stride: int = 4  # number of intermediate frames to skip

        # NOTE(ycho): Variable-length fields to apply padding for
        # `max_num_inst` ... needed for collating to batch
        pad_keys: Tuple[str, ...] = (
            'object/translation', 'object/orientation', 'object/scale')

    def __init__(self, opts: Settings):
        self.opts = opts

    def __call__(
            self, inputs: Tuple[Dict[str, th.Tensor],
                                Dict[str, th.Tensor]]):
        if inputs is None:
            return None
        context, features = inputs

        # Reject data with less than `seq_len` frames.
        # NOTE(ycho): Unsqueeze `count` which is a scalar.
        n = context['count'][0]
        if n < self.opts.seq_len:
            return None

        # Extract a slice of length `seq_len` from the sequence.
        max_stride = n // self.opts.seq_len
        stride = min(max_stride, self.opts.stride)
        strided_len = stride * self.opts.seq_len

        i0 = np.random.randint(0, n - strided_len + 1)
        i1 = i0 + strided_len

        # FIXME(ycho):
        # sequence_id is currently a bytes list instead of `string`.
        # This makes it somewhat clunky to collate, so we'll drop this field
        # for now.
        # TODO(ycho): Consider adding `stride` to `new_ctx`
        new_ctx = dict(count=self.opts.seq_len)
        new_feat = {k: v[i0:i1:stride] for (k, v) in features.items()}

        # Some property are variable-length, since they are dependent on `instance_num`.
        # Pad the sequences to account for this.
        for key in self.opts.pad_keys:
            num_inst = new_feat['instance_num']
            x = new_feat[key]

            # single reference instance
            ref = th.as_tensor(x[0])
            # Compute single-feature dimension
            # TODO(ycho): consider assert for ref.shape[0] % num_inst[0] == 0
            dim = int(ref.shape[0] // num_inst[0])

            x_pad = ref.new_empty(
                (self.opts.seq_len, dim * self.opts.max_num_inst))
            if np.all(num_inst == num_inst[0]):
                # TODO(ycho): ensure this still works if
                # `instance_num` > `max_num_inst`
                m = min(dim * self.opts.max_num_inst, ref.shape[0])
                x_pad[:, :m] = th.as_tensor(x)[:, :m]
            else:
                max_n = min(self.opts.max_num_inst, num_ints)
                for i, n in enumerate(max_n):
                    x_pad[i, :n * dim] = th.as_tensor(x[i])
            new_feat[key] = x_pad

        return (new_ctx, new_feat)


def _skip_none(batch):
    """ Wrapper around default_collate() for skipping `None`. """
    batch = [x for x in batch if (x is not None)]
    return th.utils.data.dataloader.default_collate(batch)


def main():
    opts = Objectron.Settings()
    xfm = transforms.Compose([
        DecodeImage(size=(480, 640)),
        ParseFixedLength(ParseFixedLength.Settings()),
    ])
    dataset = Objectron(opts, xfm)
    loader = th.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0,
        collate_fn=_skip_none)

    for data in loader:
        # print(data)
        break


if __name__ == '__main__':
    main()
