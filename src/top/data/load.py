#!/usr/bin/env python3

import enum
from dataclasses import dataclass, replace
from simple_parsing import Serializable
# NOTE(ycho): Required for dealing with `enum` registration
from simple_parsing.helpers.serialization import encode, register_decoding_fn
import torch as th

from top.data.colored_cube_dataset import ColoredCubeDataset
from top.data.objectron_sequence import ObjectronSequence
from top.data.objectron_detection import ObjectronDetection
from top.data.cached_dataset import CachedDataset

from collections import defaultdict

from top.data.schema import Schema


# TODO(ycho): add objectron sequence as an option.
class DatasetOptions(enum.Enum):
    CUBE = "CUBE"
    OBJECTRON = "OBJECTRON"

# NOTE(ycho): Register encoder-decoder pair for `DatasetOptions` enum.
# NOTE(ycho): Parsing from type annotations: only available for python>=3.7.


@encode.register(DatasetOptions)
def encode_dataset_options(obj: DatasetOptions) -> str:
    """Encode the enum with the underlying `str` representation."""
    return str(obj.value)


def decode_dataset_options(obj: str) -> DatasetOptions:
    return DatasetOptions(obj)


register_decoding_fn(DatasetOptions, decode_dataset_options)


@dataclass
class DatasetSettings(Serializable):
    dataset: DatasetOptions = DatasetOptions.OBJECTRON
    cube: ColoredCubeDataset.Settings = ColoredCubeDataset.Settings()
    objectron: ObjectronDetection.Settings = ObjectronDetection.Settings()
    cache: CachedDataset.Settings = CachedDataset.Settings()
    use_cached_dataset: bool = False
    shuffle: bool = False
    num_workers: int = 0


def get_dataset(opts: DatasetSettings, train: bool,
                device: th.device, transform=None):
    """Get a train/test dataset according to the specified settings.

    Provides a unified interface across multiple dataset types.
    Depending on the dataset type, some arguments are ignored.
    """
    if opts.dataset == DatasetOptions.CUBE:
        # NOTE(ycho): Ignores `train` argument.
        dataset = ColoredCubeDataset(opts.cube, device, transform)
    elif opts.dataset == DatasetOptions.OBJECTRON:
        # NOTE(ycho): Ignores `device` argument.
        dataset = ObjectronDetection(
            opts.objectron, train=train, transform=transform)
    else:
        raise ValueError(F'Invalid dataset choice : {opts.dataset} not found!')
    return dataset


def get_loaders(opts: DatasetSettings, device: th.device,
                batch_size: int, transform=None, collate_fn=None):
    """Fetch pair of (train,test) loaders for MNIST data."""

    if opts.use_cached_dataset:
        # FIXME(ycho): Hardcoded names for cache files.
        # But caching is mostly for development, so it's probably ok for now.
        # NOTE(ycho): Should NOT pass `transform` arg to get_dataset
        # since whatever transform should be passed in afterwards.
        train_dataset = CachedDataset(opts.cache,
                                      lambda: get_dataset(opts, True, device),
                                      'train-det-sample', transform)
        test_dataset = CachedDataset(opts.cache,
                                     lambda: get_dataset(opts, False, device),
                                     'test-det-sample', transform)
    else:
        train_dataset = get_dataset(opts, True, device, transform)
        test_dataset = get_dataset(opts, False, device, transform)

    train_loader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=opts.shuffle,
        num_workers=opts.num_workers,
        collate_fn=collate_fn)

    test_loader = th.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=opts.shuffle,
        num_workers=0,
        collate_fn=collate_fn)

    return (train_loader, test_loader)


def collate_cropped_img(data):
    out = defaultdict(list)
    # case that all objects in image are not visible
    # TODO(Jiyong): If all objects in batch are not visible, how to handle?

    # For index of data in batch
    idx = 0
    batch_idx = []
    for d in data:
        if all(vis == 0 for vis in d[Schema.VISIBILITY]):
            continue

        [out[k].append(v) for k, v in d.items()]

        batch_idx.append(idx)
        idx += 1

    for i, index in enumerate(out[Schema.INDEX]):
        tmp = th.full(index.shape, batch_idx[i])
        out[Schema.INDEX][i] = th.cat((tmp, index), dim=1)

    for k in out:
        v = out[k]
        if len(v) > 0 and isinstance(v[0], th.Tensor):
            out[k] = th.cat(v, dim=0)

    return out
