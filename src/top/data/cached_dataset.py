#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable
from pathlib import Path
import pickle
from typing import (List, Tuple, Dict, Callable)
from tqdm import tqdm
import torch as th


class CachedDataset(th.utils.data.IterableDataset):
    """
    Class that memoizes a subset of the given dataset,
    and retrieves them during runtime from the filesystem cache.
    """
    @dataclass
    class Settings(Serializable):
        cache_dir: str = '~/.cache/ai604/'
        num_samples: int = 8  # NOTE(ycho): How many samples to fetch.
        force_rebuild: bool = False

    def __init__(self,
                 opts: Settings,
                 dataset_fn: Callable[[None], th.utils.data.Dataset],
                 name: str,
                 transform=None):
        self.opts = opts
        self.dataset_fn = dataset_fn
        self.name = name
        self.data = self._build()
        self.xfm = transform

    def _build(self):
        samples_cache = F'{self.opts.cache_dir}/{self.name}.pkl'
        samples_path = Path(samples_cache).expanduser()

        if (self.opts.force_rebuild) or (not samples_path.exists()):
            # Download data from scratch ...
            dataset = self.dataset_fn()
            samples = []
            for i, data in tqdm(enumerate(dataset)):
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
