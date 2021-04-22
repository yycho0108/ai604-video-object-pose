#!/usr/bin/env python3

import logging
from pathlib import Path
from dataclasses import dataclass


def _ensure_directory(path: str):
    """
    Ensure that the directory structure exists.
    """
    path = Path(path)
    if path.is_dir():
        return

    if path.exists():
        raise ValueError(F'path {path} exists and is not a directory.')
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise ValueError(F'Failed to create path {path}.')


class RunPath(object):
    """
    General path management over multiple experiment runs.

    NOTE(ycho): The intent of this class is mainly to avoid overwriting
    checkpoints and existing logs from a previous run -
    instead, we maintain a collision-free index based key
    for each experiment that we run and use them in a sub-folder structure.
    """

    @dataclass
    class Settings:
        key_format: str = 'run-{:03d}'
        root: str = '/tmp/'  # Alternatively, ~/.cache/ai604/run/
        key: str = '' # Empty string indicates auto increment.
 
    def __init__(self, opts: Settings):
        self.opts = opts
        self.root = Path(opts.root).expanduser()
        _ensure_directory(self.root)

        # Resolve sub-directory key.
        key = opts.key
        if key is '':
            key = self._resolve_key(self.root, self.opts.key_format)
            logging.info(F'key={key}')

        self.dir = self.root / key
        _ensure_directory(self.dir)
        logging.info(F'self.dir={self.dir}')

    @staticmethod
    def _resolve_key(root: str, key_fmt: str) -> str:
        """ Get latest valid key according to `key_fmt` """
        # Ensure `root` is a valid directory.
        root = Path(root)
        if not root.is_dir():
            raise ValueError(F'Arg root={root} is not a dir.')

        # NOTE(ycho): Loop through integers starting from 0.
        # Not necessarily efficient, but convenient.
        index = 0
        while True:
            key = key_fmt.format(index)
            if not (root / key).exists():
                break
            index += 1
        return key

    def __getattr__(self, key: str):
        """ Convenient shorthand for fetching valid subdirectories. """
        out = self.dir / key
        _ensure_directory(out)
        return out


def main():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        rp = RunPath(RunPath.Settings(root=tmpdir))
        logging.debug(rp.log)
        rp2 = RunPath(RunPath.Settings(root=tmpdir))
        logging.debug(rp2.log)


if __name__ == '__main__':
    main()
