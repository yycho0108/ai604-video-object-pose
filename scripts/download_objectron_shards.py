#!/usr/bin/env python3

import multiprocessing as mp
import functools
import logging
import time

from typing import List
from tqdm.auto import tqdm
from dataclasses import dataclass
from simple_parsing import Serializable
from pathlib import Path
from google.cloud import storage

from top.data.objectron_detection import ObjectronDetection
from top.run.app_util import update_settings

bucket = None


def download_shard(shard: str, out_dir: str, bucket_local=None):
    """Download a single shard into `out_dir`.

    NOTE(ycho): The output file is automatically named according to the base-name of
    `shard`.
    """
    global bucket

    # Convert arg to a path object, just in case ...
    out_dir = Path(out_dir)

    # Configure names and download.
    basename = shard.split('/')[-1]
    out_file = (out_dir / basename)

    if bucket_local is None:
        # NOTE(ycho): Fallback to global bucket
        bucket_local = bucket
    blob = bucket_local.blob(shard)

    try:
        blob.download_to_filename(str(out_file))
    except KeyboardInterrupt as e:
        # NOTE(ycho): This seems to be the only working solution,
        # which is to cleanup only on SIGINT.
        # Catching a general `Exception` does not work. Not sure why.
        if out_file.exists():
            logging.debug(F'unlink: {out_file}')
            out_file.unlink()
        return 0

    # NOTE(ycho): since we're not downloading metadata through get_blob(),
    # we need to stat the local file for the size, in bytes.
    return out_file.stat().st_size


def download_shards(shards: List[str], out_dir: str,
                    stop: mp.Value, queue: mp.Queue):

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket('objectron')

    for shard in shards:
        # Check if we should stop.
        with stop.get_lock():
            if stop.value:
                break
        # Download ...
        shard_bytes = download_shard(shard, out_dir, bucket)
        if shard_bytes == 0:
            break
        # Return the number of downloaded bytes for accumulation.
        queue.put_nowait(shard_bytes)


@dataclass
class Settings(Serializable):
    max_train_bytes: int = 512 * (2 ** 30)  # default 32GB
    max_test_bytes: int = 4 * (2 ** 30)  # default 4GB
    num_workers: int = 8
    cache_dir: str = '~/.cache/ai604/'
    log_period_sec: float = 1.0
    use_pool: bool = False


def init_worker():
    """Set global variable `bucket` to point to cloud.

    NOTE(ycho): This function is only used for mp.Pool.
    """
    global bucket
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket('objectron')


def main():
    logging.basicConfig(level=logging.INFO)
    opts = Settings()
    opts = update_settings(opts)
    pool_states = [{} for _ in range(opts.num_workers)]
    for train in [False, True]:
        name = 'objectron-train' if train else 'objectron-test'
        logging.info(F'Processing {name}')

        max_bytes = opts.max_train_bytes if train else opts.max_test_bytes

        # TODO(ycho): Consider fancier (e.g. class-equalizing) shard samplers.
        shards = ObjectronDetection(
            ObjectronDetection.Settings(local=False), train).shards

        out_dir = (Path(opts.cache_dir).expanduser() / name)
        out_dir.mkdir(parents=True, exist_ok=True)

        if opts.use_pool:
            # NOTE(ycho): The initial approach based on mp.Pool().
            # Turned out that it is not possible to guarantee graceful exit in
            # this way.
            _download = functools.partial(download_shard, out_dir=out_dir)
            with mp.Pool(opts.num_workers, init_worker) as p:
                with tqdm(total=max_bytes) as pbar:
                    total_bytes = 0
                    for shard_bytes in p.imap_unordered(_download, shards):
                        pbar.update(shard_bytes)
                        # Accumulate and check for termination.
                        total_bytes += shard_bytes
                        if total_bytes >= max_bytes:
                            logging.info(F'Done: {total_bytes} > {max_bytes}')
                            # NOTE(ycho): Due to bug in mp.Pool(), imap_unordered() with close()/join()
                            # does NOT work, thus we implicitly call terminate() via context manager
                            # which may result in incomplete shards. This condition
                            # must be checked.
                            break
        else:
            init_bytes = sum(
                f.stat().st_size for f in out_dir.rglob('*') if f.is_file())
            logging.info(F'Starting from {init_bytes}/{max_bytes} ...')
            ctx = mp.get_context('fork')
            stop = ctx.Value('b', (init_bytes >= max_bytes))
            queue = ctx.Queue()
            workers = [ctx.Process(target=download_shards,
                                   args=(shards[i:: opts.num_workers],
                                         out_dir, stop, queue))
                       for i in range(opts.num_workers)]
            # Start!
            for p in workers:
                p.start()

            # Progress logging ...
            try:
                with tqdm(initial=init_bytes, total=max_bytes) as pbar:
                    # Periodically check progress...
                    total_bytes = init_bytes
                    while True:
                        shard_bytes = queue.get()
                        pbar.update(shard_bytes)
                        total_bytes += shard_bytes
                        if total_bytes >= max_bytes:
                            break
            except KeyboardInterrupt:
                logging.info('Cancelling download, trying to clean up ...')
                pass
            finally:
                # Stop.
                with stop.get_lock():
                    stop.value = True

                # Join.
                logging.info(
                    'Download completed, joining the rest of the processes...')
                for p in workers:
                    p.join()


if __name__ == '__main__':
    main()
