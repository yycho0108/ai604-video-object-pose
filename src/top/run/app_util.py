#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import sys
import logging
from typing import List
from dataclasses import dataclass, is_dataclass
from simple_parsing import Serializable, ArgumentParser

use_argcomplete = False
try:
    import argcomplete
    use_argcomplete = True
except ImportError:
    logging.info('argcomplete disabled due to missing package.')


def update_settings(opts: dataclass, argv: List[str] = None):
    """
    Update given settings from command line arguments.
    Uses `argparse`, `argcomplete` and `simple_parsing` under the hood.
    """
    if not is_dataclass(opts):
        raise ValueError('Cannot update args on non-dataclass class')

    parser = ArgumentParser()
    parser.add_arguments(type(opts), dest='opts', default=opts)
    if use_argcomplete:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.opts


def main():
    logging.basicConfig(level=logging.INFO)

    @dataclass
    class Settings:
        value: int = 0

    opts = Settings()
    opts = update_settings(opts)
    logging.info(F'got value = {opts.value}')


if __name__ == '__main__':
    main()
