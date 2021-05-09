#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import sys
import logging
import argparse
from typing import List
from dataclasses import dataclass, is_dataclass
from simple_parsing import Serializable, ArgumentParser

use_argcomplete = False
try:
    import argcomplete
    use_argcomplete = True
except ImportError:
    logging.info('argcomplete disabled due to missing package.')


def _update_settings_from_file(opts: dataclass, argv: List[str]):
    """
    Update given settings from a configuration file.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config_file',
                               help='Configuration file', dest='config_file',
                               metavar='CONFIG_FILE', default='')
    parsed_args, next_argv = config_parser.parse_known_args(argv)

    if parsed_args.config_file:
        opts = opts.load(parsed_args.config_file)
        return (opts, next_argv, [config_parser])

    # If no `config_file` arg, fallback to passthrough behavior.
    return (opts, argv, [])


def update_settings(opts: dataclass, argv: List[str] = None):
    """
    Update given settings from command line arguments.
    Uses `argparse`, `argcomplete` and `simple_parsing` under the hood.
    """
    if not is_dataclass(opts):
        raise ValueError('Cannot update args on non-dataclass class')

    # Use default system argv if not supplied.
    argv = sys.argv[1:] if argv is None else argv

    # Update from config file, if applicable.
    parser_parents = []
    if isinstance(opts, Serializable):
        opts, argv, parser_parents = _update_settings_from_file(opts, argv)

    parser = ArgumentParser(parents=parser_parents)
    parser.add_arguments(type(opts), dest='opts', default=opts)
    if use_argcomplete:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
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
