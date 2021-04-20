#!/usr/bin/env python3

import torch as th
from typing import Union


def resolve_device(device: Union[None, str, th.device]) -> th.device:
    """ Resolve torch device based on user input. """
    if device:
        device = th.device(device)
    else:
        if th.cuda.is_available():
            # NOTE(ycho): does NOT work for multi-gpu settings.
            device = th.device('cuda:0')
            th.cuda.set_device(device)
        else:
            device = th.device('cpu')
    return device


def main():
    print(resolve_device(''))


if __name__ == '__main__':
    main()
