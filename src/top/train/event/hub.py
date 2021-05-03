#!/usr/bin/env python3

from enum import Enum
from collections import defaultdict
from typing import (
    List, Dict, Tuple, Any, Union, Callable, Hashable
)


class Hub:
    """
    Simple synchronous event hub.
    Generally assumes `topic` to be `str, but can
    alternatively be any valid hashable type.
    """

    def __init__(self):
        self.subscribers = defaultdict(set)

    def publish(self, topic: Hashable, *args, **kwargs):
        """ Invoke subscriber callbacks with supplied args """
        for sub in self.subscribers[topic]:
            sub(*args, **kwargs)

    def subscribe(self, topic: Hashable, callback: Callable[[], None]):
        """ Register callbacks. """
        self.subscribers[topic].add(callback)

    def unsubscribe(self, topic: Hashable, callback: Callable[[], None]):
        """ Un-register callbacks. """
        self.subscribers[topic].remove(callback)


def test_hello_world():
    print('test_hello_world')
    hub = Hub()
    hub.subscribe('hello', lambda x: print('got : ' + x))
    hub.publish('hello', 'world')


def test_enum():
    print('test_enum')

    class Topics(Enum):
        HELLO = 'hello'
    hub = Hub()
    hub.subscribe(Topics.HELLO, lambda x: print('got : ' + x))
    hub.publish(Topics.HELLO, 'world')


def main():
    test_hello_world()
    test_enum()


if __name__ == '__main__':
    main()
