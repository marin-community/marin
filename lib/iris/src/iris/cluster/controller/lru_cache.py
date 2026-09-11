# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small thread-safe LRU cache."""

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


class LRUCache(Generic[_K, _V]):
    """Thread-safe LRU cache with a fixed maximum size."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._lock = threading.Lock()
        self._items: OrderedDict[_K, _V] = OrderedDict()

    def get(self, key: _K) -> _V | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: _K, value: _V) -> _V:
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                self._items.move_to_end(key)
                return existing
            self._items[key] = value
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)
            return value

    def pop(self, key: _K) -> None:
        with self._lock:
            self._items.pop(key, None)
