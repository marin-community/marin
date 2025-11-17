# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, MutableMapping, Sequence, TypeAlias, TypeVar

T = TypeVar("T")
U = TypeVar("U")
py_slice = slice
slice_t: TypeAlias = slice


def index_where(pred: Callable[[T], bool], xs: Sequence[T], start: int = 0) -> int:
    for i in range(start, len(xs)):
        if pred(xs[i]):
            return i
    raise ValueError("No element satisfies predicate")


class IdentityMap(MutableMapping[T, U]):
    """Map that compares keys by identity.

    This is a map that compares keys by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable mapping, but it does not support the ``__hash__`` method
    and therefore cannot be used as a dictionary key or as an element of another
    set.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, key):
        return id(key) in self._data

    def __getitem__(self, key):
        return self._data[id(key)][1]

    def __setitem__(self, key, value):
        self._data[id(key)] = [key, value]

    def __delitem__(self, key):
        del self._data[id(key)]

    def __iter__(self):
        return (x[0] for x in self._data.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityMap({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentityMap({list(str(x) for x in self._data.values())})"
