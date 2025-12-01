# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import functools
from typing import Sequence, TypeAlias, TypeVar

import equinox

from haliax.jax_utils import is_jax_array_like

T = TypeVar("T")

Unspecified: TypeAlias = type("NotSpecified", (), {})  # type: ignore
UNSPECIFIED = Unspecified()


def is_named_array(leaf):
    from .core import NamedArray

    "Typically used as the is_leaf predicate in tree_map"
    return isinstance(leaf, NamedArray)


def ensure_tuple(x: Sequence[T] | T) -> tuple[T, ...]:
    if isinstance(x, str):
        return (x,)  # type: ignore
    elif isinstance(x, Sequence):
        return tuple(x)
    return (x,)


def maybe_untuple(x: Sequence[T] | T) -> T | Sequence[T]:
    """
    If x is a tuple with one element, return that element. Otherwise return x.
    """
    if isinstance(x, tuple) and len(x) == 1:
        return x[0]
    return x


class StringHolderEnum(type):
    """Like a python enum but just holds string constants, as opposed to wrapped string constants"""

    # https://stackoverflow.com/questions/62881486/a-group-of-constants-in-python

    def __new__(cls, name, bases, members):
        cls.members = [v for k, v in members.items() if not k.startswith("__") and not callable(v)]
        return super().__new__(cls, name, bases, members)

    # giving your class an __iter__ method gives you membership checking
    # and the ability to easily convert to another iterable
    @classmethod
    def __iter__(cls):
        yield from cls.members


def is_jax_or_hax_array_like(x):
    return is_jax_array_like(x) or is_named_array(x)


def safe_wraps(fn):
    """
    Equinox has a special [equinox.module_update_wrapper][] that works with [equinox.Module][]s, but
    doesn't work with regular functions. Likewise, functools.update_wrapper doesn't work with [equinox.Module][]s.

    This function is a wrapper around both of them that works with both [equinox.Module][]s and regular functions.

    Use this if you get this exception: `dataclasses.FrozenInstanceError: cannot assign to field '__module__'`
    """
    return functools.partial(safe_update_wrapper, wrapped=fn)


def safe_update_wrapper(wrapper, wrapped):
    """
    As [safe_wraps][] but not a decorator.
    Args:
        wrapper:
        wrapped:

    Returns:

    """
    if isinstance(wrapper, equinox.Module):
        return equinox.module_update_wrapper(wrapper, wrapped)
    else:
        return functools.update_wrapper(wrapper, wrapped)


__all__ = [
    "is_named_array",
    "ensure_tuple",
    "StringHolderEnum",
    "is_jax_or_hax_array_like",
]
