# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small compatibility example for the package's Python 3.12 typing surface."""

from collections.abc import Callable
from dataclasses import dataclass

from typing_extensions import TypeIs


@dataclass(frozen=True, slots=True)
class Value[T]:
    value: T


def is_text(value: object) -> TypeIs[str]:
    return isinstance(value, str)


def recover[T](operation: Callable[[], T], fallback: T) -> Value[T]:
    """Recover a grouped runtime failure without hiding unrelated failures."""
    value = fallback
    try:
        value = operation()
    except* RuntimeError:
        pass
    return Value(value)
