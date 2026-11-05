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
    """Demonstrate except* recovery for a disposable package compatibility probe.

    The supplied operation declares RuntimeError as recoverable; every other
    exception propagates. Effects replace this probe in the next work package.
    """
    value = fallback
    try:
        value = operation()
    except* RuntimeError:
        pass
    return Value(value)
