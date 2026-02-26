# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decorator for marking step functions for remote execution via Fray.

Without ``@remote``, steps run locally in-thread. With ``@remote``, they are
submitted as Fray jobs with the specified resources.

Usage::

    @remote
    def tokenize(...): ...          # CPU defaults

    @remote(resources=ResourceConfig.with_tpu("v4-128"))
    def train(...): ...             # explicit resources
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from fray.v2.types import ResourceConfig

P = ParamSpec("P")
R = TypeVar("R")


def remote(
    fn: Callable[P, R] | None = None, *, resources: ResourceConfig | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a step function for remote execution via Fray.

    When applied without arguments (``@remote``), the function will run with
    default CPU resources. When called with ``resources=``, the supplied
    ``ResourceConfig`` is used instead.
    """
    if resources is None:
        resources = ResourceConfig.with_cpu()

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return f(*args, **kwargs)

        wrapper.__fray_resources__ = resources  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
