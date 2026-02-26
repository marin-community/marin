# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decorator for marking step functions for remote execution via Fray.

Without ``@remote``, steps run locally in-thread. With ``@remote``, they are
submitted as Fray jobs with the specified resources.

Usage::

    @remote
    def tokenize(config): ...          # CPU defaults

    @remote(resources=ResourceConfig.with_tpu("v4-128"))
    def train(config): ...             # explicit resources
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from fray.v2.types import ResourceConfig

F = TypeVar("F", bound=Callable)


def remote(fn: F | None = None, *, resources: ResourceConfig | None = None) -> F | Callable[[F], F]:
    """Mark a step function for remote execution via Fray.

    When applied without arguments (``@remote``), the function will run with
    default CPU resources. When called with ``resources=``, the supplied
    ``ResourceConfig`` is used instead.
    """
    from fray.v2.types import ResourceConfig as RC

    if resources is None:
        resources = RC.with_cpu()

    def decorator(f: F) -> F:
        f.__fray_resources__ = resources  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorator(fn)
    return decorator
