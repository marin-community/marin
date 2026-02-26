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
from typing import Generic, ParamSpec, TypeVar

from fray.v2.types import ResourceConfig

P = ParamSpec("P")
R = TypeVar("R")


class RemoteCallable(Generic[P, R]):
    """A callable wrapper that marks a function for remote execution via Fray.

    Wraps the original function and carries Fray-specific execution config:
    resources, environment variables, and pip dependency groups.
    """

    def __init__(
        self,
        fn: Callable[P, R],
        resources: ResourceConfig,
        env_vars: dict[str, str] | None = None,
        pip_dependency_groups: list[str] | None = None,
    ):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.resources = resources
        self.env_vars = env_vars or {}
        self.pip_dependency_groups = pip_dependency_groups or []

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._fn(*args, **kwargs)


def remote(
    fn: Callable[P, R] | None = None,
    *,
    resources: ResourceConfig | None = None,
    env_vars: dict[str, str] | None = None,
    pip_dependency_groups: list[str] | None = None,
) -> RemoteCallable[P, R] | Callable[[Callable[P, R]], RemoteCallable[P, R]]:
    """Mark a step function for remote execution via Fray.

    When applied without arguments (``@remote``), the function will run with
    default CPU resources. When called with ``resources=``, the supplied
    ``ResourceConfig`` is used instead.
    """
    if resources is None:
        resources = ResourceConfig.with_cpu()

    def decorator(f: Callable[P, R]) -> RemoteCallable[P, R]:
        return RemoteCallable(f, resources, env_vars=env_vars, pip_dependency_groups=pip_dependency_groups)

    if fn is not None:
        return decorator(fn)
    return decorator
