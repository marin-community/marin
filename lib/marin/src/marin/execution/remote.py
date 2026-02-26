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

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, ParamSpec, TypeVar

from fray.v2 import client as fray_client
from fray.v2.types import ResourceConfig, Entrypoint, JobRequest, create_environment

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_JOB_NAME = "fray_exec_job"


def _sanitize_job_name(name: str) -> str:
    """Ensure job names are compatible with Iris and Docker image tags."""
    sanitized = re.sub(r"[^a-z0-9_.-]+", "-", name.lower())
    sanitized = sanitized.strip("-.")
    return sanitized or DEFAULT_JOB_NAME


def _args_hash(*args: object, **kwargs: object) -> str:
    """Short hash of call arguments to disambiguate jobs with the same function name."""
    content = repr(args) + repr(sorted(kwargs.items()))
    return hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class RemoteCallable(Generic[P, R]):
    """A callable wrapper that submits its function to Fray when called.

    Carries Fray-specific execution config: resources, environment variables,
    and pip dependency groups. When called, submits the wrapped function to
    Fray and blocks until completion.
    """

    fn: Callable[P, R]
    resources: ResourceConfig
    env_vars: dict[str, str] = field(default_factory=dict)
    pip_dependency_groups: list[str] = field(default_factory=list)
    name: str | None = None

    # TODO: JobHandle doesn't have this option now, but we could make this return the R
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Submit fn to Fray and block until completion."""

        base_name = self.name or getattr(self.fn, "__name__", None) or DEFAULT_JOB_NAME
        name = f"{base_name}-{_args_hash(*args, **kwargs)}"
        c = fray_client.current_client()
        handle = c.submit(
            JobRequest(
                name=_sanitize_job_name(name),
                entrypoint=Entrypoint.from_callable(lambda: self.fn(*args, **kwargs)),
                resources=self.resources,
                environment=create_environment(
                    extras=self.pip_dependency_groups,
                    env_vars=self.env_vars,
                ),
            )
        )
        handle.wait(raise_on_failure=True)


def remote(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
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
        return RemoteCallable(
            fn=f,
            resources=resources,
            env_vars=env_vars or {},
            pip_dependency_groups=pip_dependency_groups or [],
            name=name,
        )

    if fn is not None:
        return decorator(fn)
    return decorator
