# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a zero-arg callable to run on a fray worker."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fray.v2 import client as fray_client
    from fray.v2.client import JobHandle
    from fray.v2.types import ResourceConfig


def _sanitize_job_name(name: str) -> str:
    """Ensure job names are compatible with Iris and Docker image tags."""
    sanitized = re.sub(r"[^a-z0-9_.-]+", "-", name.lower())
    sanitized = sanitized.strip("-.")
    return sanitized or "job"


def fray_exec(
    fn: Callable[[], Any],
    *,
    name: str = "job",
    resources: ResourceConfig | None = None,
    env_vars: dict[str, str] | None = None,
    pip_dependency_groups: list[str] | None = None,
    client: fray_client.Client | None = None,
) -> JobHandle:
    """Submit a zero-arg callable to run on a fray worker.

    Returns a ``JobHandle`` for polling/waiting.
    """
    from fray.v2 import client as fray_client_mod
    from fray.v2.types import Entrypoint, JobRequest, ResourceConfig, create_environment

    c = client or fray_client_mod.current_client()
    job = c.submit(
        JobRequest(
            name=_sanitize_job_name(name),
            entrypoint=Entrypoint.from_callable(fn),
            resources=resources or ResourceConfig.with_cpu(),
            environment=create_environment(
                extras=pip_dependency_groups or [],
                env_vars=env_vars or {},
            ),
        )
    )
    return job
