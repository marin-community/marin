# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep regional CPU services owned by their RL run."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from fray.current_client import current_client
from fray.types import ANY_REGION, CpuConfig, JobRequest
from iris.client.client import iris_ctx
from rigging.secrets import COREWEAVE_SECRET_NAMES, SecretBehavior, require_secret

from marin.inference.iris import _terminate_job, _wait_for_endpoint


@dataclass(frozen=True)
class RunServiceEndpoint:
    name: str
    address: str


@contextmanager
def cpu_service(request: JobRequest, *, region: str, startup_timeout: float) -> Iterator[RunServiceEndpoint]:
    """Register a run-owned CoreWeave CPU child and terminate it on every exit."""
    resources = request.resources
    if resources.target_cluster is not None:
        raise ValueError("CPU service must inherit the RL run cluster")
    if resources.regions is not None and tuple(resources.regions) != (region,):
        raise ValueError("CPU service placement must match its run region")
    if not isinstance(resources.device, CpuConfig):
        raise ValueError("Run-owned CPU service requires CPU resources")
    configured_environment = request.environment
    if configured_environment is None:
        raise ValueError("CPU service requires an explicit environment")
    storage = {
        name: configured_environment.secrets[name] if name in configured_environment.secrets else require_secret(name)
        for name in COREWEAVE_SECRET_NAMES
    }
    environment = replace(
        configured_environment,
        secrets={**configured_environment.secrets, **storage},
        secret_behavior=SecretBehavior.ISOLATE,
    )
    # CPU pools lack region node labels. The child stays in its inherited
    # single-region cluster; each service verifies its physical placement.
    resources = replace(resources, regions=(ANY_REGION,))
    job = current_client().submit(replace(request, resources=resources, environment=environment), adopt_existing=False)
    try:
        name = f"{iris_ctx().namespace}/{request.name}"
        address, _ = _wait_for_endpoint(job, name, startup_timeout)
        yield RunServiceEndpoint(name=name, address=address)
    finally:
        _terminate_job(job)
