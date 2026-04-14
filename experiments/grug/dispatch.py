# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.v2.client import current_client
from fray.v2.types import Entrypoint, JobRequest, PriorityBand, create_environment
from fray.v2.types import GpuConfig, TpuConfig

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray/Iris-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def _default_environment_extras(resources: ResourceConfig) -> list[str]:
    if isinstance(resources.device, TpuConfig):
        return ["tpu"]
    if isinstance(resources.device, GpuConfig):
        return ["gpu"]
    return []


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
    priority_band: PriorityBand | None = None,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    extras = _default_environment_extras(resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(extras=extras),
        max_retries_failure=max_retries_failure,
        priority_band=priority_band,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
