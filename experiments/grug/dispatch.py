# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.v2.client import current_client
from fray.v2.types import Entrypoint, JobRequest, create_environment
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


def _with_jax_distributed_init(fn: Callable[[ConfigT], None], config: ConfigT) -> None:
    """Wrapper that initializes JAX distributed before running the entrypoint.

    On multi-host TPU, Fray's fn_thunk subprocess doesn't auto-initialize
    JAX distributed. Calling jax.distributed.initialize() without args
    lets JAX auto-detect the TPU topology. On single-host this is a no-op.

    Also enables faulthandler so SIGABRT/SIGSEGV from libtpu (which can hard-abort
    the process when the TPU runtime detects launch-group/scheckne mismatches)
    print a Python traceback to stderr instead of dying silently. Diagnostic for
    issue #5319.
    """
    import faulthandler
    import signal
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
    for sig in (signal.SIGABRT, signal.SIGSEGV, signal.SIGFPE, signal.SIGBUS, signal.SIGILL):
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=True)
        except Exception:
            pass

    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize()
    fn(config)


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    extras = _default_environment_extras(resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(_with_jax_distributed_init, args=[local_entrypoint, config]),
        resources=resources,
        environment=create_environment(extras=extras),
        max_retries_failure=max_retries_failure,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
