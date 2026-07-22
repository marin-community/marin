# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cluster.setup_scripts import (
    cuda_toolchain_setup_script,
    default_setup_script,
    setup_is_quiet,
    wants_gpu_extra,
)
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
_FORWARDED_ENV_PREFIXES = (
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_",
    "LIBTPU_INIT_ARGS",
    "NCCL_",
    "JAX_",
    "TF_GPU_ALLOCATOR",
    "GRUG_JAXPP_",
)
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_ENV_PREFIXES) and k not in _FORWARDED_ENV_EXCLUDE
    }


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray/Iris-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
    processes_per_task: int = 1,
    post_setup_scripts: Sequence[str] = (),
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    extras = extras_for_resources(resources)
    setup_scripts = None
    if post_setup_scripts:
        setup_scripts = [
            default_setup_script(extras=extras, quiet=setup_is_quiet(env_vars)),
        ]
        if wants_gpu_extra(extras):
            setup_scripts.append(cuda_toolchain_setup_script())
        setup_scripts.extend(post_setup_scripts)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env_vars, extras=extras, setup_scripts=setup_scripts),
        max_retries_failure=max_retries_failure,
        processes_per_task=processes_per_task,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
