# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# `JobRequest.priority` is the Iris priority band as a bare int. INHERIT is Iris's own default.
INHERIT_PRIORITY = priority_band_value("inherit")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
_FORWARDED_ENV_PREFIXES = ("XLA_", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "MALLOC_")
_FORWARDED_ENV_NAMES = ("LD_PRELOAD",)
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if (k in _FORWARDED_ENV_NAMES or k.startswith(_FORWARDED_ENV_PREFIXES)) and k not in _FORWARDED_ENV_EXCLUDE
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
    max_task_failures: int = 10,
    processes_per_task: int = 1,
    priority: int = INHERIT_PRIORITY,
    pip_packages: Sequence[str] = (),
    extra_env_vars: Mapping[str, str] | None = None,
    setup_scripts: Sequence[str] | None = None,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion.

    ``INHERIT_PRIORITY`` takes the submitting job's band, or ``interactive`` when the submitter is
    not itself an Iris job -- which is the case for a launcher run from a dev box.

    ``max_retries_failure`` is the per-task retry budget and ``max_task_failures`` is the
    cumulative one. The job fails when either is exhausted, so raise the two together: a large
    per-task budget under a small cumulative one still ends the job at the cumulative limit.

    ``pip_packages``, ``extra_env_vars``, and ``setup_scripts`` extend the task environment for
    runs that need a dependency the standard image does not carry, such as the pinned
    Transformer Engine build in ``experiments/grug/te_setup.py``. ``setup_scripts`` replaces the
    task's whole setup: Iris then ignores ``pip_packages`` and the resource extras, so those
    scripts must install both themselves (``default_setup_script(extras=...)`` renders the
    standard one). Passing ``pip_packages`` alongside ``setup_scripts`` is rejected rather than
    silently dropped.
    """
    if pip_packages and setup_scripts is not None:
        raise ValueError(
            "Iris ignores pip_packages when setup_scripts is set; install them from the scripts "
            "instead (default_setup_script(pip_packages=...))."
        )
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(
        base_env={**_forwarded_env_vars(), **(extra_env_vars or {})},
        resources=resources,
    )
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(
            env_vars=env_vars,
            extras=extras_for_resources(resources),
            pip_packages=pip_packages,
            setup_scripts=setup_scripts,
        ),
        max_retries_failure=max_retries_failure,
        max_task_failures=max_task_failures,
        processes_per_task=processes_per_task,
        priority=priority,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
