# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import sys
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
# CE_ forwards CE_IMPL / CE_LIGER_CHUNK / CE_LIGER_UNROLL (read via os.environ at trace time in
# levanter.grug.loss) so the chunked CE reaches the nested training tasks.
# XLA_ covers XLA_FLAGS and XLA_PYTHON_CLIENT_ALLOCATOR (cuda_async anti-fragmentation pool).
# SCALE_ forwards experiment knobs read at trace time, including distributed
# Newton-Schulz layout and fixed-capacity expert all-to-all settings.
# WANDB_ keeps tracker authentication and routing intact across the nested job.
_FORWARDED_ENV_PREFIXES = ("XLA_", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "CE_", "SCALE_", "WANDB_")
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)

_HYBRIDEP_SETUP_SCRIPT = r"""
set -e
export PYTHONPATH="$IRIS_WORKDIR/scripts/hybridep_build_probe${PYTHONPATH:+:$PYTHONPATH}"
"$IRIS_VENV/bin/python" - <<'PY'
from pathlib import Path

from restore_torch_bundle import restore_cuda13_toolkit, restore_hybridep_bundle

restore_cuda13_toolkit()
restore_hybridep_bundle(Path("/tmp"))
PY
"""


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
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion.

    ``GRUG_RUN_INLINE=1`` runs the entrypoint in-process instead of submitting a Fray job.
    Use it when already inside an allocated node (e.g. a federated GB200 job) so the run
    uses the current node's GPUs directly rather than queuing for a second allocation.
    """
    if os.environ.get("GRUG_RUN_INLINE") == "1":
        logger.info("GRUG_RUN_INLINE=1: running grug training inline (no Fray dispatch)")
        local_entrypoint(config)
        return
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    extras = extras_for_resources(resources)
    setup_scripts = None
    if os.environ.get("SCALE_A2A_HYBRID_EP") == "1":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        setup_scripts = [
            default_setup_script(extras=extras, python_version=python_version),
            cuda_toolchain_setup_script(),
            _HYBRIDEP_SETUP_SCRIPT,
        ]
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env_vars, extras=extras, setup_scripts=setup_scripts),
        max_retries_failure=max_retries_failure,
        max_task_failures=10,
        processes_per_task=processes_per_task,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
