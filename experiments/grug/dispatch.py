# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
# CE_ forwards CE_IMPL / CE_LIGER_CHUNK / CE_LIGER_UNROLL, which select the grug fused
# cross-entropy backend (read via os.environ at trace time in levanter.grug.loss).
_FORWARDED_ENV_PREFIXES = ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "CE_")
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
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    # GRUG_RUN_INLINE=1 runs the training entrypoint in-process instead of submitting a Fray
    # job. Use it when already inside an allocated node (e.g. a dev_gpu pod) so the run uses
    # the current node's GPUs directly and does not queue for a second allocation.
    if os.environ.get("GRUG_RUN_INLINE") == "1":
        nproc = int(os.environ.get("SCALE_PROCESSES_PER_TASK", "1"))
        if nproc > 1:
            # Multi-process inline: build the config ONCE here (the executor already ran in this
            # single process), then spawn the iris.runtime.multigpu supervisor over `nproc` fresh
            # worker processes (one GPU each) that reload the pickled config and run the training.
            # This mirrors the Fray path (executor once, entrypoint N times) and avoids the
            # executor's per-step lock deadlocking an N-way collective. TE's FFI needs the
            # multi-controller topology (single-controller deadlocks).

            cfg_fd, config_path = tempfile.mkstemp(prefix="grug_inline_cfg_", suffix=".pkl")
            os.close(cfg_fd)
            with open(config_path, "wb") as handle:
                pickle.dump(config, handle)
            child_env = dict(os.environ, GRUG_INLINE_CONFIG_PATH=config_path)
            cmd = [
                sys.executable,
                "-m",
                "iris.runtime.multigpu",
                "--nproc",
                str(nproc),
                "--devices-per-proc",
                "1",
                "--",
                sys.executable,
                "-m",
                "experiments.grug.moe._inline_multigpu_worker",
            ]
            logger.info("GRUG_RUN_INLINE=1: spawning %d-process multigpu training (config=%s)", nproc, config_path)
            result = subprocess.run(cmd, env=child_env)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            return
        logger.info("GRUG_RUN_INLINE=1: running grug training inline (no Fray dispatch)")
        local_entrypoint(config)
        return
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env_vars, extras=extras_for_resources(resources)),
        max_retries_failure=max_retries_failure,
        processes_per_task=processes_per_task,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
