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
# SCALE_MUON_ forwards the distributed Newton-Schulz layout knobs read at trace time in
# levanter.optim.grugmuon.
# TF_CPP_ forwards TSL logging controls (TF_CPP_VMODULE etc.) so XLA runtime
# behavior on the train tasks can be traced from the submitter.
_FORWARDED_ENV_PREFIXES = ("XLA_", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "CE_", "SCALE_MUON_", "TF_CPP_")
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)
_PROBE_LIBRARY_PATH = "/app/.venv/lib/libmarin_cuda_module_probe.so"
_PROBE_SOURCE_PATH = "/app/experiments/grug/moe/standalone/cuda_module_probe.cc"


def _probe_build_script() -> str:
    return f"""set -e
mkdir -p "$MARIN_CUDA_MODULE_PROBE_LOG_DIR"
"$IRIS_VENV/bin/python" -m experiments.grug.moe.standalone.cuda_module_probe build \\
  --source {_PROBE_SOURCE_PATH} \\
  --output {_PROBE_LIBRARY_PATH} \\
  --compiler "${{CXX:-c++}}"
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
    env_vars: dict[str, str] | None = None,
    max_retries_failure: int = 3,
    processes_per_task: int = 1,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    explicit_env = dict(env_vars or {})
    child_env = resolve_training_env(base_env={**_forwarded_env_vars(), **explicit_env}, resources=resources)
    setup_scripts = None
    if "MARIN_CUDA_MODULE_PROBE_PROFILE" in explicit_env:
        child_env["LD_PRELOAD"] = _PROBE_LIBRARY_PATH
        extras = extras_for_resources(resources)
        setup_scripts = [
            default_setup_script(extras=extras, python_version=f"{sys.version_info.major}.{sys.version_info.minor}"),
            cuda_toolchain_setup_script(),
            _probe_build_script(),
        ]
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(
            env_vars=child_env,
            extras=extras_for_resources(resources),
            setup_scripts=setup_scripts,
        ),
        max_retries_failure=max_retries_failure,
        processes_per_task=processes_per_task,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
