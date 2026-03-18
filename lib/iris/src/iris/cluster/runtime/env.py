# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment helpers for container runtimes.

Includes shared Iris env var construction, device env vars, and workdir file writing.
Used by Docker, process, and Kubernetes runtimes.

``build_common_iris_env`` is the single source of truth for the ~15 Iris system
env vars that every runtime must inject.  Backend-specific vars (e.g.
``IRIS_ADVERTISE_HOST`` via K8s downward API) are added by each runtime after
calling this function.
"""

import json
import logging
import posixpath
from pathlib import Path

from google.protobuf import json_format

from iris.cluster.constraints import INHERITED_CONSTRAINT_KEYS
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


def build_common_iris_env(
    *,
    task_id_wire: str,
    num_tasks: int,
    bundle_id: str,
    worker_id: str | None = None,
    controller_address: str | None = None,
    advertise_host: str | None = None,
    workdir: str = "/app",
    extras: list[str] | None = None,
    pip_packages: list[str] | None = None,
    user_env_vars: dict[str, str] | None = None,
    constraints: list[cluster_pb2.Constraint] | None = None,
    ports: dict[str, int] | None = None,
) -> dict[str, str]:
    """Build the Iris system environment variables common to all runtimes.

    This is the single source of truth for the env vars that mirror
    ``JobInfo.from_env()``.  Each runtime may override specific keys
    afterwards (e.g. K8s replaces ``IRIS_ADVERTISE_HOST`` with a downward
    API ref).

    Args:
        task_id_wire: Full task-attempt ID in wire format.
        num_tasks: Total tasks in the job.
        bundle_id: Bundle identifier.
        worker_id: Worker identifier (omitted when None).
        controller_address: Controller RPC address (omitted when None).
        advertise_host: Routable host IP for the task (omitted when None;
            K8s injects via downward API instead).
        workdir: Container working directory.
        extras: Optional extras list to propagate via ``IRIS_JOB_EXTRAS``.
        pip_packages: Optional pip packages to propagate via ``IRIS_JOB_PIP_PACKAGES``.
        user_env_vars: Explicit user env vars to propagate via ``IRIS_JOB_ENV``.
        constraints: Proto constraints; only inheritable keys are propagated.
        ports: Allocated port mapping (name → port number).
    """
    env: dict[str, str] = {}

    # Core task identity — mirrors JobInfo.from_env()
    env["IRIS_TASK_ID"] = task_id_wire
    env["IRIS_NUM_TASKS"] = str(num_tasks)
    env["IRIS_BUNDLE_ID"] = bundle_id

    if worker_id:
        env["IRIS_WORKER_ID"] = worker_id

    if controller_address:
        env["IRIS_CONTROLLER_ADDRESS"] = controller_address
        env["IRIS_CONTROLLER_URL"] = controller_address

    env["IRIS_BIND_HOST"] = "0.0.0.0"
    if advertise_host:
        env["IRIS_ADVERTISE_HOST"] = advertise_host

    env["IRIS_WORKDIR"] = workdir
    env["IRIS_PYTHON"] = "python"

    # Propagate extras and pip_packages so child jobs can inherit them
    if extras:
        env["IRIS_JOB_EXTRAS"] = json.dumps(extras)
    if pip_packages:
        env["IRIS_JOB_PIP_PACKAGES"] = json.dumps(pip_packages)

    # Serialize explicit user env vars so child jobs can inherit them
    if user_env_vars:
        env["IRIS_JOB_ENV"] = json.dumps(user_env_vars)

    # Only propagate region/zone constraints to children
    if constraints:
        inheritable = [c for c in constraints if c.key in INHERITED_CONSTRAINT_KEYS]
        if inheritable:
            env["IRIS_JOB_CONSTRAINTS"] = json.dumps(
                [json_format.MessageToDict(c, preserving_proto_field_name=True) for c in inheritable]
            )

    # Inject allocated ports
    if ports:
        for name, port in ports.items():
            env[f"IRIS_PORT_{name.upper()}"] = str(port)

    return env


def normalize_workdir_relative_path(path: str) -> str:
    """Return a normalized relative path safe to write under a task workdir."""
    candidate = path.replace("\\", "/")
    if candidate.startswith("/"):
        raise ValueError(f"Invalid workdir file path (absolute paths are not allowed): {path}")
    normalized = posixpath.normpath(candidate)
    if normalized in {"", "."}:
        raise ValueError(f"Invalid workdir file path: {path}")
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"Invalid workdir file path (path traversal): {path}")
    return normalized


def write_workdir_files(dest: Path, files: dict[str, bytes]) -> None:
    """Write workdir files under ``dest`` with path validation."""
    for name, data in files.items():
        normalized = normalize_workdir_relative_path(name)
        path = dest / normalized
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def build_device_env_vars(config: ContainerConfig) -> dict[str, str]:
    """Build device-specific environment variables for the container.

    When TPU resources are requested, adds JAX/PJRT environment variables
    and TPU metadata from the worker's environment. These environment variables
    enable JAX to properly initialize on TPU devices inside the container.
    """
    env: dict[str, str] = {}

    if not config.resources:
        logger.debug("No resources on container config; skipping device env vars")
        return env

    has_device = config.resources.HasField("device")
    has_tpu = has_device and config.resources.device.HasField("tpu")

    # N.B. We originally set all of the TPU environment variables explicitly, but this interferes with Jax's
    # automatic Cloud TPU detection. Forcing Jax to do Cloud TPU init is sufficient.
    if has_tpu:
        env["JAX_PLATFORMS"] = "tpu,cpu"
        env["PJRT_DEVICE"] = "TPU"

        # Jax likes to ignore the fact we're on a TPU for some reason.
        env["JAX_FORCE_TPU_INIT"] = "1"

    return env
