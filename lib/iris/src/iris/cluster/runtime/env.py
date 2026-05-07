# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment helpers for container runtimes.

Includes the shared env-var builder used by both the worker (Docker/process)
and Kubernetes paths, plus workdir file writing utilities.
"""

import json
import logging
import posixpath
from collections.abc import Sequence
from pathlib import Path

from google.protobuf import json_format

from iris.cluster.constraints import INHERITED_CONSTRAINT_KEYS
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


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


def build_common_iris_env(
    *,
    task_id: str,
    attempt_id: int,
    num_tasks: int,
    bundle_id: str,
    controller_address: str | None,
    environment: job_pb2.EnvironmentConfig,
    constraints: Sequence[job_pb2.Constraint],
    ports: Sequence[str],
    resources: job_pb2.ResourceSpecProto | None,
) -> dict[str, str]:
    """Build the Iris system env vars shared by both worker and k8s paths.

    This is the single source of truth for env vars derived from a
    RunTaskRequest. Path-specific additions (IRIS_WORKER_ID, IRIS_ADVERTISE_HOST)
    are layered on by each caller.

    All arguments are keyword-only primitives extracted from the proto so that
    callers from both paths can supply them without importing the full request.
    """
    env: dict[str, str] = {}

    # Task identity. Append :attempt_id for retries so tasks see the correct
    # attempt via JobInfo (matches TaskAttemptIdentity.to_wire()).
    wire_task_id = f"{task_id}:{attempt_id}" if attempt_id else task_id
    env["IRIS_TASK_ID"] = wire_task_id
    env["IRIS_NUM_TASKS"] = str(num_tasks)
    env["IRIS_BUNDLE_ID"] = bundle_id

    # Controller connectivity
    if controller_address:
        env["IRIS_CONTROLLER_ADDRESS"] = controller_address
        env["IRIS_CONTROLLER_URL"] = controller_address

    # Standard paths and binaries
    env["IRIS_BIND_HOST"] = "0.0.0.0"
    env["IRIS_WORKDIR"] = "/app"
    env["IRIS_PYTHON"] = "python"
    env["UV_PYTHON_INSTALL_DIR"] = "/uv/cache/python"
    env["CARGO_TARGET_DIR"] = "/root/.cargo/target"

    # Propagate extras and pip_packages so child jobs can inherit them
    extras = list(environment.extras)
    if extras:
        env["IRIS_JOB_EXTRAS"] = json.dumps(extras)
    pip_packages = list(environment.pip_packages)
    if pip_packages:
        env["IRIS_JOB_PIP_PACKAGES"] = json.dumps(pip_packages)

    # Serialize user env vars for child job inheritance via IRIS_JOB_ENV
    user_env_vars = dict(environment.env_vars)
    if user_env_vars:
        env["IRIS_JOB_ENV"] = json.dumps(user_env_vars)

    # Only propagate region/zone constraints to children; device constraints
    # are re-derived from each child's own resource spec.
    inheritable = [c for c in constraints if c.key in INHERITED_CONSTRAINT_KEYS]
    if inheritable:
        env["IRIS_JOB_CONSTRAINTS"] = json.dumps(
            [json_format.MessageToDict(c, preserving_proto_field_name=True) for c in inheritable]
        )

    # Ports: k8s sets "0" (kernel-assigned at runtime), worker path overrides
    # with real allocated ports after calling this function.
    for port_name in ports:
        env[f"IRIS_PORT_{port_name.upper()}"] = "0"

    # Device env vars (TPU/GPU platform selection)
    if resources is not None and resources.HasField("device"):
        dev = resources.device
        if dev.HasField("tpu"):
            env["JAX_PLATFORMS"] = "tpu,cpu"
            env["PJRT_DEVICE"] = "TPU"
            env["JAX_FORCE_TPU_INIT"] = "1"

    # Expose the task's resource limits so user code can query them via
    # iris.env_resources.TaskResources.from_environment() without relying
    # solely on cgroup introspection.
    # We serialize the proto directly; the reader deserializes it back.
    # Zero-valued fields are omitted by proto3 JSON so env_resources falls
    # back to cgroups / host values for unspecified dimensions.
    if resources is not None:
        resource_json = json_format.MessageToJson(resources, preserving_proto_field_name=True)
        # proto3 JSON omits zero-valued fields, so an empty proto yields "{}".
        if resource_json != "{}":
            env["IRIS_TASK_RESOURCES"] = resource_json

    return env
