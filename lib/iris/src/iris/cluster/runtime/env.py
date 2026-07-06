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
from iris.cluster.tpu_topology import get_tpu_topology
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

IRIS_SLICE_COUNT = "IRIS_SLICE_COUNT"
IRIS_TASKS_PER_SLICE = "IRIS_TASKS_PER_SLICE"

# Container paths shared across runtimes: the bundle unpacks into WORKDIR_PATH and
# the setup script populates the venv at VENV_PATH (which the run phase activates).
WORKDIR_PATH = "/app"
VENV_PATH = f"{WORKDIR_PATH}/.venv"

# Heredoc delimiter for materializing a setup script to disk. Distinctive enough
# that a real setup script will not contain it as a standalone line.
_SETUP_STEP_DELIMITER = "__IRIS_SETUP_STEP__"


def render_setup_steps(scripts: Sequence[str]) -> list[str]:
    """Render bash lines that run each setup script as a separate step.

    Each script is written to its own file and run in a fresh ``bash`` with a
    banner, rather than concatenated, so a failure points at the exact step. The
    caller's ``set -e`` stops the sequence on the first non-zero step.
    """
    lines: list[str] = []
    total = len(scripts)
    for index, script in enumerate(scripts, start=1):
        step_file = f"/tmp/iris-setup-step-{index}.sh"
        lines.append(f"cat > {step_file} <<'{_SETUP_STEP_DELIMITER}'")
        lines.append(script.rstrip("\n"))
        lines.append(_SETUP_STEP_DELIMITER)
        lines.append(f'echo "[iris setup] step {index}/{total}"')
        lines.append(f"bash {step_file}")
    return lines


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


def slice_topology_env(resources: job_pb2.ResourceSpecProto | None, num_tasks: int) -> dict[str, str]:
    if num_tasks <= 1 or resources is None or not resources.HasField("device") or not resources.device.HasField("tpu"):
        return {}

    tpu = resources.device.tpu
    if not tpu.variant:
        return {}

    try:
        tasks_per_slice = get_tpu_topology(tpu.variant).vm_count
    except ValueError:
        return {}

    if tasks_per_slice <= 0:
        return {}
    if num_tasks % tasks_per_slice != 0:
        raise ValueError(
            f"TPU task count ({num_tasks}) must be divisible by TPU VM count ({tasks_per_slice}) "
            f"for variant {tpu.variant!r}"
        )

    num_slices = num_tasks // tasks_per_slice
    if num_slices <= 1:
        return {}

    return {IRIS_SLICE_COUNT: str(num_slices), IRIS_TASKS_PER_SLICE: str(tasks_per_slice)}


def with_slice_topology_env(
    environment: job_pb2.EnvironmentConfig,
    resources: job_pb2.ResourceSpecProto | None,
    num_tasks: int,
) -> job_pb2.EnvironmentConfig:
    """Return ``environment`` with Iris TPU slice topology vars derived from resources."""
    result = job_pb2.EnvironmentConfig()
    result.CopyFrom(environment)
    result.env_vars.pop(IRIS_SLICE_COUNT, None)
    result.env_vars.pop(IRIS_TASKS_PER_SLICE, None)
    result.env_vars.update(slice_topology_env(resources, num_tasks))
    return result


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

    # Task identity in canonical TaskAttempt wire form, always carrying the
    # attempt suffix (/user/job/0:0 for the first attempt) so the id — and the
    # finelog log key derived from it — is identical across attempts and every
    # backend.
    wire_task_id = f"{task_id}:{attempt_id}"
    env["IRIS_TASK_ID"] = wire_task_id
    env["IRIS_NUM_TASKS"] = str(num_tasks)
    env["IRIS_BUNDLE_ID"] = bundle_id

    # Controller connectivity
    if controller_address:
        env["IRIS_CONTROLLER_ADDRESS"] = controller_address
        env["IRIS_CONTROLLER_URL"] = controller_address

    # Standard paths and binaries
    env["IRIS_BIND_HOST"] = "0.0.0.0"
    env["IRIS_WORKDIR"] = WORKDIR_PATH
    env["IRIS_PYTHON"] = "python"
    # Canonical venv the setup script populates and the run phase activates.
    # UV_PROJECT_ENVIRONMENT points uv (sync/pip install) at the same path so a
    # custom setup script does not have to depend on uv's cwd-relative default.
    env["IRIS_VENV"] = VENV_PATH
    env["UV_PROJECT_ENVIRONMENT"] = VENV_PATH
    env["UV_PYTHON_INSTALL_DIR"] = "/uv/cache/python"
    env["CARGO_TARGET_DIR"] = "/root/.cargo/target"

    # Propagate the resolved setup scripts so child jobs reproduce the parent's
    # environment. Always set (even when empty) so a child can tell a no-setup
    # parent (bring-your-own image) from a top-level submission with no parent.
    env["IRIS_JOB_SETUP_SCRIPTS"] = json.dumps(list(environment.setup_scripts))

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
            env.update(slice_topology_env(resources, num_tasks))

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
