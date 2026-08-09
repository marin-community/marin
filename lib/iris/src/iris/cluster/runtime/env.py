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
from dataclasses import replace
from pathlib import Path

from iris.cluster.constraints import (
    INHERITED_CONSTRAINT_KEYS,
    AttributeValue,
    Constraint,
    ConstraintOp,
)
from iris.cluster.runtime.types import MountKind, MountSpec
from iris.cluster.tpu_topology import get_tpu_topology
from iris.resources.execution import (
    IRIS_SLICE_COUNT,
    IRIS_TASKS_PER_SLICE,
    CpuDevice,
    Environment,
    GpuDevice,
    ResourceSpec,
    TpuDevice,
)

logger = logging.getLogger(__name__)

IRIS_NODE_NAME_ENV = "IRIS_NODE_NAME"
IRIS_NAMESPACE_ENV = "IRIS_NAMESPACE"

# Container paths shared across runtimes: the bundle unpacks into WORKDIR_PATH and
# the setup script populates the venv at VENV_PATH (which the run phase activates).
WORKDIR_PATH = "/app"
VENV_PATH = f"{WORKDIR_PATH}/.venv"

# Download caches, bound to node-local storage that outlives the container
# (hostPath on K8s, cache_dir on Docker) so a wheel or a model is fetched once
# per node instead of once per task. Paths sit outside $HOME because a task may
# bring its own image: build_common_iris_env points each tool here explicitly, so
# nothing depends on that image's HOME.
UV_CACHE_PATH = "/uv/cache"
HF_HUB_CACHE_PATH = "/hf/cache"
CARGO_HOME_PATH = "/cargo"
# Unclaimed node-local scratch, for anything that needs a real directory on the
# node rather than a bucket. Tasks pick their own subdirectory; nothing prunes
# it. `iris.runtime.jax_init` puts XLA's per-fusion autotune cache under
# `/cache/xla` because XLA opens that directory from C++ through `tsl::Env`,
# which has no object-store filesystem.
SCRATCH_CACHE_PATH = "/cache"

# The task container filesystem, as mounted by every runtime. Each runtime binds
# a CACHE entry to cache_host_dirname(path) under its own cache_dir, so one node
# directory backs a task whether it lands as a K8s pod or a Docker container.
#
# This list and the cache env in build_common_iris_env are one contract: the env
# names these exact paths, so both must be defined together. A cache whose env
# var and mount disagree still runs -- it just writes to the container's own
# writable layer and re-downloads on every task, with nothing to see in a log.
WORKDIR_MOUNT = MountSpec("workdir", WORKDIR_PATH, kind=MountKind.WORKDIR)

STANDARD_MOUNTS: tuple[MountSpec, ...] = (
    WORKDIR_MOUNT,
    MountSpec("tmpfs", "/tmp", kind=MountKind.TMPFS),
    MountSpec("uv-cache", UV_CACHE_PATH, kind=MountKind.CACHE),
    MountSpec("hf-cache", HF_HUB_CACHE_PATH, kind=MountKind.CACHE),
    MountSpec("cargo", CARGO_HOME_PATH, kind=MountKind.CACHE),
    MountSpec("scratch-cache", SCRATCH_CACHE_PATH, kind=MountKind.CACHE),
)


def cache_host_dirname(container_path: str) -> str:
    """Host directory name for a CACHE mount, relative to a runtime's cache_dir."""
    return container_path.strip("/").replace("/", "-")


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


def slice_topology_env(resources: ResourceSpec | None, num_tasks: int) -> dict[str, str]:
    if num_tasks <= 1 or resources is None or not isinstance(resources.device, TpuDevice):
        return {}

    tpu = resources.device
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
    environment: Environment,
    resources: ResourceSpec | None,
    num_tasks: int,
) -> Environment:
    """Return ``environment`` with Iris TPU slice topology vars derived from resources."""
    env_vars = dict(environment.env_vars)
    env_vars.pop(IRIS_SLICE_COUNT, None)
    env_vars.pop(IRIS_TASKS_PER_SLICE, None)
    env_vars.update(slice_topology_env(resources, num_tasks))
    return replace(environment, env_vars=env_vars)


def _attribute_value_proto_json(value: AttributeValue) -> dict[str, object]:
    if isinstance(value.value, str):
        return {"string_value": value.value}
    if isinstance(value.value, int):
        return {"int_value": str(value.value)}
    return {"float_value": value.value}


def _constraint_proto_json(value: Constraint) -> dict[str, object]:
    result: dict[str, object] = {"key": value.key, "op": f"CONSTRAINT_OP_{value.op.name}"}
    if value.op is ConstraintOp.IN:
        result["values"] = [_attribute_value_proto_json(item) for item in value.values]
    elif value.values:
        result["value"] = _attribute_value_proto_json(value.values[0])
    result["mode"] = f"CONSTRAINT_MODE_{value.mode.name}"
    return result


def _resource_proto_json(value: ResourceSpec) -> str:
    """Serialize native resources in the established ResourceSpecProto JSON shape."""
    result: dict[str, object] = {}
    if value.cpu_millicores:
        result["cpu_millicores"] = value.cpu_millicores
    if value.memory:
        result["memory_bytes"] = str(value.memory)
    device = value.device
    if isinstance(device, CpuDevice):
        result["device"] = {"cpu": {"variant": device.variant}}
    elif isinstance(device, GpuDevice):
        result["device"] = {"gpu": {"variant": device.variant, "count": device.count}}
    elif isinstance(device, TpuDevice):
        tpu: dict[str, object] = {"variant": device.variant}
        if device.topology:
            tpu["topology"] = device.topology
        if device.count:
            tpu["count"] = device.count
        result["device"] = {"tpu": tpu}
    if value.disk:
        result["disk_bytes"] = str(value.disk)
    return json.dumps(result, indent=2)


def build_common_iris_env(
    *,
    task_id: str,
    attempt_id: int,
    num_tasks: int,
    bundle_id: str,
    controller_address: str | None,
    environment: Environment,
    constraints: Sequence[Constraint],
    ports: Sequence[str],
    resources: ResourceSpec | None,
) -> dict[str, str]:
    """Build the Iris system env vars shared by both worker and k8s paths.

    This is the single source of truth for env vars derived from a
    RunTaskRequest. Path-specific additions (IRIS_WORKER_ID, IRIS_ADVERTISE_HOST)
    are layered on by each caller.

    All arguments are keyword-only native values so runtimes share one launch
    contract without importing the retired ControllerService messages.
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
    # Point each tool at its STANDARD_MOUNTS cache. Set here rather than in the
    # task image so a task running its own image still hits the shared caches.
    # HF_HOME is left alone on purpose: it holds the submitter's HF_TOKEN, which
    # must not land on a node directory every other task can read. HF_HUB_CACHE
    # covers the part worth sharing -- the content-addressed model/dataset blobs.
    env["UV_CACHE_DIR"] = UV_CACHE_PATH
    env["UV_PYTHON_INSTALL_DIR"] = f"{UV_CACHE_PATH}/python"
    env["HF_HUB_CACHE"] = HF_HUB_CACHE_PATH
    # CARGO_HOME moves the crate registry onto the mount; a rustup toolchain
    # installed elsewhere still resolves, since PATH finds the binary.
    env["CARGO_HOME"] = CARGO_HOME_PATH
    env["CARGO_TARGET_DIR"] = f"{CARGO_HOME_PATH}/target"

    # Propagate the resolved setup scripts so child jobs reproduce the parent's
    # environment. Always set (even when empty) so a child can tell a no-setup
    # parent (bring-your-own image) from a top-level submission with no parent.
    env["IRIS_JOB_SETUP_SCRIPTS"] = json.dumps(list(environment.setup_scripts))

    # Serialize user env vars for child job inheritance via IRIS_JOB_ENV
    user_env_vars = dict(sorted(environment.env_vars.items()))
    if user_env_vars:
        env["IRIS_JOB_ENV"] = json.dumps(user_env_vars)

    # Only propagate region/zone constraints to children; device constraints
    # are re-derived from each child's own resource spec.
    inheritable = [c for c in constraints if c.key in INHERITED_CONSTRAINT_KEYS]
    if inheritable:
        env["IRIS_JOB_CONSTRAINTS"] = json.dumps([_constraint_proto_json(c) for c in inheritable])

    # Ports: k8s sets "0" (kernel-assigned at runtime), worker path overrides
    # with real allocated ports after calling this function.
    for port_name in ports:
        env[f"IRIS_PORT_{port_name.upper()}"] = "0"

    # Device env vars (TPU/GPU platform selection)
    if resources is not None and isinstance(resources.device, TpuDevice):
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
        env["IRIS_TASK_RESOURCES"] = _resource_proto_json(resources)

    return env
