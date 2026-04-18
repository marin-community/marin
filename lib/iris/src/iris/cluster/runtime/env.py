# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment helpers for container runtimes.

Includes device environment variable construction and workdir file writing.
Used by Docker, process, and Kubernetes runtimes.
"""

import logging
import posixpath
from pathlib import Path

from iris.cluster.runtime.types import ContainerConfig

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
