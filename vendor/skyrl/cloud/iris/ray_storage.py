"""Ray object-store paths shared by the Iris launcher and node controller."""

import json
import os
import shlex
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

DEFAULT_RAY_SPILL_DIR = "/tmp/skyrl-ray-spill"
RAY_SPILL_BUFFER_SIZE = 100 * 1024 * 1024
_RAY_LOCAL_SPILL_FLAG = "--object-spilling-directory"


class RaySpillBackend(StrEnum):
    LOCAL = "local"
    R2 = "r2"


class RaySpillTarget(Protocol):
    location: str

    def prepare_node(self) -> None: ...

    def shell_preflight(self) -> str:
        """Return a semicolon-terminated shell guard, or an empty string when none is needed."""
        ...

    def head_flags(self) -> list[str]: ...

    def worker_flags(self) -> list[str]: ...

    def description(self) -> str: ...


@dataclass(frozen=True)
class LocalRaySpillTarget:
    location: str

    def __post_init__(self) -> None:
        validate_ray_spill_dir(self.location)

    def prepare_node(self) -> None:
        try:
            os.makedirs(self.location, exist_ok=True)
        except OSError as error:
            raise RuntimeError(f"Could not create Ray spill directory {self.location!r}: {error}") from error

    def shell_preflight(self) -> str:
        # This must fail before the controller imports MarinSkyRL. prepare_node repeats
        # the contract at Ray's immediate subprocess boundary.
        quoted_path = shlex.quote(self.location)
        error_message = shlex.quote(
            f"[rl-iris] Could not prepare local Ray spill directory {self.location!r} before controller startup"
        )
        return f"if ! mkdir -p -- {quoted_path} || [ ! -d {quoted_path} ]; then echo {error_message} >&2; exit 1; fi; "

    def head_flags(self) -> list[str]:
        return [f"{_RAY_LOCAL_SPILL_FLAG}={self.location}"]

    def worker_flags(self) -> list[str]:
        return [f"{_RAY_LOCAL_SPILL_FLAG}={self.location}"]

    def description(self) -> str:
        return f"Ray object spilling -> launcher-owned local scratch {self.location}"


@dataclass(frozen=True)
class R2RaySpillTarget:
    location: str

    def __post_init__(self) -> None:
        if not self.location.startswith("s3://"):
            raise ValueError(f"R2 Ray spill target must use s3://, got {self.location!r}")

    def prepare_node(self) -> None:
        # Ray's smart_open spill backend imports boto3 inside each node process.
        try:
            import boto3  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "--ray-spill-backend=r2 requires boto3 in the locked runtime; use local spilling or fix the profile"
            ) from error

    def shell_preflight(self) -> str:
        return ""

    def head_flags(self) -> list[str]:
        spilling_config = json.dumps(
            {
                "type": "smart_open",
                "params": {"uri": self.location, "buffer_size": RAY_SPILL_BUFFER_SIZE},
            }
        )
        system_config = json.dumps({"object_spilling_config": spilling_config, "min_spilling_size": 0})
        return [f"--system-config={system_config}"]

    def worker_flags(self) -> list[str]:
        # The head's remote system config propagates through GCS. Ray rejects the
        # same system-config flag on a worker joining an existing cluster.
        return []

    def description(self) -> str:
        return f"Ray object spilling -> R2 prefix {self.location} (remote backend)"


def validate_ray_spill_dir(path: str) -> str:
    """Validate a node-local path without interpreting it on the launch host."""
    if not os.path.isabs(path) or "://" in path:
        raise ValueError(f"Ray spill directory must be an absolute local path, got {path!r}")
    return path


def resolve_ray_spill_target(
    rendezvous_dir: str | None,
    backend: RaySpillBackend,
    local_spill_dir: str,
) -> RaySpillTarget:
    """Resolve one valid local or remote Ray spill target."""
    if backend is RaySpillBackend.LOCAL:
        return LocalRaySpillTarget(location=validate_ray_spill_dir(local_spill_dir))
    if local_spill_dir != DEFAULT_RAY_SPILL_DIR:
        raise ValueError("--ray-spill-dir only applies to --ray-spill-backend=local")
    if not rendezvous_dir or not rendezvous_dir.startswith("s3://"):
        raise ValueError("--ray-spill-backend=r2 requires an s3:// rendezvous directory")
    return R2RaySpillTarget(location=f"{rendezvous_dir.rstrip('/')}/ray_spill")
