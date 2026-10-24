# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-attempt secret file lifecycle for VM tasks."""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

from rigging.secrets import COREWEAVE_SECRET_NAMES, secret_file_contents

from iris.rpc import job_pb2

WORKER_SECRETS_ROOT = Path(f"/dev/shm/iris-secrets-{os.getuid()}")
VM_WORKER_SECRETS_ROOT = Path("/dev/shm/iris-secrets-0")


def task_secret_values(environment: job_pb2.EnvironmentConfig) -> dict[str, str]:
    """Include explicit legacy storage env entries in native credential files.

    Submitter defaults arrive in the secrets map. Explicit CW env overrides
    retain their existing precedence for storage clients and output uploaders.
    """
    return {
        **environment.secrets,
        **{name: environment.env_vars[name] for name in COREWEAVE_SECRET_NAMES if name in environment.env_vars},
    }


def worker_secret_root(cache_dir: Path) -> Path:
    """Separate workers sharing a host while retaining paths across restart."""
    worker_scope = hashlib.sha256(str(cache_dir.resolve()).encode()).hexdigest()[:16]
    return WORKER_SECRETS_ROOT / worker_scope


def write_task_secrets(values: Mapping[str, str], directory: Path, *, uid: int, gid: int) -> None:
    """Create owner-readable files on host tmpfs.

    The caller owns directory cleanup, including partial writes on failure.
    The directory must be outside bundles, caches and captured task outputs.
    """
    files = secret_file_contents(values)
    directory.mkdir(mode=0o700)
    for name, value in files.items():
        path = directory / name
        with open(path, "x", encoding="utf-8", opener=lambda path, flags: os.open(path, flags, 0o400)) as handle:
            handle.write(value)
        os.chown(path, uid, gid)
    os.chown(directory, uid, gid)
