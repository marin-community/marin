# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared bundle staging logic for local-filesystem runtimes (Docker, process)."""

import shutil
from collections.abc import Callable
from pathlib import Path


def stage_bundle_to_local(
    *,
    bundle_gcs_path: str,
    workdir: Path,
    workdir_files: dict[str, bytes],
    fetch_bundle: Callable[[str], Path],
) -> None:
    """Fetch a task bundle and materialize it plus workdir files on the local filesystem.

    Used by DockerRuntime and ProcessRuntime, which both execute from
    worker-local paths. KubernetesRuntime materializes in-pod instead.
    """
    bundle_path = fetch_bundle(bundle_gcs_path)
    shutil.copytree(bundle_path, workdir, dirs_exist_ok=True)
    for name, data in workdir_files.items():
        (workdir / name).write_bytes(data)
