# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture a completed agent workspace for an isolated verifier."""

import os
import re
import stat
from pathlib import Path

from harbor.environments.base import BaseEnvironment

WORKSPACE_ROOT = "/app"
MAX_WORKSPACE_FILES = 4096
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


class UnsafeWorkspaceError(ValueError):
    """The final workspace cannot be copied without exposing outside state."""


def _mount_path(value: str) -> str:
    return MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def validate_workspace_snapshot(root: Path) -> None:
    """Reject links, special files, nested mounts, and oversized snapshots."""
    if root.is_symlink() or not root.is_dir():
        raise UnsafeWorkspaceError("Workspace snapshot root is not a directory")
    device = root.stat().st_dev
    count = 0
    size = 0
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            metadata = path.lstat()
            if metadata.st_dev != device or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise UnsafeWorkspaceError(f"Unsafe workspace entry: {path.relative_to(root)}")
            if stat.S_ISREG(metadata.st_mode):
                count += 1
                size += metadata.st_size
                if count > MAX_WORKSPACE_FILES or size > MAX_WORKSPACE_BYTES:
                    raise UnsafeWorkspaceError("Workspace snapshot exceeds file or size limit")


async def capture_workspace(environment: BaseEnvironment, destination: Path) -> Path:
    """Copy `/app` after the agent phase and validate the verifier-side copy."""
    root = await environment.exec("test -d /app && test ! -L /app")
    if root.return_code != 0:
        raise UnsafeWorkspaceError("Agent workspace root is missing or is a symlink")
    mounts = await environment.exec("cat /proc/self/mountinfo")
    if mounts.return_code != 0 or mounts.stdout is None or mounts.stdout_truncated:
        raise UnsafeWorkspaceError("Cannot inspect workspace mount boundaries")
    for line in mounts.stdout.splitlines():
        fields = line.split(" ")
        if len(fields) < 5:
            raise UnsafeWorkspaceError("Invalid workspace mount information")
        mount = _mount_path(fields[4])
        if mount.startswith(f"{WORKSPACE_ROOT}/"):
            raise UnsafeWorkspaceError(f"Workspace contains a nested mount: {mount}")
    destination.mkdir(parents=True, exist_ok=False)
    await environment.download_dir(WORKSPACE_ROOT, destination)
    validate_workspace_snapshot(destination)
    return destination
