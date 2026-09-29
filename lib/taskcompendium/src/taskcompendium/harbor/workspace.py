# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture a completed agent workspace for an isolated verifier."""

import os
import re
import stat
from pathlib import Path

from harbor.environments.base import BaseEnvironment

from taskcompendium.submission import WORKSPACE_ROOT

MAX_WORKSPACE_ENTRIES = 4096
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
MAX_WORKSPACE_DEPTH = 128
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
    entries = 0
    size = 0
    pending = [(root, 0)]
    try:
        while pending:
            current, depth = pending.pop()
            with os.scandir(current) as directory:
                for entry in directory:
                    path = Path(entry.path)
                    metadata = path.lstat()
                    if metadata.st_dev != device or not (
                        stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
                    ):
                        raise UnsafeWorkspaceError(f"Unsafe workspace entry: {path.relative_to(root)}")
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        raise UnsafeWorkspaceError("Workspace snapshot exceeds entry limit")
                    if stat.S_ISDIR(metadata.st_mode):
                        if depth >= MAX_WORKSPACE_DEPTH:
                            raise UnsafeWorkspaceError("Workspace snapshot exceeds depth limit")
                        pending.append((path, depth + 1))
                    else:
                        size += metadata.st_size
                        if size > MAX_WORKSPACE_BYTES:
                            raise UnsafeWorkspaceError("Workspace snapshot exceeds size limit")
    except OSError as error:
        raise UnsafeWorkspaceError("Cannot inspect workspace snapshot") from error


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
