# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture a completed agent workspace for an isolated verifier."""

import asyncio
import os
import re
import stat
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import cast

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment

from taskcompendium.submission import WORKSPACE_ROOT

MAX_WORKSPACE_ENTRIES = 4096
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
MAX_WORKSPACE_DEPTH = 128
DOCKER_COPY_TIMEOUT = 120
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


def _extract_workspace_archive(archive: tarfile.TarFile, destination: Path) -> None:
    """Write only bounded regular files and directories from Docker's tar stream."""
    entries = 0
    size = 0
    destination.mkdir(parents=True, exist_ok=False)
    for member in archive:
        if member.name.startswith("/") or "\\" in member.name or ".." in member.name.split("/"):
            raise UnsafeWorkspaceError("Workspace archive contains an unsafe path")
        parts = [part for part in member.name.split("/") if part not in ("", ".")]
        if not parts:
            continue
        entries += 1
        if entries > MAX_WORKSPACE_ENTRIES or len(parts) > MAX_WORKSPACE_DEPTH:
            raise UnsafeWorkspaceError("Workspace archive exceeds entry or depth limit")
        target = destination.joinpath(*parts)
        if not target.parent.is_dir() or target.exists():
            raise UnsafeWorkspaceError("Workspace archive contains an out-of-order or duplicate path")
        if member.isdir():
            target.mkdir(mode=0o777)
            target.chmod(0o777)
            continue
        if not member.isfile():
            raise UnsafeWorkspaceError("Workspace archive contains a link or special file")
        size += member.size
        if size > MAX_WORKSPACE_BYTES:
            raise UnsafeWorkspaceError("Workspace archive exceeds size limit")
        source = archive.extractfile(member)
        if source is None:
            raise UnsafeWorkspaceError("Workspace archive file is missing")
        with source, target.open("xb") as output:
            remaining = member.size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise UnsafeWorkspaceError("Workspace archive file is incomplete")
                output.write(chunk)
                remaining -= len(chunk)
        target.chmod(0o777 if member.mode & 0o111 else 0o666)


def _bounded_docker_copy(container_id: str, destination: Path) -> None:
    """Stream a container workspace into a size-limited host directory."""
    command = ["docker", "cp", f"{container_id}:{WORKSPACE_ROOT}/.", "-"]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as error:
        raise UnsafeWorkspaceError("Cannot start Docker workspace copy") from error
    timer = threading.Timer(DOCKER_COPY_TIMEOUT, process.kill)
    timer.start()
    try:
        assert process.stdout is not None
        with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            _extract_workspace_archive(archive, destination)
        if process.wait(timeout=5) != 0:
            raise UnsafeWorkspaceError("Docker workspace copy failed")
    except (OSError, tarfile.TarError, subprocess.TimeoutExpired) as error:
        raise UnsafeWorkspaceError("Cannot read Docker workspace archive") from error
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


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
    container = await cast(DockerEnvironment, environment)._run_docker_compose_command(
        ["ps", "-q", "main"], timeout_sec=10
    )
    if container.return_code != 0 or container.stdout is None or container.stdout_truncated:
        raise UnsafeWorkspaceError("Cannot identify agent container")
    container_id = container.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise UnsafeWorkspaceError("Agent container ID is invalid")
    await asyncio.to_thread(_bounded_docker_copy, container_id, destination)
    validate_workspace_snapshot(destination)
    return destination
