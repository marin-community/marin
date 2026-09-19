# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, link-free snapshots of a native Docker workspace."""

import asyncio
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MAX_SNAPSHOT_BYTES = 268435456
MAX_SNAPSHOT_ENTRIES = 100000
SNAPSHOT_TIMEOUT = 120


def extract_snapshot(archive_path: Path, destination: Path, excluded_paths: tuple[str, ...]) -> None:
    """Extract only regular files and directories within the snapshot budget."""
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(archive_path, mode="r:") as archive:
        for index, member in enumerate(archive):
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if member.isdir() and name in {"", "."}:
                continue
            if (
                index >= MAX_SNAPSHOT_ENTRIES
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or not (member.isfile() or member.isdir())
                or member.issym()
                or member.islnk()
                or any(path == PurePosixPath(excluded) or path.is_relative_to(excluded) for excluded in excluded_paths)
            ):
                raise ValueError(f"Unsafe workspace archive member: {member.name}")
            total += member.size
            if total > MAX_SNAPSHOT_BYTES:
                raise ValueError("Workspace snapshot exceeds its byte budget")
            target = destination / path
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError("Workspace archive escapes its destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


async def download_snapshot(container_id: str, workdir: str, destination: Path, excluded_paths: tuple[str, ...]) -> None:
    """Stream a size-limited archive from Docker; never execute candidate files on the host."""
    command = ["docker", "exec", "--user", "root", "--workdir", "/", container_id, "/bin/tar", "-C", workdir]
    command.extend(["--no-wildcards", *[f"--exclude=./{path}" for path in excluded_paths], "-cf", "-", "."])
    with tempfile.TemporaryDirectory(prefix="taskcompendium-snapshot-") as temporary:
        archive_path = Path(temporary) / "workspace.tar"
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL
        )
        assert process.stdout is not None
        try:
            async with asyncio.timeout(SNAPSHOT_TIMEOUT):
                total = 0
                with archive_path.open("wb") as archive:
                    while data := await process.stdout.read(65536):
                        total += len(data)
                        if total > MAX_SNAPSHOT_BYTES:
                            raise ValueError("Workspace archive exceeds its byte budget")
                        archive.write(data)
                if await process.wait() != 0:
                    raise RuntimeError("Docker workspace export failed")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        extract_snapshot(archive_path, destination, excluded_paths)
