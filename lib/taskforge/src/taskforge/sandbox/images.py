# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker build contexts for task environments, and the step that pins one as a registry image.

A task's docker environment is a TaskCompendium ``DockerBuild`` (a Dockerfile plus its context,
inline in the TaskSpec) or a ``RegistryImage``. Local Docker builds a ``DockerBuild`` itself;
the Iris machine backend accepts only registry references, so a task bound for Iris is published
first: an ``ImageBuilder`` builds the context for linux/amd64, pushes it, and returns the image
pinned by manifest digest. ``scripts/build_image_job.py`` implements the builder as an Iris job.
"""

import gzip
import hashlib
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from taskcompendium.environment import DockerBuild, EnvironmentFile, RegistryImage

MANIFEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# Repository path components per the OCI distribution spec.
REPOSITORY = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*")


class BuildTooLarge(ValueError):
    """A build context exceeds the limits a builder accepts."""


@dataclass(frozen=True)
class BuildLimits:
    """Bounds on a build context; the Iris builder ships the context inside the job spec."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int


# The Iris controller offloads job files over 10 KiB to its blob store; keep a whole context well
# under the 25 MiB workspace-bundle ceiling so a task's TaskSpec JSON stays reviewable.
BUILD_LIMITS = BuildLimits(max_files=2048, max_file_bytes=8 * 1024 * 1024, max_total_bytes=16 * 1024 * 1024)


def check_build_limits(build: DockerBuild, limits: BuildLimits) -> None:
    """Raise ``BuildTooLarge`` naming the first limit the context exceeds."""
    if len(build.files) > limits.max_files:
        raise BuildTooLarge(f"build context has {len(build.files)} files; the limit is {limits.max_files}")
    for file in build.files:
        if len(file.content) > limits.max_file_bytes:
            raise BuildTooLarge(f"{file.path} is {len(file.content)} bytes; the limit is {limits.max_file_bytes}")
    total = sum(len(file.content) for file in build.files)
    if total > limits.max_total_bytes:
        raise BuildTooLarge(f"build context is {total} bytes; the limit is {limits.max_total_bytes}")


def docker_build_from_directory(directory: Path, limits: BuildLimits) -> DockerBuild:
    """Read every regular file under ``directory`` into a ``DockerBuild`` rooted at ``/``.

    The Dockerfile path is ``DockerBuild``'s default. Symlinks and other special files are refused:
    a TaskSpec context carries bytes and modes only.
    """
    root = directory.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"build context may hold only regular files and directories: {path}")
        if path.is_dir():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        files.append(EnvironmentFile(path=f"/{relative}", content=path.read_bytes(), mode=path.stat().st_mode & 0o777))
    build = DockerBuild(files=tuple(files))
    check_build_limits(build, limits)
    return build


def build_digest(build: DockerBuild) -> str:
    """sha256 over the Dockerfile path and every file's path, mode and bytes, in path order."""
    hasher = hashlib.sha256()
    hasher.update(f"dockerfile\0{build.dockerfile}\0".encode())
    for file in sorted(build.files, key=lambda f: f.path):
        hasher.update(f"file\0{file.path}\0{file.mode:o}\0{len(file.content)}\0".encode())
        hasher.update(file.content)
    return f"sha256:{hasher.hexdigest()}"


def build_context_archive(build: DockerBuild) -> bytes:
    """A gzipped tar of the context, byte-identical for identical builds (fixed owner and times)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for file in sorted(build.files, key=lambda f: f.path):
            info = tarfile.TarInfo(file.path.lstrip("/"))
            info.size = len(file.content)
            info.mode = file.mode
            archive.addfile(info, io.BytesIO(file.content))
    return gzip.compress(raw.getvalue(), mtime=0)


def pinned_image(registry: str, repository: str, manifest_digest: str) -> RegistryImage:
    """``registry/repository@digest`` as the TaskSpec image a backend pulls."""
    if not REPOSITORY.fullmatch(repository):
        raise ValueError(f"not an OCI repository name: {repository!r}")
    if not MANIFEST_DIGEST.fullmatch(manifest_digest):
        raise ValueError(f"manifest digest must be sha256:<64 hex>, got {manifest_digest!r}")
    return RegistryImage(reference=f"{registry}/{repository}@{manifest_digest}")


class ImageBuilder(Protocol):
    """Build a context for linux/amd64, push it under ``repository``, and return it pinned by digest."""

    async def publish(self, build: DockerBuild, repository: str) -> RegistryImage: ...
