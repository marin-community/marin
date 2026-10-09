# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker build contexts for task images, and the step that pins one as a registry image.

A container task names its image only as a digest-pinned registry reference
(``EnvironmentRequirements.docker_image``); both the laptop Docker backend (through Skopeo) and the
Iris gVisor backend pull it. A ``DockerBuild`` is Taskforge's own build input and never part of a
TaskSpec: an ``ImageBuilder`` builds the context for linux/amd64, pushes it, and returns the
reference pinned by manifest digest. ``scripts/build_image_job.py`` implements the builder as Iris jobs.
"""

import gzip
import hashlib
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taskcompendium.models import DOCKER_IMAGE_PATTERN, TaskResource
from taskcompendium.runtime.resources import inline_resource, resource_bytes

# The verifier machine of a script grader on a task without its own image: docker/grader-base built for
# linux/amd64 and pinned by manifest digest. Publish it with
#   uv run scripts/build_image_job.py --context docker/grader-base --repository capability-infra/taskforge-grader-base
# and put the printed reference here.
GRADER_BASE_IMAGE = (
    "envreg.208261-marin-gpu.coreweave.app/capability-infra/taskforge-grader-base"
    "@sha256:63eb0b5c3efb5b2e3def1e81f9e076a4c62a077ad5f631cfef95ea9f441240b3"
)
MANIFEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# Repository path components per the OCI distribution spec.
REPOSITORY = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*")
DEFAULT_MODE = "644"


@dataclass(frozen=True)
class DockerBuild:
    """A Dockerfile and its build context: the ImageBuilder's input, never part of a TaskSpec.

    ``files`` paths are relative to the context root (``Dockerfile``, ``app/run.sh``); ``dockerfile``
    names one of them.
    """

    files: tuple[TaskResource, ...]
    dockerfile: str = "Dockerfile"

    def __post_init__(self) -> None:
        paths = [resource.path for resource in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"build context paths repeat: {sorted(paths)}")
        if self.dockerfile not in paths:
            raise ValueError(f"build context has no {self.dockerfile!r}")


class BuildTooLarge(ValueError):
    """A build context exceeds the limits a builder accepts."""


@dataclass(frozen=True)
class BuildLimits:
    """Bounds on a build context; the Iris builder ships the context inside the job spec."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int


# The Iris controller offloads job files over 10 KiB to its blob store; keep a whole context well
# under the 25 MiB workspace-bundle ceiling.
BUILD_LIMITS = BuildLimits(max_files=2048, max_file_bytes=8 * 1024 * 1024, max_total_bytes=16 * 1024 * 1024)


def check_build_limits(build: DockerBuild, limits: BuildLimits) -> None:
    """Raise ``BuildTooLarge`` naming the first limit the context exceeds."""
    if len(build.files) > limits.max_files:
        raise BuildTooLarge(f"build context has {len(build.files)} files; the limit is {limits.max_files}")
    total = 0
    for resource in build.files:
        size = len(resource_bytes(resource))
        if size > limits.max_file_bytes:
            raise BuildTooLarge(f"{resource.path} is {size} bytes; the limit is {limits.max_file_bytes}")
        total += size
    if total > limits.max_total_bytes:
        raise BuildTooLarge(f"build context is {total} bytes; the limit is {limits.max_total_bytes}")


def docker_build_from_directory(directory: Path, limits: BuildLimits) -> DockerBuild:
    """Read every regular file under ``directory`` into a ``DockerBuild`` with the default Dockerfile.

    Symlinks and other special files are refused: a context carries bytes and modes only.
    """
    root = directory.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"build context may hold only regular files and directories: {path}")
        if path.is_dir():
            continue
        resource = inline_resource(path.relative_to(root).as_posix(), path.read_bytes())
        files.append(resource.model_copy(update={"mode": f"{path.stat().st_mode & 0o777:o}"}))
    build = DockerBuild(files=tuple(files))
    check_build_limits(build, limits)
    return build


def _ordered(build: DockerBuild) -> list[TaskResource]:
    return sorted(build.files, key=lambda resource: resource.path)


def build_digest(build: DockerBuild) -> str:
    """sha256 over the Dockerfile path and every file's path, mode and bytes, in path order."""
    hasher = hashlib.sha256()
    hasher.update(f"dockerfile\0{build.dockerfile}\0".encode())
    for resource in _ordered(build):
        content = resource_bytes(resource)
        hasher.update(f"file\0{resource.path}\0{resource.mode or DEFAULT_MODE}\0{len(content)}\0".encode())
        hasher.update(content)
    return f"sha256:{hasher.hexdigest()}"


def build_context_archive(build: DockerBuild) -> bytes:
    """A gzipped tar of the context, byte-identical for identical builds (fixed owner and times)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for resource in _ordered(build):
            content = resource_bytes(resource)
            info = tarfile.TarInfo(resource.path)
            info.size = len(content)
            info.mode = int(resource.mode or DEFAULT_MODE, 8)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(raw.getvalue(), mtime=0)


def pinned_image(registry: str, repository: str, manifest_digest: str) -> str:
    """``registry/repository@digest``: the ``docker_image`` reference a backend pulls."""
    if not REPOSITORY.fullmatch(repository):
        raise ValueError(f"not an OCI repository name: {repository!r}")
    if not MANIFEST_DIGEST.fullmatch(manifest_digest):
        raise ValueError(f"manifest digest must be sha256:<64 hex>, got {manifest_digest!r}")
    reference = f"{registry}/{repository}@{manifest_digest}"
    if not re.fullmatch(DOCKER_IMAGE_PATTERN, reference):
        raise ValueError(f"not a digest-pinned image reference: {reference!r}")
    return reference


class ImageBuilder(Protocol):
    """Build a context for linux/amd64, push it under ``repository``, and return it pinned by digest."""

    async def publish(self, build: DockerBuild, repository: str) -> str: ...
