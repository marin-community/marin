# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package files as inline resources, and container images as pinned references or build contexts."""

import base64
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.importers.harbor.config import UnsupportedHarborFeature
from taskcompendium.importers.harbor.dockerfile import (
    PINNED_IMAGE,
    dockerfile_images,
    dockerfile_workdir,
    pin_dockerfile,
)
from taskcompendium.models import DOCKERFILE, DockerBuildContext, InlineFile, TaskResource

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

type Pin = Callable[[str], str]


@dataclass(frozen=True)
class ImageSource:
    """A prebuilt image, named as the package names it."""

    reference: str


@dataclass(frozen=True)
class BuildSource:
    """A directory holding ``Dockerfile`` and its build context."""

    context: Path


type ContainerSource = ImageSource | BuildSource


@dataclass(frozen=True)
class LoweredSource:
    """Exactly one of a digest-pinned image or a build context with pinned bases."""

    docker_image: str | None
    docker_build: DockerBuildContext | None
    workdir: str | None
    """The ``WORKDIR`` the Dockerfile declares; ``None`` leaves the image's own."""


def directory_resources(directory: Path, prefix: str = "") -> tuple[TaskResource, ...]:
    """Every regular file below ``directory``, executable or not; symlinks are rejected.

    Package stores keep only the executable bit, so modes are normalized to 0755 or 0644 and modification
    times are omitted; the checkout's umask and download time would otherwise change task identity.
    """
    if not directory.is_dir():
        return ()
    resources = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise UnsupportedHarborFeature(f"package symlink {path.relative_to(directory.parent)}")
        if not path.is_file():
            continue
        resources.append(
            TaskResource(
                path=f"{prefix}{path.relative_to(directory).as_posix()}",
                source=InlineFile(content_base64=base64.b64encode(path.read_bytes()).decode("ascii")),
                mode="0755" if path.stat().st_mode & stat.S_IXUSR else "0644",
            )
        )
    return tuple(resources)


def source_references(source: ContainerSource) -> tuple[str, ...]:
    """The external images a container needs before it can start."""
    if isinstance(source, ImageSource):
        return (source.reference,)
    return dockerfile_images((source.context / DOCKERFILE).read_text())


def pinner(image_digests: Mapping[str, str]) -> Pin:
    """Pin a reference to its resolved digest as ``name:tag@sha256:...``; pinned references are kept."""

    def pin(reference: str) -> str:
        if PINNED_IMAGE.fullmatch(reference):
            return reference
        if reference not in image_digests:
            raise ValueError(f"No digest was resolved for image {reference!r}")
        digest = image_digests[reference]
        if DIGEST.fullmatch(digest) is None:
            raise ValueError(f"Image digest for {reference!r} must be sha256:<64 hex digits>, not {digest!r}")
        return f"{reference}@{digest}"

    return pin


def _inline(text: str) -> InlineFile:
    return InlineFile(content_base64=base64.b64encode(text.encode()).decode())


def _with_dockerfile(files: tuple[TaskResource, ...], dockerfile: str) -> tuple[TaskResource, ...]:
    return tuple(
        resource.model_copy(update={"source": _inline(dockerfile)}) if resource.path == DOCKERFILE else resource
        for resource in files
    )


def lower_source(source: ContainerSource, pin: Pin) -> LoweredSource:
    if isinstance(source, ImageSource):
        return LoweredSource(pin(source.reference), None, None)
    dockerfile = (source.context / DOCKERFILE).read_bytes().decode()
    files = _with_dockerfile(directory_resources(source.context), pin_dockerfile(dockerfile, pin))
    return LoweredSource(None, DockerBuildContext(files=files), dockerfile_workdir(dockerfile))


def extend_source(source: LoweredSource, instructions: str) -> LoweredSource:
    """``source`` built with Dockerfile ``instructions`` after its own; a prebuilt image becomes their base."""
    if source.docker_build is None:
        dockerfile = f"FROM {source.docker_image}\n{instructions}"
        files = (TaskResource(path=DOCKERFILE, source=_inline(dockerfile), mode="0644"),)
    else:
        resource = next(file for file in source.docker_build.files if file.path == DOCKERFILE)
        assert isinstance(resource.source, InlineFile)
        base = base64.b64decode(resource.source.content_base64).decode()
        files = _with_dockerfile(source.docker_build.files, f"{base.rstrip()}\n{instructions}")
    return LoweredSource(None, DockerBuildContext(files=files), source.workdir)
