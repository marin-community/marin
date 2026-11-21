# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize original Docker build inputs for an explicitly selected executor."""

import os
from pathlib import Path

from shellbox.image import DockerfileSource
from shellbox.machine import UnsupportedMachineSpec

from taskcompendium.models import DockerBuildContext
from taskcompendium.runtime.resources import resource_bytes


def materialize_build_directories(build: DockerBuildContext, directory: Path) -> None:
    """Restore only the empty directories explicitly declared by the source archive."""
    for entry in build.empty_directories:
        path = directory / entry.path
        path.mkdir(parents=True, exist_ok=False)
        path.chmod(int(entry.mode, 8))
        os.utime(path, ns=(entry.mtime_ns, entry.mtime_ns))


def materialize_docker_build(build: DockerBuildContext, directory: Path) -> DockerfileSource:
    """Write the declared bytes and metadata into a fresh caller-owned context."""
    directory.mkdir(parents=True, exist_ok=False)
    for resource in build.files:
        path = directory / resource.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource_bytes(resource))
        path.chmod(int(resource.mode, 8) if resource.mode is not None else 0o644)
        if resource.mtime_ns is not None:
            os.utime(path, ns=(resource.mtime_ns, resource.mtime_ns))
    materialize_build_directories(build, directory)
    return DockerfileSource(directory, directory / "Dockerfile")


def require_original_docker_build(build: DockerBuildContext, source: DockerfileSource) -> None:
    """Reject a selected context that differs from the source's declared inputs."""
    expected = {resource.path for resource in build.files}
    actual = {path.relative_to(source.context).as_posix() for path in source.context.rglob("*") if path.is_file()}
    if actual != expected or source.dockerfile != source.context / "Dockerfile":
        raise UnsupportedMachineSpec("Selected Docker build context differs from the original declaration")
    expected_directories = {entry.path for entry in build.empty_directories}
    for name in (*expected, *expected_directories):
        expected_directories.update(parent.as_posix() for parent in Path(name).parents if parent != Path("."))
    actual_directories = {
        path.relative_to(source.context).as_posix() for path in source.context.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise UnsupportedMachineSpec("Selected Docker build directories differ from the original declaration")
    for entry in build.empty_directories:
        path = source.context / entry.path
        if path.stat().st_mode & 0o7777 != int(entry.mode, 8) or path.stat().st_mtime_ns != entry.mtime_ns:
            raise UnsupportedMachineSpec(f"Selected Docker build directory metadata changed: {entry.path}")
    for resource in build.files:
        path = source.context / resource.path
        if (
            path.read_bytes() != resource_bytes(resource)
            or path.stat().st_mode & 0o777 != (int(resource.mode, 8) if resource.mode is not None else 0o644)
            or (resource.mtime_ns is not None and path.stat().st_mtime_ns != resource.mtime_ns)
        ):
            raise UnsupportedMachineSpec(f"Selected Docker build input changed: {resource.path}")
