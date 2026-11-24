# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""File resources used by the optional episode runtime."""

import base64
import os
import tempfile
import zipfile
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError

from shellbox.image import DockerfileSource

from taskcompendium.importers.scienceagentbench_assets.release import (
    Archive,
    copy_local_member,
    copy_member,
    download_release,
)
from taskcompendium.models import (
    DOCKERFILE,
    DockerBuildContext,
    InlineFile,
    PrivateVerifierResource,
    PublisherFile,
    TaskResource,
)


def inline_resource(path: str, data: bytes) -> TaskResource:
    """Build an inline resource at a relative workspace path."""
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(data).decode("ascii")))


def resource_bytes(resource: TaskResource | PrivateVerifierResource) -> bytes:
    """Decode an inline resource without resolving external content."""
    if not isinstance(resource.source, InlineFile):
        raise ValueError("Publisher resources must be staged to a file")
    return base64.b64decode(resource.source.content_base64, validate=True)


LARGE_PUBLISHER_TRANSFER = 64 * 1024 * 1024


class ResourceReader(ExitStack):
    """Reuse the publisher ZIP directory and range buffer when staging a group of files."""

    def __init__(self, publisher_files: Iterable[PublisherFile] = ()):
        super().__init__()
        self.publisher_bytes = sum(file.size for file in publisher_files)
        self.archive: zipfile.ZipFile | None = None
        self.release_path: Path | None = None

    def copy_publisher_file(self, source: PublisherFile, output: BinaryIO) -> None:
        if self.release_path is None and max(self.publisher_bytes, source.size) < LARGE_PUBLISHER_TRANSFER:
            start = output.tell()
            try:
                if self.archive is None:
                    stream = self.enter_context(Archive())
                    self.archive = self.enter_context(zipfile.ZipFile(stream))
                copy_member(self.archive, source.member, source.sha256, output, source.size)
                return
            except HTTPError:
                raise
            except (URLError, TimeoutError, zipfile.BadZipFile):
                # A range transfer can fail after part of this member has already been written.
                # ZipFile also wraps an OSError while reading the central directory in BadZipFile.
                output.seek(start)
                output.truncate()
        if self.release_path is None:
            release_path = Path(self.enter_context(tempfile.TemporaryDirectory())) / "release.zip"
            download_release(release_path)
            self.release_path = release_path
        copy_local_member(self.release_path, source.member, source.sha256, output, source.size)

    def write_resource(self, resource: TaskResource, path: Path) -> None:
        """Materialize a verified file in a private staging directory."""
        if isinstance(resource.source, PublisherFile):
            with path.open("wb") as output:
                self.copy_publisher_file(resource.source, output)
        else:
            path.write_bytes(resource_bytes(resource))


def write_build_context(context: DockerBuildContext, directory: Path) -> Path:
    """Write a build context's files under ``directory`` with their modes and timestamps; return its Dockerfile."""
    with ResourceReader() as reader:
        for resource in context.files:
            target = directory / resource.path
            target.parent.mkdir(parents=True, exist_ok=True)
            reader.write_resource(resource, target)
            target.chmod(0o644 if resource.mode is None else int(resource.mode, 8))
            if resource.mtime_ns is not None:
                os.utime(target, ns=(resource.mtime_ns, resource.mtime_ns))
    return directory / DOCKERFILE


def materialize_build_context(recipe: DockerBuildContext, directory: Path) -> DockerfileSource:
    """Create a private build context for a grading stage."""
    directory.mkdir(parents=True, exist_ok=False)
    return DockerfileSource(directory, write_build_context(recipe, directory))
