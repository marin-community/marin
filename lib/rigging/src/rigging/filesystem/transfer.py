# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable tree copies between fsspec filesystems."""

import logging
import posixpath
from dataclasses import dataclass
from enum import StrEnum

from fsspec.spec import AbstractFileSystem

import rigging.filesystem.factory as filesystem_factory
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

_COPY_CHUNK_BYTES = 8 * 1024 * 1024


class TreeTransferMode(StrEnum):
    """How :func:`copy_tree` treats files already present at the destination."""

    RESUME = "resume"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class TreeTransferResult:
    """Summary of one tree transfer."""

    copied_files: int
    skipped_files: int
    copied_bytes: int


def _path_join(root: str, relative: str) -> str:
    if not relative or relative == ".":
        return root
    return posixpath.join(root, relative)


def _relative_path(path: str, root: str) -> str:
    normalized_path = posixpath.normpath(f"/{path.lstrip('/')}")
    normalized_root = posixpath.normpath(f"/{root.lstrip('/')}")
    if normalized_path == normalized_root:
        return ""
    if posixpath.commonpath((normalized_path, normalized_root)) != normalized_root:
        raise ValueError(f"filesystem walk returned {path!r} outside source root {root!r}")
    return posixpath.relpath(normalized_path, normalized_root)


def _reject_recursive_copy(source: StoragePath, destination: StoragePath) -> None:
    try:
        relative = destination.relative_to(source)
    except ValueError:
        return
    location = "the source" if not relative else "inside source"
    raise ValueError(f"copy_tree destination {destination} is {location} {source}")


def _copy_file(
    source_fs: AbstractFileSystem,
    source_path: str,
    destination_fs: AbstractFileSystem,
    destination_path: str,
) -> None:
    destination_parent = posixpath.dirname(destination_path)
    if destination_parent:
        destination_fs.makedirs(destination_parent, exist_ok=True)
    with filesystem_factory.atomic_rename(destination_path, destination_fs) as temporary_path:
        with (
            source_fs.open(source_path, "rb") as source_file,
            destination_fs.open(temporary_path, "wb") as destination_file,
        ):
            while chunk := source_file.read(_COPY_CHUNK_BYTES):
                destination_file.write(chunk)


def copy_tree(
    source: StoragePath,
    destination: StoragePath,
    *,
    mode: TreeTransferMode,
) -> TreeTransferResult:
    """Copy the contents of one directory tree between fsspec filesystems.

    ``RESUME`` skips destination files with the same byte size and replaces missing
    or differently sized files. ``OVERWRITE`` replaces every source file. Neither
    mode deletes destination-only files. Replacements are atomic.
    """
    _reject_recursive_copy(source, destination)
    source_fs, source_root = filesystem_factory.url_to_fs(str(source))
    destination_fs, destination_root = filesystem_factory.url_to_fs(str(destination))
    if not source_fs.isdir(source_root):
        raise ValueError(f"copy_tree source must be a directory, got: {source}")

    destination_fs.makedirs(destination_root, exist_ok=True)
    copied_files = 0
    skipped_files = 0
    copied_bytes = 0

    for directory, _subdirectories, files in source_fs.walk(source_root):
        relative_directory = _relative_path(directory, source_root)
        destination_directory = _path_join(destination_root, relative_directory)
        destination_fs.makedirs(destination_directory, exist_ok=True)
        for filename in files:
            source_file = posixpath.join(directory, filename)
            destination_file = posixpath.join(destination_directory, filename)
            source_size = source_fs.size(source_file)
            if (
                mode is TreeTransferMode.RESUME
                and destination_fs.isfile(destination_file)
                and destination_fs.size(destination_file) == source_size
            ):
                skipped_files += 1
                continue
            _copy_file(source_fs, source_file, destination_fs, destination_file)
            copied_files += 1
            copied_bytes += source_size

    result = TreeTransferResult(
        copied_files=copied_files,
        skipped_files=skipped_files,
        copied_bytes=copied_bytes,
    )
    logger.info(
        "copied tree %s -> %s: %d files (%d bytes), %d skipped [%s]",
        source,
        destination,
        result.copied_files,
        result.copied_bytes,
        result.skipped_files,
        mode.value,
    )
    return result
