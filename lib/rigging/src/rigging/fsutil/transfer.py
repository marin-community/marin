# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy, move, and synchronization plans across routed filesystems."""

import base64
import binascii
import hashlib
import shutil
from dataclasses import dataclass
from glob import has_magic
from typing import Any, cast

from fsspec import AbstractFileSystem

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

COPY_CHUNK_BYTES = 8 * 1024 * 1024
DIRECTORY_TYPE = "directory"


class TransferError(ValueError):
    """An invalid or unsafe transfer request."""


@dataclass(frozen=True)
class TransferSource:
    url: str
    filesystem: AbstractFileSystem
    path: str
    is_directory: bool


@dataclass(frozen=True)
class CopyAction:
    source_url: str
    source_filesystem: AbstractFileSystem
    source_path: str
    destination_url: str
    destination_filesystem: AbstractFileSystem
    destination_path: str


@dataclass(frozen=True)
class CopyPlan:
    sources: tuple[TransferSource, ...]
    copies: tuple[CopyAction, ...]
    skipped: tuple[CopyAction, ...]


@dataclass(frozen=True)
class DeleteAction:
    url: str
    filesystem: AbstractFileSystem
    path: str


@dataclass(frozen=True)
class SyncPlan:
    copies: tuple[CopyAction, ...]
    deletes: tuple[DeleteAction, ...]


def copy_plan(
    source_urls: tuple[str, ...],
    destination_url: str,
    *,
    recursive: bool,
    no_clobber: bool,
) -> CopyPlan:
    """Resolve copy destinations and validate every source before writing."""
    assert source_urls
    destination_filesystem, destination_path = filesystem_for(destination_url)
    sources = _sources(source_urls)
    for source in sources:
        if source.is_directory and not recursive:
            raise TransferError(f"{source.url} is a directory; pass -r to copy it recursively")
        if source.is_directory and (
            StoragePath(source.url) == StoragePath(destination_url) or _strictly_contains(source.url, destination_url)
        ):
            raise TransferError(f"destination {destination_url} is inside source {source.url}")

    destination_is_directory = (
        len(sources) > 1
        or any(has_magic(url) for url in source_urls)
        or destination_url.endswith("/")
        or destination_filesystem.isdir(destination_path)
    )
    if destination_is_directory and destination_filesystem.exists(destination_path):
        if not destination_filesystem.isdir(destination_path):
            raise TransferError(f"{destination_url} is not a directory")

    copies = []
    skipped = []
    for source in sources:
        actions = _source_copy_actions(
            source,
            destination_url,
            destination_filesystem,
            destination_path,
            destination_is_directory=destination_is_directory,
        )
        for action in actions:
            if no_clobber and action.destination_filesystem.exists(action.destination_path):
                skipped.append(action)
            else:
                copies.append(action)
    for action in copies:
        if _same_location(action):
            raise TransferError(f"source and destination are the same: {action.source_url}")
    return CopyPlan(sources, tuple(copies), tuple(skipped))


def execute_copies(actions: tuple[CopyAction, ...]) -> None:
    """Execute a validated sequence of copies."""
    for action in actions:
        if _same_location(action):
            raise TransferError(f"source and destination are the same: {action.source_url}")
        _copy_file(
            action.source_filesystem,
            action.source_path,
            action.destination_filesystem,
            action.destination_path,
        )


def remove_sources(sources: tuple[TransferSource, ...]) -> None:
    """Remove sources after every copy in a move has succeeded."""
    for source in sources:
        source.filesystem.rm(source.path, recursive=source.is_directory)


def sync_plan(source_url: str, destination_url: str, *, delete: bool, checksum: bool) -> SyncPlan:
    """Plan the operations needed to make a destination contain the source files.

    Size, modification time, and provider MD5 metadata are compared without reading
    object bodies. ``checksum`` forces a full MD5 read of equal-sized files.
    """
    source_filesystem, source_path = filesystem_for(source_url)
    destination_filesystem, destination_path = filesystem_for(destination_url)
    if not source_filesystem.exists(source_path):
        raise TransferError(f"{source_url} does not exist")
    if not source_filesystem.isdir(source_path):
        raise TransferError(f"{source_url} is not a directory")
    if destination_filesystem.exists(destination_path) and not destination_filesystem.isdir(destination_path):
        raise TransferError(f"{destination_url} is not a directory")
    if _strictly_contains(source_url, destination_url) or _strictly_contains(destination_url, source_url):
        raise TransferError("source and destination directories overlap")

    source_files = _manifest(source_filesystem, source_path)
    destination_files = (
        _manifest(destination_filesystem, destination_path) if destination_filesystem.exists(destination_path) else {}
    )
    copies = []
    for relative_path, source_info in source_files.items():
        destination_object_path = _join(destination_path, relative_path)
        destination_info = destination_files.get(relative_path)
        if destination_info is not None and _matching_files(
            source_filesystem,
            source_info,
            destination_filesystem,
            destination_info,
            checksum=checksum,
        ):
            continue
        copies.append(
            CopyAction(
                source_url=_qualified_url(source_url, source_info["name"]),
                source_filesystem=source_filesystem,
                source_path=source_info["name"],
                destination_url=_qualified_url(destination_url, destination_object_path),
                destination_filesystem=destination_filesystem,
                destination_path=destination_object_path,
            )
        )

    deletes = []
    if delete:
        for relative_path in destination_files.keys() - source_files.keys():
            info = destination_files[relative_path]
            deletes.append(
                DeleteAction(
                    url=_qualified_url(destination_url, info["name"]),
                    filesystem=destination_filesystem,
                    path=info["name"],
                )
            )
    copies.sort(key=lambda action: action.source_url)
    deletes.sort(key=lambda action: action.url)
    return SyncPlan(tuple(copies), tuple(deletes))


def execute_sync(plan: SyncPlan) -> None:
    """Execute copies before deletions so a failed copy cannot discard destination data."""
    execute_copies(plan.copies)
    for action in plan.deletes:
        action.filesystem.rm(action.path)


def _sources(urls: tuple[str, ...]) -> tuple[TransferSource, ...]:
    sources = []
    for url in urls:
        filesystem, path = filesystem_for(url)
        if not has_magic(path):
            if not filesystem.exists(path):
                raise TransferError(f"{url} does not exist")
            sources.append(TransferSource(url, filesystem, path, filesystem.isdir(path)))
            continue
        matches = filesystem.glob(path)
        if not matches:
            raise TransferError(f"{url} matched no files")
        sources.extend(
            TransferSource(
                _qualified_url(url, match),
                filesystem,
                match,
                filesystem.isdir(match),
            )
            for match in matches
        )
    return tuple(sources)


def _source_copy_actions(
    source: TransferSource,
    destination_url: str,
    destination_filesystem: AbstractFileSystem,
    destination_path: str,
    *,
    destination_is_directory: bool,
) -> list[CopyAction]:
    if not source.is_directory:
        target_path = _join(destination_path, _basename(source.path)) if destination_is_directory else destination_path
        return [
            CopyAction(
                source.url,
                source.filesystem,
                source.path,
                _qualified_url(destination_url, target_path),
                destination_filesystem,
                target_path,
            )
        ]

    root = _join(destination_path, _basename(source.path)) if destination_is_directory else destination_path
    actions = []
    for info in _files(source.filesystem, source.path).values():
        relative_path = _relative_path(info["name"], source.path)
        target_path = _join(root, relative_path)
        actions.append(
            CopyAction(
                _qualified_url(source.url, info["name"]),
                source.filesystem,
                info["name"],
                _qualified_url(destination_url, target_path),
                destination_filesystem,
                target_path,
            )
        )
    actions.sort(key=lambda action: action.source_url)
    return actions


def _manifest(filesystem: AbstractFileSystem, root: str) -> dict[str, dict[str, Any]]:
    return {_relative_path(info["name"], root): info for info in _files(filesystem, root).values()}


def _files(filesystem: AbstractFileSystem, path: str) -> dict[str, dict[str, Any]]:
    found = cast(dict[str, dict[str, Any]] | list[str], filesystem.find(path, detail=True))
    if not isinstance(found, dict):
        details = {name: cast(dict[str, Any], filesystem.info(name)) for name in found}
        return {name: info for name, info in details.items() if info.get("type") != DIRECTORY_TYPE}
    return {name: info for name, info in found.items() if info.get("type") != DIRECTORY_TYPE}


def _matching_files(
    source_filesystem: AbstractFileSystem,
    source_info: dict[str, Any],
    destination_filesystem: AbstractFileSystem,
    destination_info: dict[str, Any],
    *,
    checksum: bool,
) -> bool:
    if (source_info.get("size") or 0) != (destination_info.get("size") or 0):
        return False
    if checksum:
        return _md5(source_filesystem, source_info["name"]) == _md5(
            destination_filesystem,
            destination_info["name"],
        )

    source_mtime = _mtime(source_info)
    destination_mtime = _mtime(destination_info)
    if source_mtime is not None and destination_mtime is not None and source_mtime == destination_mtime:
        return True

    source_md5 = _metadata_md5(source_info)
    destination_md5 = _metadata_md5(destination_info)
    if source_md5 is not None and destination_md5 is not None:
        return source_md5 == destination_md5
    if _is_local(source_filesystem) and _is_local(destination_filesystem):
        return _md5(source_filesystem, source_info["name"]) == _md5(
            destination_filesystem,
            destination_info["name"],
        )

    # Without two timestamps or two provider digests, size is the only metadata
    # comparison available. --checksum is the explicit full-content fallback.
    return source_mtime is None or destination_mtime is None


def _is_local(filesystem: AbstractFileSystem) -> bool:
    protocols = (filesystem.protocol,) if isinstance(filesystem.protocol, str) else filesystem.protocol
    return "file" in protocols or "local" in protocols


def _mtime(info: dict[str, Any]) -> Any | None:
    for key in ("mtime", "LastModified", "updated"):
        if key in info:
            return info[key]
    return None


def _metadata_md5(info: dict[str, Any]) -> bytes | None:
    value = next((info[key] for key in ("md5Hash", "md5", "ETag", "etag") if info.get(key)), None)
    if isinstance(value, bytes):
        return value if len(value) == 16 else None
    if not isinstance(value, str):
        return None

    encoded = value.strip('"')
    if "-" in encoded:
        return None
    if len(encoded) == 32:
        try:
            return bytes.fromhex(encoded)
        except ValueError:
            return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return None
    return decoded if len(decoded) == 16 else None


def _md5(filesystem: AbstractFileSystem, path: str) -> bytes:
    digest = hashlib.md5(usedforsecurity=False)
    with filesystem.open(path, "rb") as file:
        while chunk := file.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.digest()


def _copy_file(
    source_filesystem: AbstractFileSystem,
    source_path: str,
    destination_filesystem: AbstractFileSystem,
    destination_path: str,
) -> None:
    parent, separator, _ = destination_path.rpartition("/")
    if separator:
        destination_filesystem.makedirs(parent, exist_ok=True)
    with (
        source_filesystem.open(source_path, "rb") as source,
        destination_filesystem.open(destination_path, "wb") as destination,
    ):
        shutil.copyfileobj(source, destination, COPY_CHUNK_BYTES)


def _same_location(action: CopyAction) -> bool:
    same_filesystem_path = action.source_filesystem is action.destination_filesystem and action.source_path.rstrip(
        "/"
    ) == action.destination_path.rstrip("/")
    return same_filesystem_path or StoragePath(action.source_url) == StoragePath(action.destination_url)


def _strictly_contains(parent_url: str, child_url: str) -> bool:
    parent = StoragePath(parent_url)
    child = StoragePath(child_url)
    same_root = (parent.scheme, parent.netloc, parent.rooted) == (child.scheme, child.netloc, child.rooted)
    return (
        same_root
        and len(parent.segments) < len(child.segments)
        and child.segments[: len(parent.segments)] == parent.segments
    )


def _qualified_url(reference_url: str, path: str) -> str:
    scheme = StoragePath(reference_url).scheme
    return f"{scheme}://{path}" if scheme else path


def _relative_path(path: str, root: str) -> str:
    return path[len(root.rstrip("/")) :].lstrip("/")


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _join(parent: str, child: str) -> str:
    return f"{parent.rstrip('/')}/{child.lstrip('/')}"
