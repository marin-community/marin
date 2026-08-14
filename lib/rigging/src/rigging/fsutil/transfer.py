# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy, move, and synchronization plans across routed filesystems."""

import base64
import binascii
import shutil
from dataclasses import dataclass
from glob import has_magic
from typing import Any, cast

from fsspec import AbstractFileSystem

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.fsutil.hashing import HashAlgorithm, file_hashes_for_path

COPY_CHUNK_BYTES = 8 * 1024 * 1024
DIRECTORY_TYPE = "directory"
MD5_DIGEST_BYTES = 16
MD5_ONLY = frozenset({HashAlgorithm.MD5})


class TransferError(ValueError):
    """An invalid or unsafe transfer request."""


@dataclass(frozen=True)
class TransferLocation:
    url: str
    filesystem: AbstractFileSystem
    path: str

    @classmethod
    def from_path(cls, reference_url: str, filesystem: AbstractFileSystem, path: str) -> "TransferLocation":
        scheme = StoragePath(reference_url).scheme
        url = f"{scheme}://{path}" if scheme else path
        return cls(url, filesystem, path)

    @property
    def name(self) -> str:
        return StoragePath(self.url).name

    def child(self, relative_path: str) -> "TransferLocation":
        return TransferLocation(
            prefix_join(self.url, relative_path),
            self.filesystem,
            prefix_join(self.path, relative_path),
        )

    def relative_to(self, root: "TransferLocation") -> str:
        return StoragePath(self.url).relative_to(StoragePath(root.url))


@dataclass(frozen=True)
class TransferSource:
    location: TransferLocation
    is_directory: bool


@dataclass(frozen=True)
class CopyAction:
    source: TransferLocation
    destination: TransferLocation


@dataclass(frozen=True)
class CopyPlan:
    sources: tuple[TransferSource, ...]
    copies: tuple[CopyAction, ...]
    skipped: tuple[CopyAction, ...]


@dataclass(frozen=True)
class DeleteAction:
    location: TransferLocation


@dataclass(frozen=True)
class FileMetadata:
    location: TransferLocation
    size: int
    mtime: Any | None
    md5: bytes | None


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
    destination = TransferLocation(destination_url, destination_filesystem, destination_path)
    sources = _sources(source_urls)
    for source in sources:
        if source.is_directory and not recursive:
            raise TransferError(f"{source.location.url} is a directory; pass -r to copy it recursively")
        if source.is_directory and (
            StoragePath(source.location.url) == StoragePath(destination_url)
            or _strictly_contains(source.location.url, destination_url)
        ):
            raise TransferError(f"destination {destination_url} is inside source {source.location.url}")

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
            destination,
            destination_is_directory=destination_is_directory,
        )
        for action in actions:
            if no_clobber and action.destination.filesystem.exists(action.destination.path):
                skipped.append(action)
            else:
                copies.append(action)
    for action in copies:
        if _same_location(action):
            raise TransferError(f"source and destination are the same: {action.source.url}")
    return CopyPlan(sources, tuple(copies), tuple(skipped))


def execute_copies(actions: tuple[CopyAction, ...]) -> None:
    """Execute a validated sequence of copies."""
    for action in actions:
        if _same_location(action):
            raise TransferError(f"source and destination are the same: {action.source.url}")
        _copy_file(action.source, action.destination)


def remove_sources(sources: tuple[TransferSource, ...]) -> None:
    """Remove sources after every copy in a move has succeeded."""
    for source in sources:
        source.location.filesystem.rm(source.location.path, recursive=source.is_directory)


def sync_plan(source_url: str, destination_url: str, *, delete: bool, checksum: bool) -> SyncPlan:
    """Plan the operations needed to make a destination contain the source files.

    Size, modification time, and provider MD5 metadata are compared first. Equal-sized
    local files with differing times are hashed; ``checksum`` forces hashing regardless.
    """
    source_filesystem, source_path = filesystem_for(source_url)
    destination_filesystem, destination_path = filesystem_for(destination_url)
    source = TransferLocation(source_url, source_filesystem, source_path)
    destination = TransferLocation(destination_url, destination_filesystem, destination_path)
    if not source_filesystem.exists(source_path):
        raise TransferError(f"{source_url} does not exist")
    if not source_filesystem.isdir(source_path):
        raise TransferError(f"{source_url} is not a directory")
    if destination_filesystem.exists(destination_path) and not destination_filesystem.isdir(destination_path):
        raise TransferError(f"{destination_url} is not a directory")
    if _strictly_contains(source_url, destination_url) or _strictly_contains(destination_url, source_url):
        raise TransferError("source and destination directories overlap")

    source_files = _manifest(source)
    destination_files = _manifest(destination) if destination_filesystem.exists(destination_path) else {}
    copies = []
    for relative_path, source_info in source_files.items():
        destination_info = destination_files.get(relative_path)
        if destination_info is not None and _matching_files(
            source_info,
            destination_info,
            checksum=checksum,
        ):
            continue
        copies.append(CopyAction(source_info.location, destination.child(relative_path)))

    deletes = []
    if delete:
        for relative_path in destination_files.keys() - source_files.keys():
            info = destination_files[relative_path]
            deletes.append(DeleteAction(info.location))
    copies.sort(key=lambda action: action.source.url)
    deletes.sort(key=lambda action: action.location.url)
    return SyncPlan(tuple(copies), tuple(deletes))


def execute_sync(plan: SyncPlan) -> None:
    """Execute copies before deletions so a failed copy cannot discard destination data."""
    execute_copies(plan.copies)
    for action in plan.deletes:
        action.location.filesystem.rm(action.location.path)


def _sources(urls: tuple[str, ...]) -> tuple[TransferSource, ...]:
    sources = []
    for url in urls:
        filesystem, path = filesystem_for(url)
        if not has_magic(path):
            if not filesystem.exists(path):
                raise TransferError(f"{url} does not exist")
            location = TransferLocation(url, filesystem, path)
            sources.append(TransferSource(location, filesystem.isdir(path)))
            continue
        matches = filesystem.glob(path)
        if not matches:
            raise TransferError(f"{url} matched no files")
        sources.extend(
            TransferSource(
                TransferLocation.from_path(url, filesystem, match),
                filesystem.isdir(match),
            )
            for match in matches
        )
    return tuple(sources)


def _source_copy_actions(
    source: TransferSource,
    destination: TransferLocation,
    *,
    destination_is_directory: bool,
) -> list[CopyAction]:
    if not source.is_directory:
        target = destination.child(source.location.name) if destination_is_directory else destination
        return [CopyAction(source.location, target)]

    root = destination.child(source.location.name) if destination_is_directory else destination
    actions = []
    for info in _files(source.location):
        relative_path = info.location.relative_to(source.location)
        actions.append(CopyAction(info.location, root.child(relative_path)))
    actions.sort(key=lambda action: action.source.url)
    return actions


def _manifest(root: TransferLocation) -> dict[str, FileMetadata]:
    return {info.location.relative_to(root): info for info in _files(root)}


def _files(root: TransferLocation) -> list[FileMetadata]:
    found = cast(dict[str, dict[str, Any]] | list[str], root.filesystem.find(root.path, detail=True))
    if not isinstance(found, dict):
        found = {name: cast(dict[str, Any], root.filesystem.info(name)) for name in found}
    return [
        FileMetadata(
            location=TransferLocation.from_path(root.url, root.filesystem, name),
            size=int(info.get("size") or 0),
            mtime=_mtime(info),
            md5=_metadata_md5(info),
        )
        for name, info in found.items()
        if info.get("type") != DIRECTORY_TYPE
    ]


def _matching_files(
    source: FileMetadata,
    destination: FileMetadata,
    *,
    checksum: bool,
) -> bool:
    if source.size != destination.size:
        return False
    if checksum:
        return _md5(source.location) == _md5(destination.location)

    if source.mtime is not None and destination.mtime is not None and source.mtime == destination.mtime:
        return True

    if source.md5 is not None and destination.md5 is not None:
        return source.md5 == destination.md5
    if _is_local(source.location.filesystem) and _is_local(destination.location.filesystem):
        return _md5(source.location) == _md5(destination.location)

    # Without two timestamps or two provider digests, size is the only metadata
    # comparison available. --checksum is the explicit full-content fallback.
    return source.mtime is None or destination.mtime is None


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
        return value if len(value) == MD5_DIGEST_BYTES else None
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
    return decoded if len(decoded) == MD5_DIGEST_BYTES else None


def _md5(location: TransferLocation) -> bytes:
    digest = file_hashes_for_path(location.filesystem, location.path, MD5_ONLY).md5
    assert digest is not None
    return digest


def _copy_file(source: TransferLocation, destination: TransferLocation) -> None:
    parent, separator, _ = destination.path.rpartition("/")
    if separator:
        destination.filesystem.makedirs(parent, exist_ok=True)
    with (
        source.filesystem.open(source.path, "rb") as source_file,
        destination.filesystem.open(destination.path, "wb") as destination_file,
    ):
        shutil.copyfileobj(source_file, destination_file, COPY_CHUNK_BYTES)


def _same_location(action: CopyAction) -> bool:
    same_filesystem_path = (
        action.source.filesystem is action.destination.filesystem and action.source.path == action.destination.path
    )
    return same_filesystem_path or StoragePath(action.source.url) == StoragePath(action.destination.url)


def _strictly_contains(parent_url: str, child_url: str) -> bool:
    parent = StoragePath(parent_url)
    child = StoragePath(child_url)
    same_root = (parent.scheme, parent.netloc, parent.rooted) == (child.scheme, child.netloc, child.rooted)
    return (
        same_root
        and len(parent.segments) < len(child.segments)
        and child.segments[: len(parent.segments)] == parent.segments
    )
