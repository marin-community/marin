# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable, content-verified prefix copies across routed filesystems."""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from fsspec import AbstractFileSystem

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import S3UploadPolicy, filesystem_for
from rigging.fsutil.transfer import (
    COPY_CHUNK_BYTES,
    TransferLocation,
    _backend,
    _join_path,
    _join_url,
    _relative_path,
    _same_location,
    _strictly_contains,
)

DEFAULT_VERIFIED_COPY_WORKERS = 4
S3_UPLOAD_PART_BYTES = 50 * 1024 * 1024
COMPLETION_MANIFEST = ".verified-copy-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
SHA256_IDENTITY_PREFIX = "sha256="
ETAG_IDENTITY_PREFIX = "etag="

logger = logging.getLogger(__name__)


class VerifiedCopyError(ValueError):
    """A verified prefix copy cannot safely continue."""


@dataclass(frozen=True, order=True)
class VerifiedFile:
    path: str
    size: int
    sha256: str
    source_identity: str | None


@dataclass(frozen=True)
class VerifiedCopyResult:
    manifest_url: str
    copied_files: int
    resumed_files: int
    total_files: int
    total_bytes: int


@dataclass(frozen=True)
class _SourceFile:
    path: str
    source_path: str
    size: int
    identity: str | None


@dataclass(frozen=True)
class _DestinationFile:
    size: int
    identity: str | None


@dataclass(frozen=True)
class _CopiedFile:
    sha256: str
    size: int
    expected_etag: str | None


class _CopyDisposition(StrEnum):
    COPIED = "copied"
    RESUMED = "resumed"


@dataclass(frozen=True)
class _CopyOutcome:
    file: VerifiedFile
    disposition: _CopyDisposition


@dataclass(frozen=True)
class _ResumeMarker:
    source_path: str
    source_identity: str | None
    path: str
    size: int
    sha256: str
    destination_identity: str


@dataclass(frozen=True)
class _CompletionManifest:
    schema_version: int
    source: str
    destination: str
    total_files: int
    total_bytes: int
    files: list[VerifiedFile]


class _MultipartEtag:
    """Hash a byte stream using the content-derived S3/R2 multipart ETag rules."""

    def __init__(self, part_size: int):
        self.part_size = part_size
        self.part_hashes: list[bytes] = []
        self.current_hash = hashlib.md5(usedforsecurity=False)
        self.current_size = 0

    def update(self, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            length = min(self.part_size - self.current_size, len(data) - offset)
            self.current_hash.update(data[offset : offset + length])
            self.current_size += length
            offset += length
            if self.current_size == self.part_size:
                self.part_hashes.append(self.current_hash.digest())
                self.current_hash = hashlib.md5(usedforsecurity=False)
                self.current_size = 0

    def etag(self, total_size: int) -> str:
        if total_size < self.part_size:
            return self.current_hash.hexdigest()
        part_hashes = [*self.part_hashes]
        if self.current_size:
            part_hashes.append(self.current_hash.digest())
        digest = hashlib.md5(b"".join(part_hashes), usedforsecurity=False).hexdigest()
        return f"{digest}-{len(part_hashes)}"


def verified_copy_prefix(
    source_url: str,
    destination_url: str,
    *,
    status_url: str | None = None,
    workers: int = DEFAULT_VERIFIED_COPY_WORKERS,
) -> VerifiedCopyResult:
    """Copy and verify a prefix before publishing its completion manifest.

    Verified per-object records allow retries to reuse destination objects when
    their source and destination identities still match.
    """
    if workers < 1:
        raise VerifiedCopyError("workers must be at least 1")

    source = _resolved_location(source_url, S3UploadPolicy.STANDARD)
    destination = _resolved_location(destination_url, S3UploadPolicy.FIXED_PARTS)
    status = _resolved_location(
        status_url or f"{destination.url}.verified-copy-status",
        S3UploadPolicy.FIXED_PARTS,
    )
    _validate_disjoint_locations(source, destination, status)

    source_url, source_fs, source_root = source.url, source.filesystem, source.path
    destination_url, destination_fs, destination_root = destination.url, destination.filesystem, destination.path
    status_fs, status_root = status.filesystem, status.path
    sources = _source_files(source_fs, source_root)
    if not sources:
        raise VerifiedCopyError(f"source prefix contains no files: {source_url}")
    if any(source.path == COMPLETION_MANIFEST for source in sources):
        raise VerifiedCopyError(f"source prefix reserves {COMPLETION_MANIFEST!r}")

    manifest_path = _join_path(destination_root, COMPLETION_MANIFEST)
    manifest_url = _join_url(destination_url, COMPLETION_MANIFEST)
    if destination_fs.exists(manifest_path):
        manifest = _completion_manifest(_read_json(destination_fs, manifest_path), source_url, destination_url)
        verified = manifest.files
        _validate_completed_source(source_fs, sources, verified)
        return _result(manifest_url, verified, copied_files=0, resumed_files=len(verified))

    destination_files = _destination_files(destination_fs, destination_root)
    expected_paths = {source.path for source in sources}
    extras = sorted(destination_files.keys() - expected_paths)
    if extras:
        sample = ", ".join(extras[:3])
        raise VerifiedCopyError(f"destination contains {len(extras)} unexpected file(s): {sample}")

    copied_files = 0
    resumed_files = 0
    verified_files: list[VerifiedFile] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _copy_or_resume,
                source,
                source_fs=source_fs,
                destination_fs=destination_fs,
                destination_root=destination_root,
                status_fs=status_fs,
                status_root=status_root,
                destination=destination_files.get(source.path),
            ): source.path
            for source in sources
        }
        for future in as_completed(futures):
            outcome = future.result()
            verified_files.append(outcome.file)
            copied_files += int(outcome.disposition is _CopyDisposition.COPIED)
            resumed_files += int(outcome.disposition is _CopyDisposition.RESUMED)

    verified_files.sort()
    manifest = _CompletionManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source=source_url,
        destination=destination_url,
        total_files=len(verified_files),
        total_bytes=sum(file.size for file in verified_files),
        files=verified_files,
    )
    _write_json_atomic(destination_fs, manifest_path, asdict(manifest))
    return _result(manifest_url, verified_files, copied_files=copied_files, resumed_files=resumed_files)


def _source_files(filesystem: AbstractFileSystem, root: str) -> list[_SourceFile]:
    if not filesystem.exists(root) or not filesystem.isdir(root):
        raise VerifiedCopyError(f"source is not a directory: {root}")
    files = []
    for path, info in _find_files(filesystem, root).items():
        relative = _relative_path(path, root)
        files.append(_SourceFile(relative, path, int(info.get("size") or 0), _source_identity(info)))
    files.sort(key=lambda file: file.path)
    return files


def _destination_files(filesystem: AbstractFileSystem, root: str) -> dict[str, _DestinationFile]:
    if not filesystem.exists(root):
        return {}
    if not filesystem.isdir(root):
        raise VerifiedCopyError(f"destination is not a directory: {root}")
    return {
        _relative_path(path, root): _DestinationFile(
            size=int(info.get("size") or 0),
            identity=_destination_identity(info),
        )
        for path, info in _find_files(filesystem, root).items()
        if _relative_path(path, root) != COMPLETION_MANIFEST
    }


def _find_files(filesystem: AbstractFileSystem, root: str) -> dict[str, dict[str, Any]]:
    found = cast(dict[str, dict[str, Any]] | list[str], filesystem.find(root, detail=True))
    if not isinstance(found, dict):
        return {path: filesystem.info(path) for path in found if not filesystem.isdir(path)}
    return {path: info for path, info in found.items() if info.get("type") != "directory"}


def _copy_or_resume(
    source: _SourceFile,
    *,
    source_fs: AbstractFileSystem,
    destination_fs: AbstractFileSystem,
    destination_root: str,
    status_fs: AbstractFileSystem,
    status_root: str,
    destination: _DestinationFile | None,
) -> _CopyOutcome:
    destination_path = _join_path(destination_root, source.path)
    marker_path = _join_path(status_root, f"{hashlib.sha256(source.path.encode()).hexdigest()}.json")
    marker = _resume_marker(status_fs, marker_path)
    if (
        destination is not None
        and destination.size == source.size
        and source.identity is not None
        and marker is not None
    ):
        expected = VerifiedFile(source.path, source.size, marker.sha256, source.identity)
        if (
            marker.source_path == source.source_path
            and marker.source_identity == source.identity
            and marker.path == source.path
            and marker.size == source.size
            and _destination_matches_marker(destination_fs, destination_path, destination, marker)
        ):
            return _CopyOutcome(expected, _CopyDisposition.RESUMED)

    copied = _copy_with_hash(source_fs, source.source_path, destination_fs, destination_path)
    if copied.size != source.size:
        destination_fs.rm(destination_path)
        raise VerifiedCopyError(f"source size changed while copying {source.path}")
    current_info = source_fs.info(source.source_path)
    current_size = int(current_info.get("size") or 0)
    current_identity = _source_identity(current_info)
    if current_size != source.size or current_identity != source.identity:
        destination_fs.rm(destination_path)
        raise VerifiedCopyError(f"source identity changed while copying {source.path}")
    try:
        destination_identity = _verify_destination(
            destination_fs,
            destination_path,
            expected_size=source.size,
            expected_sha256=copied.sha256,
            expected_etag=copied.expected_etag,
        )
    except VerifiedCopyError:
        destination_fs.rm(destination_path)
        raise
    verified = VerifiedFile(source.path, source.size, copied.sha256, source.identity)
    _write_json_atomic(
        status_fs,
        marker_path,
        asdict(
            _ResumeMarker(
                source.source_path,
                source.identity,
                source.path,
                source.size,
                copied.sha256,
                destination_identity,
            )
        ),
    )
    return _CopyOutcome(verified, _CopyDisposition.COPIED)


def _copy_with_hash(
    source_fs: AbstractFileSystem,
    source_path: str,
    destination_fs: AbstractFileSystem,
    destination_path: str,
) -> _CopiedFile:
    parent, separator, _ = destination_path.rpartition("/")
    if separator:
        destination_fs.makedirs(parent, exist_ok=True)
    digest = hashlib.sha256()
    multipart_etag = _MultipartEtag(S3_UPLOAD_PART_BYTES) if _has_fixed_s3_uploads(destination_fs) else None
    copied_size = 0
    destination_options = {"block_size": S3_UPLOAD_PART_BYTES} if multipart_etag is not None else {}
    with (
        source_fs.open(source_path, "rb") as source,
        destination_fs.open(destination_path, "wb", **destination_options) as destination,
    ):
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            if multipart_etag is not None:
                multipart_etag.update(chunk)
            destination.write(chunk)
            copied_size += len(chunk)
    expected_etag = multipart_etag.etag(copied_size) if multipart_etag is not None else None
    return _CopiedFile(digest.hexdigest(), copied_size, expected_etag)


def _verify_destination(
    filesystem: AbstractFileSystem,
    path: str,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_etag: str | None,
) -> str:
    filesystem.invalidate_cache(path)
    info = filesystem.info(path)
    if int(info.get("size") or 0) != expected_size:
        raise VerifiedCopyError(f"destination size mismatch for {path}")
    if expected_etag is None:
        if _sha256(filesystem, path) != expected_sha256:
            raise VerifiedCopyError(f"destination hash mismatch for {path}")
        return _destination_identity(info) or f"{SHA256_IDENTITY_PREFIX}{expected_sha256}"

    actual_etag = _etag(info)
    if actual_etag != expected_etag:
        raise VerifiedCopyError(f"destination ETag mismatch for {path}")
    return f"{ETAG_IDENTITY_PREFIX}{actual_etag}"


def _destination_matches_marker(
    filesystem: AbstractFileSystem,
    path: str,
    destination: _DestinationFile,
    marker: _ResumeMarker,
) -> bool:
    if marker.destination_identity.startswith(SHA256_IDENTITY_PREFIX):
        return _sha256(filesystem, path) == marker.destination_identity.removeprefix(SHA256_IDENTITY_PREFIX)
    return destination.identity == marker.destination_identity


def _sha256(filesystem: AbstractFileSystem, path: str) -> str:
    digest = hashlib.sha256()
    with filesystem.open(path, "rb") as file:
        while chunk := file.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_marker(filesystem: AbstractFileSystem, path: str) -> _ResumeMarker | None:
    if not filesystem.exists(path):
        return None
    try:
        data = _read_json(filesystem, path)
        return _ResumeMarker(
            source_path=str(data["source_path"]),
            source_identity=str(data["source_identity"]) if data.get("source_identity") is not None else None,
            path=str(data["path"]),
            size=int(data["size"]),
            sha256=str(data["sha256"]),
            destination_identity=str(data["destination_identity"]),
        )
    except (KeyError, TypeError, ValueError, VerifiedCopyError) as error:
        logger.warning("Ignoring invalid verified-copy resume marker %s: %s", path, error)
        return None


def _resolved_location(url: str, upload_policy: S3UploadPolicy) -> TransferLocation:
    filesystem, path = filesystem_for(url, s3_upload_policy=upload_policy)
    location = TransferLocation.from_path(url, filesystem, path)
    if _backend(location) == "file":
        path = os.path.realpath(os.path.abspath(path))
        location = TransferLocation.from_path(path, filesystem, path)
    return location


def _validate_disjoint_locations(
    source: TransferLocation,
    destination: TransferLocation,
    status: TransferLocation,
) -> None:
    locations = (("source", source), ("destination", destination), ("status", status))
    for index, (left_name, left) in enumerate(locations):
        for right_name, right in locations[index + 1 :]:
            if _same_location(left, right) or _strictly_contains(left, right) or _strictly_contains(right, left):
                raise VerifiedCopyError(f"{left_name} and {right_name} prefixes overlap")


def _write_json_atomic(filesystem: AbstractFileSystem, path: str, value: dict[str, Any]) -> None:
    parent, separator, _ = path.rpartition("/")
    if separator:
        filesystem.makedirs(parent, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with atomic_rename(path, filesystem=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "wb") as file:
            file.write(payload)


def _read_json(filesystem: AbstractFileSystem, path: str) -> dict[str, Any]:
    try:
        with filesystem.open(path, "rb") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise VerifiedCopyError(f"invalid JSON at {path}") from error
    if not isinstance(value, dict):
        raise VerifiedCopyError(f"expected JSON object at {path}")
    return value


def _completion_manifest(data: dict[str, Any], source_url: str, destination_url: str) -> _CompletionManifest:
    try:
        files = [
            VerifiedFile(
                path=str(item["path"]),
                size=int(item["size"]),
                sha256=str(item["sha256"]),
                source_identity=str(item["source_identity"]) if item.get("source_identity") is not None else None,
            )
            for item in data["files"]
        ]
        manifest = _CompletionManifest(
            schema_version=int(data["schema_version"]),
            source=str(data["source"]),
            destination=str(data["destination"]),
            total_files=int(data["total_files"]),
            total_bytes=int(data["total_bytes"]),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VerifiedCopyError("completion manifest has invalid file records") from error
    files.sort()
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise VerifiedCopyError("completion manifest has an unsupported schema")
    if manifest.source != source_url or manifest.destination != destination_url:
        raise VerifiedCopyError("completion manifest belongs to a different transfer")
    if manifest.total_files != len(files) or manifest.total_bytes != sum(file.size for file in files):
        raise VerifiedCopyError("completion manifest totals do not match its file records")
    return manifest


def _validate_completed_source(
    source_fs: AbstractFileSystem, sources: list[_SourceFile], verified: list[VerifiedFile]
) -> None:
    current_inventory = [(file.path, file.size) for file in sources]
    completed_inventory = [(file.path, file.size) for file in verified]
    if current_inventory != completed_inventory:
        raise VerifiedCopyError("source path or size changed after destination completion")
    for source, completed in zip(sources, verified, strict=True):
        if source.identity is not None:
            if source.identity != completed.source_identity:
                raise VerifiedCopyError(f"source identity changed after destination completion: {source.path}")
            continue
        if _sha256(source_fs, source.source_path) != completed.sha256:
            raise VerifiedCopyError(f"source content changed after destination completion: {source.path}")


def _source_identity(info: dict[str, Any]) -> str | None:
    for key in (
        "generation",
        "version_id",
        "VersionId",
        "md5Hash",
        "crc32c",
        "checksum",
        "ChecksumSHA256",
        "etag",
        "ETag",
        "updated",
        "mtime",
        "LastModified",
    ):
        value = info.get(key)
        if value is not None:
            return f"{key}={value!s}"
    return None


def _destination_identity(info: dict[str, Any]) -> str | None:
    etag = _etag(info)
    if etag is not None:
        return f"{ETAG_IDENTITY_PREFIX}{etag}"
    for key in ("ChecksumSHA256", "checksum", "md5Hash", "crc32c", "version_id", "VersionId"):
        value = info.get(key)
        if value is not None:
            return f"{key}={value!s}"
    return None


def _etag(info: dict[str, Any]) -> str | None:
    value = info.get("etag") or info.get("ETag")
    if value is None:
        return None
    return str(value).strip('"')


def _has_fixed_s3_uploads(filesystem: AbstractFileSystem) -> bool:
    protocol = filesystem.protocol
    if isinstance(protocol, str):
        is_s3 = protocol == "s3"
    else:
        is_s3 = "s3" in protocol
    return is_s3 and bool(getattr(filesystem, "fixed_upload_size", False))


def _result(
    manifest_url: str,
    files: list[VerifiedFile],
    *,
    copied_files: int,
    resumed_files: int,
) -> VerifiedCopyResult:
    return VerifiedCopyResult(
        manifest_url=manifest_url,
        copied_files=copied_files,
        resumed_files=resumed_files,
        total_files=len(files),
        total_bytes=sum(file.size for file in files),
    )
