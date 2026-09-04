# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable, content-verified prefix copies across routed filesystems."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, cast

from fsspec import AbstractFileSystem

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.transfer import _join_path, _join_url, _relative_path

COPY_CHUNK_BYTES = 8 * 1024 * 1024
COMPLETION_MANIFEST = ".verified-copy-manifest.json"
MANIFEST_SCHEMA_VERSION = 1


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
class _ResumeMarker:
    source_path: str
    source_identity: str | None
    path: str
    size: int
    sha256: str


def verified_copy_prefix(
    source_url: str,
    destination_url: str,
    *,
    status_url: str | None = None,
    workers: int = 4,
) -> VerifiedCopyResult:
    """Copy and verify a prefix before publishing its completion manifest.

    Verified per-object records allow retries to reuse destination objects when
    their source identity and content hash still match.
    """
    if workers < 1:
        raise VerifiedCopyError("workers must be at least 1")

    source_url = source_url.rstrip("/")
    destination_url = destination_url.rstrip("/")
    if source_url == destination_url:
        raise VerifiedCopyError("source and destination must differ")
    status_url = (status_url or f"{destination_url}.verified-copy-status").rstrip("/")

    source_fs, source_root = filesystem_for(source_url)
    destination_fs, destination_root = filesystem_for(destination_url, fixed_upload_size=True)
    status_fs, status_root = filesystem_for(status_url, fixed_upload_size=True)
    sources = _source_files(source_fs, source_root)
    if not sources:
        raise VerifiedCopyError(f"source prefix contains no files: {source_url}")
    if any(source.path == COMPLETION_MANIFEST for source in sources):
        raise VerifiedCopyError(f"source prefix reserves {COMPLETION_MANIFEST!r}")

    manifest_path = _join_path(destination_root, COMPLETION_MANIFEST)
    manifest_url = _join_url(destination_url, COMPLETION_MANIFEST)
    if destination_fs.exists(manifest_path):
        manifest = _read_json(destination_fs, manifest_path)
        verified = _verified_files_from_manifest(manifest, source_url, destination_url)
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
                destination_size=destination_files.get(source.path),
            ): source.path
            for source in sources
        }
        for future in as_completed(futures):
            verified, copied = future.result()
            verified_files.append(verified)
            copied_files += int(copied)
            resumed_files += int(not copied)

    verified_files.sort()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": source_url,
        "destination": destination_url,
        "total_files": len(verified_files),
        "total_bytes": sum(file.size for file in verified_files),
        "files": [asdict(file) for file in verified_files],
    }
    _write_json_atomic(destination_fs, manifest_path, manifest)
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


def _destination_files(filesystem: AbstractFileSystem, root: str) -> dict[str, int]:
    if not filesystem.exists(root):
        return {}
    if not filesystem.isdir(root):
        raise VerifiedCopyError(f"destination is not a directory: {root}")
    return {
        _relative_path(path, root): int(info.get("size") or 0)
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
    destination_size: int | None,
) -> tuple[VerifiedFile, bool]:
    destination_path = _join_path(destination_root, source.path)
    marker_path = _join_path(status_root, f"{hashlib.sha256(source.path.encode()).hexdigest()}.json")
    marker = _resume_marker(status_fs, marker_path)
    if destination_size == source.size and source.identity is not None and marker is not None:
        expected = VerifiedFile(source.path, source.size, marker.sha256, source.identity)
        if (
            marker.source_path == source.source_path
            and marker.source_identity == source.identity
            and marker.path == source.path
            and marker.size == source.size
            and _sha256(destination_fs, destination_path) == marker.sha256
        ):
            return expected, False

    sha256, copied_size = _copy_with_hash(source_fs, source.source_path, destination_fs, destination_path)
    if copied_size != source.size:
        destination_fs.rm(destination_path)
        raise VerifiedCopyError(f"source size changed while copying {source.path}")
    current_info = source_fs.info(source.source_path)
    current_size = int(current_info.get("size") or 0)
    current_identity = _source_identity(current_info)
    if current_size != source.size or current_identity != source.identity:
        destination_fs.rm(destination_path)
        raise VerifiedCopyError(f"source identity changed while copying {source.path}")
    destination_sha256 = _sha256(destination_fs, destination_path)
    if destination_sha256 != sha256:
        destination_fs.rm(destination_path)
        raise VerifiedCopyError(f"destination hash mismatch for {source.path}")
    verified = VerifiedFile(source.path, source.size, sha256, source.identity)
    _write_json_atomic(
        status_fs,
        marker_path,
        asdict(_ResumeMarker(source.source_path, source.identity, source.path, source.size, sha256)),
    )
    return verified, True


def _copy_with_hash(
    source_fs: AbstractFileSystem,
    source_path: str,
    destination_fs: AbstractFileSystem,
    destination_path: str,
) -> tuple[str, int]:
    parent, separator, _ = destination_path.rpartition("/")
    if separator:
        destination_fs.makedirs(parent, exist_ok=True)
    digest = hashlib.sha256()
    copied_size = 0
    with source_fs.open(source_path, "rb") as source, destination_fs.open(destination_path, "wb") as destination:
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            destination.write(chunk)
            copied_size += len(chunk)
    return digest.hexdigest(), copied_size


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
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


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


def _verified_files_from_manifest(manifest: dict[str, Any], source_url: str, destination_url: str) -> list[VerifiedFile]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise VerifiedCopyError("completion manifest has an unsupported schema")
    if manifest.get("source") != source_url or manifest.get("destination") != destination_url:
        raise VerifiedCopyError("completion manifest belongs to a different transfer")
    try:
        files = [
            VerifiedFile(
                path=str(item["path"]),
                size=int(item["size"]),
                sha256=str(item["sha256"]),
                source_identity=str(item["source_identity"]) if item.get("source_identity") is not None else None,
            )
            for item in manifest["files"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise VerifiedCopyError("completion manifest has invalid file records") from error
    files.sort()
    return files


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
