# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Inventory the source, then stream each object into an unpublished write at its
# final path. Publish the write only after its stream matches the source, check the
# committed object against the streamed digests, and publish the manifest last.
"""Resumable, content-verified prefix copies across routed filesystems."""

import base64
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

import google_crc32c
from fsspec import AbstractFileSystem

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import S3UploadPolicy, filesystem_for
from rigging.filesystem.protocols import normalize_protocols
from rigging.filesystem.transfer import (
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
PARTIAL_SUFFIX = ".verified-copy-partial"
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


class _DestinationKind(StrEnum):
    """How a destination keeps a write unpublished and checks it after commit."""

    S3_FIXED_PARTS = "s3_fixed_parts"
    S3 = "s3"
    GCS = "gcs"
    LOCAL = "local"


class _ChecksumAlgorithm(StrEnum):
    SHA256 = "sha256"
    MD5 = "md5"
    CRC32C = "crc32c"


@dataclass(frozen=True)
class _SourceChecksum:
    """A content digest that the source publishes, as lowercase hex."""

    algorithm: _ChecksumAlgorithm
    digest: str


@dataclass(frozen=True)
class _SourceFile:
    path: str
    source_path: str
    size: int
    identity: str | None
    checksum: _SourceChecksum | None


@dataclass(frozen=True)
class _DestinationFile:
    size: int
    identity: str | None


@dataclass(frozen=True)
class _CopiedFile:
    sha256: str
    destination_identity: str


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


class _Digest(Protocol):
    def update(self, data: bytes, /) -> object: ...

    def digest(self) -> bytes: ...


class _S3MultipartFilesystem(Protocol):
    def call_s3(self, method: str, **kwargs: object) -> dict[str, Any]: ...

    def abort_mpu(self, bucket: str, key: str, mpu: str) -> object: ...


class _Upload(Protocol):
    def write(self, data: bytes) -> None: ...

    def commit(self) -> None: ...

    def discard(self) -> None: ...


class _MultipartEtag:
    """Hash a byte stream as S3 computes the ETag of a fixed-part multipart upload.

    This deliberately implements the small streaming calculation locally: none
    of our S3 dependencies exposes it, and adding a dependency solely for this
    calculation would not improve correctness. The algorithm is specified at
    https://developers.cloudflare.com/r2/objects/upload-objects/#etags.
    """

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

    def etag(self) -> str:
        part_hashes = [*self.part_hashes]
        if self.current_size:
            part_hashes.append(self.current_hash.digest())
        if not part_hashes:
            # s3fs publishes an empty file with a one-shot PUT, whose ETag is the MD5 of no bytes.
            return hashlib.md5(b"", usedforsecurity=False).hexdigest()
        digest = hashlib.md5(b"".join(part_hashes), usedforsecurity=False).hexdigest()
        return f"{digest}-{len(part_hashes)}"


class _StreamDigest:
    """The size and digests of one copied byte stream."""

    def __init__(self, algorithms: set[_ChecksumAlgorithm], multipart_part_size: int | None):
        self.size = 0
        self._digests = {algorithm: _new_digest(algorithm) for algorithm in {_ChecksumAlgorithm.SHA256, *algorithms}}
        self._multipart = _MultipartEtag(multipart_part_size) if multipart_part_size is not None else None

    def update(self, data: bytes) -> None:
        self.size += len(data)
        for digest in self._digests.values():
            digest.update(data)
        if self._multipart is not None:
            self._multipart.update(data)

    def hex_digest(self, algorithm: _ChecksumAlgorithm) -> str:
        return self._digests[algorithm].digest().hex()

    def multipart_etag(self) -> str:
        assert self._multipart is not None, "this stream does not track a multipart ETag"
        return self._multipart.etag()


class _DeferredObjectUpload:
    """An S3 multipart or GCS resumable upload that stays invisible until committed."""

    def __init__(self, filesystem: AbstractFileSystem, path: str, kind: _DestinationKind):
        options: dict[str, int | bool] = {"autocommit": False}
        if kind is _DestinationKind.S3_FIXED_PARTS:
            options["block_size"] = S3_UPLOAD_PART_BYTES
        self._file = filesystem.open(path, "wb", **options)

    def write(self, data: bytes) -> None:
        self._file.write(data)

    def commit(self) -> None:
        # Closing an uncommitted writer uploads its buffered data without publishing it.
        self._file.close()
        self._file.commit()

    def discard(self) -> None:
        # An s3fs writer left open would start a new upload when garbage collected.
        try:
            self._file.close()
        finally:
            self._file.discard()


class _LocalUpload:
    """A local file written beside its final path and renamed into place on commit."""

    def __init__(self, filesystem: AbstractFileSystem, path: str):
        self._filesystem = filesystem
        self._path = path
        self._partial_path = f"{path}{PARTIAL_SUFFIX}"
        self._file = filesystem.open(self._partial_path, "wb")

    def write(self, data: bytes) -> None:
        self._file.write(data)

    def commit(self) -> None:
        self._file.close()
        self._filesystem.mv(self._partial_path, self._path)

    def discard(self) -> None:
        self._file.close()
        self._filesystem.rm(self._partial_path)


def verified_copy_prefix(
    source_url: str,
    destination_url: str,
    *,
    status_url: str | None = None,
    workers: int = DEFAULT_VERIFIED_COPY_WORKERS,
) -> VerifiedCopyResult:
    """Copy and verify a prefix before publishing its completion manifest.

    Each file is written once, directly to its final path, and published only after
    its whole source stream has been read and checked, so a partial file never
    appears at a final path. Files can appear before the manifest, so readers must
    require the manifest. Verified per-file records allow retries to reuse
    published files when their source and destination identities still match.
    """
    if workers < 1:
        raise VerifiedCopyError("workers must be at least 1")

    source = _resolved_location(source_url, S3UploadPolicy.STANDARD)
    destination = _resolved_location(destination_url, S3UploadPolicy.FIXED_PARTS)
    if _backend(destination) != "file" and "/" not in destination.path.rstrip("/"):
        raise VerifiedCopyError("destination must name a prefix inside an object-store bucket")
    status = (
        _resolved_location(status_url, S3UploadPolicy.FIXED_PARTS)
        if status_url is not None
        else _sibling_location(destination, ".verified-copy-status")
    )
    _validate_disjoint_locations(source, destination, status)
    kind = _destination_kind(destination.filesystem)

    source_url, source_fs, source_root = source.url, source.filesystem, source.path
    destination_url, destination_fs, destination_root = destination.url, destination.filesystem, destination.path
    status_fs, status_root = status.filesystem, status.path
    sources = _source_files(source_fs, source_root)
    if not sources:
        raise VerifiedCopyError(f"source prefix contains no files: {source_url}")
    reserved = [file.path for file in sources if file.path == COMPLETION_MANIFEST or file.path.endswith(PARTIAL_SUFFIX)]
    if reserved:
        raise VerifiedCopyError(f"source prefix uses a name reserved by verified copies: {reserved[0]}")

    manifest_path = _join_path(destination_root, COMPLETION_MANIFEST)
    manifest_url = _join_url(destination_url, COMPLETION_MANIFEST)
    if destination_fs.exists(manifest_path):
        manifest = _completion_manifest(_read_json(destination_fs, manifest_path), source_url, destination_url)
        verified = manifest.files
        _validate_completed_source(source_fs, sources, verified)
        return _result(manifest_url, verified, copied_files=0, resumed_files=len(verified))

    destination_files = _destination_files(destination_fs, destination_root)
    partial_paths = sorted(path for path in destination_files if path.endswith(PARTIAL_SUFFIX))
    expected_paths = {source.path for source in sources}
    extras = sorted(destination_files.keys() - expected_paths - set(partial_paths))
    if extras:
        sample = ", ".join(extras[:3])
        raise VerifiedCopyError(f"destination contains {len(extras)} unexpected file(s): {sample}")
    _discard_interrupted_writes(destination_fs, destination_root, partial_paths, kind)

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
                kind=kind,
                status_fs=status_fs,
                status_root=status_root,
                published=destination_files.get(source.path),
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
        files.append(
            _SourceFile(relative, path, int(info.get("size") or 0), _source_identity(info), _source_checksum(info))
        )
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


def _discard_interrupted_writes(
    filesystem: AbstractFileSystem,
    root: str,
    partial_paths: list[str],
    kind: _DestinationKind,
) -> None:
    """Remove the local partial files and S3 multipart uploads that a killed copy left."""
    for path in partial_paths:
        filesystem.rm(_join_path(root, path))
    if kind in (_DestinationKind.S3_FIXED_PARTS, _DestinationKind.S3):
        _abort_unfinished_uploads(cast(_S3MultipartFilesystem, filesystem), root)


def _abort_unfinished_uploads(filesystem: _S3MultipartFilesystem, root: str) -> None:
    """Abort every unfinished multipart upload below ``root``.

    A destination prefix belongs to one verified copy at a time, so an unfinished
    upload below it was left by an earlier attempt that could not abort it.
    """
    bucket, _, key = root.partition("/")
    request: dict[str, object] = {"Bucket": bucket, "Prefix": f"{key.rstrip('/')}/"}
    while True:
        page = filesystem.call_s3("list_multipart_uploads", **request)
        for upload in page.get("Uploads", []):
            logger.info("Aborting unfinished upload of s3://%s/%s", bucket, upload["Key"])
            filesystem.abort_mpu(bucket, upload["Key"], upload["UploadId"])
        if not page.get("IsTruncated"):
            return
        request["KeyMarker"] = page["NextKeyMarker"]
        request["UploadIdMarker"] = page["NextUploadIdMarker"]


def _copy_or_resume(
    source: _SourceFile,
    *,
    source_fs: AbstractFileSystem,
    destination_fs: AbstractFileSystem,
    destination_root: str,
    kind: _DestinationKind,
    status_fs: AbstractFileSystem,
    status_root: str,
    published: _DestinationFile | None,
) -> _CopyOutcome:
    destination_path = _join_path(destination_root, source.path)
    marker_path = _join_path(status_root, f"{hashlib.sha256(source.path.encode()).hexdigest()}.json")
    marker = _resume_marker(status_fs, marker_path)
    if (
        source.identity is not None
        and marker is not None
        and (marker.source_path, marker.source_identity, marker.path, marker.size)
        == (source.source_path, source.identity, source.path, source.size)
        and published is not None
        and published.size == source.size
        and _destination_matches_marker(destination_fs, destination_path, published, marker)
    ):
        return _CopyOutcome(
            VerifiedFile(source.path, source.size, marker.sha256, source.identity),
            _CopyDisposition.RESUMED,
        )

    copied = _copy_file(source, source_fs=source_fs, destination_fs=destination_fs, path=destination_path, kind=kind)
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
                copied.destination_identity,
            )
        ),
    )
    return _CopyOutcome(
        VerifiedFile(source.path, source.size, copied.sha256, source.identity),
        _CopyDisposition.COPIED,
    )


def _copy_file(
    source: _SourceFile,
    *,
    source_fs: AbstractFileSystem,
    destination_fs: AbstractFileSystem,
    path: str,
    kind: _DestinationKind,
) -> _CopiedFile:
    """Stream ``source`` to ``path`` and publish it only after the stream is verified."""
    algorithms = {source.checksum.algorithm} if source.checksum is not None else set()
    if kind is _DestinationKind.GCS:
        algorithms.add(_ChecksumAlgorithm.MD5)
    digest = _StreamDigest(algorithms, S3_UPLOAD_PART_BYTES if kind is _DestinationKind.S3_FIXED_PARTS else None)
    with source_fs.open(source.source_path, "rb") as reader:
        upload = _open_upload(destination_fs, path, kind)
        try:
            while chunk := reader.read(COPY_CHUNK_BYTES):
                digest.update(chunk)
                upload.write(chunk)
            _check_streamed_source(source_fs, source, digest)
            upload.commit()
        except BaseException:
            upload.discard()
            raise
    try:
        identity = _committed_identity(destination_fs, path, digest, kind)
    except VerifiedCopyError:
        destination_fs.rm(path)
        raise
    return _CopiedFile(digest.hex_digest(_ChecksumAlgorithm.SHA256), identity)


def _open_upload(filesystem: AbstractFileSystem, path: str, kind: _DestinationKind) -> _Upload:
    parent, separator, _ = path.rpartition("/")
    if separator:
        filesystem.makedirs(parent, exist_ok=True)
    if kind is _DestinationKind.LOCAL:
        return _LocalUpload(filesystem, path)
    return _DeferredObjectUpload(filesystem, path, kind)


def _check_streamed_source(source_fs: AbstractFileSystem, source: _SourceFile, digest: _StreamDigest) -> None:
    if digest.size != source.size:
        raise VerifiedCopyError(f"source size changed while copying {source.path}")
    checksum = source.checksum
    if checksum is not None and digest.hex_digest(checksum.algorithm) != checksum.digest:
        raise VerifiedCopyError(f"source {checksum.algorithm} does not match its published checksum: {source.path}")
    source_fs.invalidate_cache(source.source_path)
    current = source_fs.info(source.source_path)
    if int(current.get("size") or 0) != source.size or _source_identity(current) != source.identity:
        raise VerifiedCopyError(f"source identity changed while copying {source.path}")


def _committed_identity(
    filesystem: AbstractFileSystem,
    path: str,
    digest: _StreamDigest,
    kind: _DestinationKind,
) -> str:
    """Check a committed object against its streamed digests and return its resume identity."""
    filesystem.invalidate_cache(path)
    info = filesystem.info(path)
    if int(info.get("size") or 0) != digest.size:
        raise VerifiedCopyError(f"destination size mismatch for {path}")
    sha256 = digest.hex_digest(_ChecksumAlgorithm.SHA256)
    if kind is _DestinationKind.S3_FIXED_PARTS:
        if _etag(info) != digest.multipart_etag():
            raise VerifiedCopyError(f"destination ETag mismatch for {path}")
    elif kind is _DestinationKind.GCS:
        md5 = info.get("md5Hash")
        if md5 is None or _base64_hex(md5) != digest.hex_digest(_ChecksumAlgorithm.MD5):
            raise VerifiedCopyError(f"destination MD5 mismatch for {path}")
    elif _sha256(filesystem, path) != sha256:
        raise VerifiedCopyError(f"destination hash mismatch for {path}")
    return _destination_identity(info) or f"{SHA256_IDENTITY_PREFIX}{sha256}"


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


def _new_digest(algorithm: _ChecksumAlgorithm) -> _Digest:
    if algorithm is _ChecksumAlgorithm.SHA256:
        return hashlib.sha256()
    if algorithm is _ChecksumAlgorithm.MD5:
        return hashlib.md5(usedforsecurity=False)
    return google_crc32c.Checksum()


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


def _destination_kind(filesystem: AbstractFileSystem) -> _DestinationKind:
    protocols = normalize_protocols(filesystem.protocol)
    if "s3" in protocols:
        if getattr(filesystem, "fixed_upload_size", False):
            return _DestinationKind.S3_FIXED_PARTS
        return _DestinationKind.S3
    if "gs" in protocols or "gcs" in protocols:
        return _DestinationKind.GCS
    if "file" in protocols:
        return _DestinationKind.LOCAL
    raise VerifiedCopyError(f"verified copies cannot hold back publication on {protocols[0]}:// destinations")


def _sibling_location(location: TransferLocation, suffix: str) -> TransferLocation:
    return TransferLocation(
        f"{location.url.rstrip('/')}{suffix}",
        location.filesystem,
        f"{location.path.rstrip('/')}{suffix}",
    )


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
        "blob_id",
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


def _source_checksum(info: dict[str, Any]) -> _SourceChecksum | None:
    """Return the content digest that a Hugging Face or GCS listing publishes."""
    lfs = info.get("lfs")
    if lfs is not None:
        return _SourceChecksum(_ChecksumAlgorithm.SHA256, str(lfs["sha256"]))
    for key, algorithm in (("md5Hash", _ChecksumAlgorithm.MD5), ("crc32c", _ChecksumAlgorithm.CRC32C)):
        value = info.get(key)
        if value is not None:
            return _SourceChecksum(algorithm, _base64_hex(value))
    return None


def _base64_hex(value: str) -> str:
    return base64.b64decode(value).hex()


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
