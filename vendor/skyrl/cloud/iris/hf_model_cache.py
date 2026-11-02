"""Immutable, manifest-verified Hugging Face mirrors for Iris model loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import posixpath
import time
from typing import Literal

from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfApi, HfFileSystem, snapshot_download
from huggingface_hub.hf_api import RepoFile
from loguru import logger
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.distributed_lock import create_lock, lease_refresh

from cloud.iris.artifacts import (
    ArtifactSource,
    atomic_directory_update,
    file_inventory,
    fs_and_path,
    materialize_inventory,
    read_json,
)
from marinskyrl.hf_model import (
    TOKENIZER_CONFIG_NAME,
    TOKENIZER_JSON_NAME,
    hugging_face_hub_online,
    immutable_model_cache_key,
    normalize_fast_tokenizer_metadata_bytes,
    validate_portable_hf_model_files,
)
from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    MODEL_MANIFEST_FILENAME,
    ModelManifest,
    ModelManifestFile,
    build_model_manifest,
    build_safetensors_weight_index,
    read_safetensors_header,
    sha256_file,
    validate_model_file_names,
    validate_model_file_path,
    validate_sha256_digest,
)
from marinskyrl.hugging_face_retry import (
    call_with_hugging_face_retry,
    load_hugging_face_with_retry,
)
from marinskyrl.remote_io import open_output_stream
from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from marinskyrl.speculative_decoding import is_hugging_face_commit

_CACHE_PREFIX = "marinskyrl/hf-models"
_CACHE_POLL_INTERVAL = 10.0
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_STREAM_CHUNK_BYTES = 8 * 2**20
_MAX_METADATA_BYTES = 256 * 2**20
_MODEL_CONFIG_NAME = "config.json"
_MODEL_METADATA_FILES = (_MODEL_CONFIG_NAME, TOKENIZER_JSON_NAME, TOKENIZER_CONFIG_NAME)


def _is_safetensors(path: str) -> bool:
    return path.endswith(".safetensors")


@dataclass(frozen=True)
class HuggingFaceSnapshotFile:
    """One immutable file advertised by a pinned Hugging Face revision."""

    path: str
    size: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        validate_model_file_path(self.path)
        if self.size < 0:
            raise ValueError(f"Hugging Face snapshot size must be non-negative: {self.path}")
        if self.sha256 is not None:
            validate_sha256_digest(self.sha256)


@dataclass(frozen=True)
class HuggingFaceSnapshot:
    filesystem: AbstractFileSystem
    root: str
    files: tuple[HuggingFaceSnapshotFile, ...]


@dataclass(frozen=True)
class _MirroredSnapshotFile:
    entry: ModelManifestFile
    tensor_keys: tuple[str, ...]
    reused: bool


@dataclass(frozen=True)
class _MirrorTransferStats:
    streamed_files: int
    streamed_bytes: int
    streamed_weight_bytes: int
    reused_files: int
    reused_bytes: int
    reused_weight_bytes: int


@dataclass(frozen=True)
class _MirroredSnapshot:
    files: dict[str, ModelManifestFile]
    shard_headers: dict[str, tuple[int, tuple[str, ...]]]
    stats: _MirrorTransferStats


def load_model_manifest(model_uri: str) -> ModelManifest:
    marker_uri = join_resource_path(model_uri, MODEL_MANIFEST_FILENAME)
    return ModelManifest.from_mapping(read_json(marker_uri), marker_uri)


def _cached_manifest(
    cache_uri: str,
    model_id: str,
    revision: str,
    tokenizer_mode: Literal["embedded", "policy"],
) -> ModelManifest | None:
    marker_uri = join_resource_path(cache_uri, MODEL_MANIFEST_FILENAME)
    filesystem, marker_path = fs_and_path(marker_uri)
    if not filesystem.exists(marker_path):
        return None
    # The manifest is the completion record. A malformed or mismatched marker
    # identifies a partial cache that the next lock holder can rebuild.
    try:
        manifest = load_model_manifest(cache_uri)
    except ValueError:
        return None
    if manifest.model_id != model_id or manifest.revision != revision or manifest.tokenizer_mode != tokenizer_mode:
        return None
    return manifest


def download_hugging_face_snapshot(
    model_id: str,
    *,
    revision: str | None,
    destination: Path | None = None,
    allow_patterns: tuple[str, ...] = (),
) -> Path:
    """Download a Hub snapshot while preserving the runtime's offline environment."""
    with hugging_face_hub_online():
        snapshot = load_hugging_face_with_retry(
            lambda: snapshot_download(
                model_id,
                revision=revision,
                local_dir=destination,
                allow_patterns=list(allow_patterns) or None,
            ),
            resource_id=model_id,
            resource_kind="model snapshot",
        )
    return Path(snapshot)


def resolve_hugging_face_revision(model_id: str, revision: str | None) -> str:
    """Resolve a branch, tag, or omitted revision to one immutable Hub commit."""
    if revision and is_hugging_face_commit(revision):
        return revision
    with hugging_face_hub_online():
        resolved = call_with_hugging_face_retry(
            lambda: HfApi().model_info(model_id, revision=revision).sha,
            operation=f"resolve Hugging Face revision {model_id}@{revision or 'main'}",
        )
    if not resolved or not is_hugging_face_commit(resolved):
        raise ValueError(f"Hugging Face did not resolve {model_id}@{revision or 'main'} to a commit")
    return resolved


def _open_hugging_face_snapshot(
    model_id: str,
    revision: str,
) -> HuggingFaceSnapshot:
    """Open and inventory one pinned Hub tree without downloading its files."""
    with hugging_face_hub_online():
        entries = call_with_hugging_face_retry(
            lambda: tuple(HfApi().list_repo_tree(model_id, recursive=True, revision=revision)),
            operation=f"list Hugging Face snapshot {model_id}@{revision}",
        )
        filesystem = HfFileSystem(block_size=0)
    files = tuple(
        HuggingFaceSnapshotFile(
            path=entry.path,
            size=entry.size,
            sha256=entry.lfs.sha256 if entry.lfs is not None else None,
        )
        for entry in entries
        if isinstance(entry, RepoFile)
    )
    if not files:
        raise ValueError(f"Hugging Face snapshot contains no files: {model_id}@{revision}")
    return HuggingFaceSnapshot(filesystem=filesystem, root=f"{model_id}@{revision}", files=files)


def _destination_matches(
    filesystem: AbstractFileSystem,
    path: str,
    *,
    size: int,
    sha256: str | None,
) -> bool:
    if sha256 is None or not filesystem.exists(path):
        return False
    info = filesystem.info(path)
    if int(info["size"]) != size:
        return False

    def hash_existing() -> str:
        digest = hashlib.sha256()
        with filesystem.open(path, "rb") as source:
            while chunk := source.read(_STREAM_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    return hash_existing() == sha256


def _read_source_bytes(
    filesystem: AbstractFileSystem,
    path: str,
    *,
    expected_size: int,
) -> bytes:
    if expected_size > _MAX_METADATA_BYTES:
        raise ValueError(f"Model metadata exceeds {_MAX_METADATA_BYTES} bytes: {path}")

    def read() -> bytes:
        chunks = []
        total = 0
        with filesystem.open(path, "rb") as source:
            while chunk := source.read(min(_STREAM_CHUNK_BYTES, _MAX_METADATA_BYTES - total + 1)):
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_METADATA_BYTES:
                    raise ValueError(f"Model metadata exceeds {_MAX_METADATA_BYTES} bytes: {path}")
        payload = b"".join(chunks)
        if len(payload) != expected_size:
            raise ValueError(f"Hugging Face size mismatch for {path}: expected {expected_size}, found {len(payload)}")
        return payload

    return call_with_hugging_face_retry(read, operation=f"read Hugging Face metadata {path}")


def _write_bytes(filesystem: AbstractFileSystem, path: str, payload: bytes) -> None:
    with open_output_stream(filesystem, path) as destination:
        destination.write(payload)


def _publish_bytes(
    filesystem: AbstractFileSystem,
    root: str,
    path: str,
    payload: bytes,
) -> ModelManifestFile:
    digest = hashlib.sha256(payload).hexdigest()
    destination_path = posixpath.join(root, path)
    if not _destination_matches(filesystem, destination_path, size=len(payload), sha256=digest):
        _write_bytes(filesystem, destination_path, payload)
    return ModelManifestFile(path=path, size=len(payload), sha256=digest)


def _stream_snapshot_file(
    source: HuggingFaceSnapshot,
    snapshot_file: HuggingFaceSnapshotFile,
    destination_filesystem: AbstractFileSystem,
    destination_path: str,
) -> _MirroredSnapshotFile:
    """Mirror one file without a disk copy and optionally collect tensor names."""
    source_path = posixpath.join(source.root, snapshot_file.path)
    is_safetensors = _is_safetensors(snapshot_file.path)
    if _destination_matches(
        destination_filesystem,
        destination_path,
        size=snapshot_file.size,
        sha256=snapshot_file.sha256,
    ):
        if is_safetensors:
            with destination_filesystem.open(destination_path, "rb") as existing:
                _header, keys = read_safetensors_header(existing, destination_path)
        else:
            keys = ()
        return _MirroredSnapshotFile(
            entry=ModelManifestFile(path=snapshot_file.path, size=snapshot_file.size, sha256=snapshot_file.sha256),
            tensor_keys=keys,
            reused=True,
        )

    def transfer() -> tuple[ModelManifestFile, tuple[str, ...]]:
        digest = hashlib.sha256()
        transferred = 0
        keys: tuple[str, ...] = ()
        with source.filesystem.open(source_path, "rb") as input_stream:
            with open_output_stream(destination_filesystem, destination_path) as destination:
                if is_safetensors:
                    header, keys = read_safetensors_header(input_stream, source_path)
                    destination.write(header)
                    digest.update(header)
                    transferred += len(header)
                while chunk := input_stream.read(_STREAM_CHUNK_BYTES):
                    destination.write(chunk)
                    digest.update(chunk)
                    transferred += len(chunk)
                if transferred != snapshot_file.size:
                    raise ValueError(
                        f"Hugging Face size mismatch for {snapshot_file.path}: "
                        f"expected {snapshot_file.size}, found {transferred}"
                    )
                actual_sha256 = digest.hexdigest()
                if snapshot_file.sha256 is not None and actual_sha256 != snapshot_file.sha256:
                    raise ValueError(
                        f"Hugging Face checksum mismatch for {snapshot_file.path}: "
                        f"expected {snapshot_file.sha256}, found {actual_sha256}"
                    )
        return ModelManifestFile(path=snapshot_file.path, size=transferred, sha256=actual_sha256), keys

    entry, keys = call_with_hugging_face_retry(
        transfer,
        operation=f"stream Hugging Face file {snapshot_file.path}",
    )
    return _MirroredSnapshotFile(entry=entry, tensor_keys=keys, reused=False)


def _remove_unexpected_files(filesystem: AbstractFileSystem, root: str, expected_paths: set[str]) -> None:
    found = filesystem.find(root, detail=True)
    for path, info in found.items():
        if info["type"] != "file":
            continue
        relative = relative_resource_path(root, path)
        if relative not in expected_paths:
            filesystem.rm(path)


def _mirror_snapshot_files(
    source: HuggingFaceSnapshot,
    files_by_path: dict[str, HuggingFaceSnapshotFile],
    destination_filesystem: AbstractFileSystem,
    destination_root: str,
) -> _MirroredSnapshot:
    manifest_files: dict[str, ModelManifestFile] = {}
    shard_headers: dict[str, tuple[int, tuple[str, ...]]] = {}
    deferred_metadata = {*_MODEL_METADATA_FILES, HF_WEIGHT_INDEX_FILENAME}
    streamed_files = 0
    streamed_bytes = 0
    streamed_weight_bytes = 0
    reused_files = 0
    reused_bytes = 0
    reused_weight_bytes = 0

    for path, snapshot_file in sorted(files_by_path.items()):
        if path in deferred_metadata:
            continue
        destination_path = posixpath.join(destination_root, path)
        mirrored = _stream_snapshot_file(
            source,
            snapshot_file,
            destination_filesystem,
            destination_path,
        )
        manifest_files[path] = mirrored.entry
        if mirrored.reused:
            reused_files += 1
            reused_bytes += mirrored.entry.size
            if _is_weight(path):
                reused_weight_bytes += mirrored.entry.size
        else:
            streamed_files += 1
            streamed_bytes += mirrored.entry.size
            if _is_weight(path):
                streamed_weight_bytes += mirrored.entry.size
        if _is_safetensors(path):
            shard_headers[path] = (mirrored.entry.size, mirrored.tensor_keys)

    stats = _MirrorTransferStats(
        streamed_files=streamed_files,
        streamed_bytes=streamed_bytes,
        streamed_weight_bytes=streamed_weight_bytes,
        reused_files=reused_files,
        reused_bytes=reused_bytes,
        reused_weight_bytes=reused_weight_bytes,
    )
    return _MirroredSnapshot(files=manifest_files, shard_headers=shard_headers, stats=stats)


def _publish_snapshot_metadata(
    source: HuggingFaceSnapshot,
    files_by_path: dict[str, HuggingFaceSnapshotFile],
    destination_filesystem: AbstractFileSystem,
    destination_root: str,
    *,
    model_source: str,
) -> dict[str, ModelManifestFile]:
    manifest_files = {}

    for path in _MODEL_METADATA_FILES:
        snapshot_file = files_by_path.get(path)
        if snapshot_file is None:
            continue
        payload = _read_source_bytes(
            source.filesystem,
            posixpath.join(source.root, path),
            expected_size=snapshot_file.size,
        )
        if path in {_MODEL_CONFIG_NAME, TOKENIZER_JSON_NAME}:
            json.loads(payload)
        if path == TOKENIZER_CONFIG_NAME:
            payload = normalize_fast_tokenizer_metadata_bytes(
                payload,
                has_tokenizer_json=TOKENIZER_JSON_NAME in files_by_path,
                source=model_source,
            )
        manifest_files[path] = _publish_bytes(destination_filesystem, destination_root, path, payload)
    return manifest_files


def _publish_weight_index(
    source: HuggingFaceSnapshot,
    files_by_path: dict[str, HuggingFaceSnapshotFile],
    destination_filesystem: AbstractFileSystem,
    destination_root: str,
    shard_headers: dict[str, tuple[int, tuple[str, ...]]],
    *,
    model_source: str,
) -> ModelManifestFile:
    """Validate or synthesize the mirrored Transformers weight index."""

    index_file = files_by_path.get(HF_WEIGHT_INDEX_FILENAME)
    existing_index = None
    if index_file is not None:
        existing_index = _read_source_bytes(
            source.filesystem,
            posixpath.join(source.root, HF_WEIGHT_INDEX_FILENAME),
            expected_size=index_file.size,
        )
    index_bytes = build_safetensors_weight_index(
        shard_headers,
        f"{model_source}/{HF_WEIGHT_INDEX_FILENAME}",
        existing=existing_index,
    )
    return _publish_bytes(destination_filesystem, destination_root, HF_WEIGHT_INDEX_FILENAME, index_bytes)


def publish_hugging_face_snapshot(
    snapshot: HuggingFaceSnapshot,
    destination_uri: str,
    *,
    model_id: str,
    revision: str,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> ModelManifest:
    """Stream a pinned Hub snapshot into object storage and publish its manifest last."""
    model_source = f"{model_id}@{revision}"
    files_by_path = {entry.path: entry for entry in snapshot.files}
    if len(files_by_path) != len(snapshot.files):
        raise ValueError(f"Hugging Face snapshot contains duplicate paths: {model_source}")
    if MODEL_MANIFEST_FILENAME in files_by_path:
        raise ValueError(f"Hugging Face snapshot uses reserved path {MODEL_MANIFEST_FILENAME}")
    validate_model_file_names(set(files_by_path), model_source, tokenizer_mode)

    destination_filesystem, destination_root = fs_and_path(destination_uri)
    destination_filesystem.makedirs(destination_root, exist_ok=True)
    marker_path = posixpath.join(destination_root, MODEL_MANIFEST_FILENAME)
    if destination_filesystem.exists(marker_path):
        destination_filesystem.rm(marker_path)

    mirrored = _mirror_snapshot_files(
        snapshot,
        files_by_path,
        destination_filesystem,
        destination_root,
    )
    manifest_files = dict(mirrored.files)
    manifest_files.update(
        _publish_snapshot_metadata(
            snapshot,
            files_by_path,
            destination_filesystem,
            destination_root,
            model_source=model_source,
        )
    )
    manifest_files[HF_WEIGHT_INDEX_FILENAME] = _publish_weight_index(
        snapshot,
        files_by_path,
        destination_filesystem,
        destination_root,
        mirrored.shard_headers,
        model_source=model_source,
    )

    manifest = build_model_manifest(
        tuple(manifest_files[path] for path in sorted(manifest_files)),
        model_id,
        revision,
        tokenizer_mode=tokenizer_mode,
    )
    expected_paths = set(manifest_files) | {MODEL_MANIFEST_FILENAME}
    _remove_unexpected_files(destination_filesystem, destination_root, expected_paths)
    manifest_bytes = (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    _write_bytes(destination_filesystem, marker_path, manifest_bytes)
    logger.info(
        "Hugging Face mirror published: model={} revision={} uri={} files={} artifact_bytes={} "
        "streamed_files={} streamed_bytes={} streamed_weight_bytes={} reused_files={} reused_bytes={} "
        "reused_weight_bytes={} local_weight_disk_bytes=0 identity={}",
        model_id,
        revision,
        destination_uri,
        len(manifest.files),
        sum(entry.size for entry in manifest.files),
        mirrored.stats.streamed_files,
        mirrored.stats.streamed_bytes,
        mirrored.stats.streamed_weight_bytes,
        mirrored.stats.reused_files,
        mirrored.stats.reused_bytes,
        mirrored.stats.reused_weight_bytes,
        manifest.identity,
    )
    return manifest


def ensure_hugging_face_model_cache(
    model_id: str,
    revision: str,
    *,
    ttl_days: int,
    source_prefix: str,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> tuple[str, ModelManifest]:
    """Mirror one immutable Hub snapshot once and return its URI and manifest."""
    revision = resolve_hugging_face_revision(model_id, revision)
    cache_identity = model_id if tokenizer_mode == "embedded" else f"{model_id}#tokenizer={tokenizer_mode}"
    cache_uri = marin_temp_bucket(
        ttl_days,
        prefix=f"{_CACHE_PREFIX}/{immutable_model_cache_key(cache_identity, revision)}",
        source_prefix=source_prefix,
    ).rstrip("/")
    if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
        return cache_uri, manifest

    lock = create_lock(f"{cache_uri}.lock")
    while not lock.try_acquire():
        if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
            return cache_uri, manifest
        time.sleep(_CACHE_POLL_INTERVAL)

    try:
        if manifest := _cached_manifest(cache_uri, model_id, revision, tokenizer_mode):
            return cache_uri, manifest
        with lease_refresh(lock):
            with hugging_face_hub_online():
                snapshot = _open_hugging_face_snapshot(model_id, revision)
                manifest = publish_hugging_face_snapshot(
                    snapshot,
                    cache_uri,
                    model_id=model_id,
                    revision=revision,
                    tokenizer_mode=tokenizer_mode,
                )
        return cache_uri, manifest
    finally:
        lock.release()


def _is_weight(path: str) -> bool:
    return path.endswith(_WEIGHT_SUFFIXES)


def stage_model_metadata(model_uri: str, manifest: ModelManifest, local_path: str) -> None:
    """Cache verified model metadata locally while excluding all weight shards."""
    target = Path(local_path)
    metadata_files = tuple(entry for entry in manifest.files if not _is_weight(entry.path))
    if not metadata_files:
        raise ValueError(f"Model manifest contains no metadata: {model_uri}")

    def matches() -> bool:
        expected_paths = {entry.path for entry in metadata_files} | {MODEL_MANIFEST_FILENAME}
        actual_paths = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
        return actual_paths == expected_paths and all(
            (path := target / entry.path).is_file()
            and path.stat().st_size == entry.size
            and sha256_file(path) == entry.sha256
            for entry in metadata_files
        )

    if matches():
        return
    with atomic_directory_update(target, staging_prefix=f".{target.name}.metadata-") as staging:
        staging.mkdir()
        filesystem, root = fs_and_path(model_uri)
        for entry in metadata_files:
            destination = staging / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(posixpath.join(root, entry.path), str(destination))
            if destination.stat().st_size != entry.size or sha256_file(destination) != entry.sha256:
                raise ValueError(f"Model metadata checksum mismatch for {entry.path}: {model_uri}")
        (staging / MODEL_MANIFEST_FILENAME).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )


def stage_artifact_model_metadata(model_uri: str, source_identity: str, local_path: str) -> int:
    """Stage metadata for a Marin-owned immutable model artifact without copying its weights."""
    filesystem, root = fs_and_path(model_uri)
    inventory = file_inventory(filesystem, root)
    validate_portable_hf_model_files({entry.path for _, entry in inventory}, model_uri)
    metadata_inventory = tuple(item for item in inventory if not _is_weight(item[1].path))
    if not metadata_inventory:
        raise ValueError(f"Model artifact contains no metadata: {model_uri}")
    artifact = materialize_inventory(
        ArtifactSource(uri=model_uri, identity=source_identity, local_path=local_path),
        filesystem,
        metadata_inventory,
    )
    return sum(entry.size for entry in artifact.files)


def stage_artifact_model(model_uri: str, source_identity: str, local_path: str) -> int:
    """Materialize a complete Marin-owned immutable model artifact on one node."""
    filesystem, root = fs_and_path(model_uri)
    inventory = file_inventory(filesystem, root)
    validate_portable_hf_model_files({entry.path for _, entry in inventory}, model_uri)
    artifact = materialize_inventory(
        ArtifactSource(uri=model_uri, identity=source_identity, local_path=local_path),
        filesystem,
        inventory,
    )
    return sum(entry.size for entry in artifact.files)
