# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent local compilation caches for managed vLLM subprocesses."""

import dataclasses
import hashlib
import json
import logging
import multiprocessing
import shutil
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr
from rigging.filesystem import StoragePath, marin_temp_bucket

from marin.inference.config import VllmCompilationCacheMode
from marin.profiling.trace_summary import sha256_for_path

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION = 4
LATEST_FILENAME = "latest.json"
GENERATIONS_DIR = "generations"
MANIFEST_FILENAME = "manifest.json"
ARCHIVE_FILENAME = "cache.zip"

_CACHE_TTL_DAYS = 30
_CACHE_PREFIX = "vllm-compilation-cache"
_TRANSFER_DEADLINE_SECONDS = 120
_TRANSFER_START_METHOD = "forkserver"
_MAX_METADATA_BYTES = 1 << 20
_MAX_ARCHIVE_BYTES = 20 << 30
_MAX_EXTRACTED_BYTES = 20 << 30
_MAX_ENTRIES = 200_000
_TRANSFER_PROCESS_GRACE_SECONDS = 5
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_TRANSIENT_SUFFIXES = (".lock", ".tmp")
_XLA_CACHE_SUBDIR = "xla"
_CHAT_TEMPLATE_ARG = "--chat-template"
_CHAT_TEMPLATE_DIGEST_ARG = "--chat-template-sha256"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_Digest = Annotated[str, Field(pattern=_DIGEST_PATTERN)]
_ArchiveSize = Annotated[int, Field(strict=True, ge=0, le=_MAX_ARCHIVE_BYTES)]
_ExtractedSize = Annotated[int, Field(strict=True, ge=0, le=_MAX_EXTRACTED_BYTES)]
_EntryCount = Annotated[int, Field(strict=True, ge=0, le=_MAX_ENTRIES)]


class _CacheMiss(Exception):
    pass


class _TransferStatus(StrEnum):
    OK = "ok"
    MISS = "miss"
    ERROR = "error"


class _CacheModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VllmCompileIdentity(_CacheModel):
    """Inputs that conservatively identify one vLLM compilation cache."""

    model_name_or_path: StrictStr
    extra_cli_args: tuple[StrictStr, ...]

    @classmethod
    def from_vllm_args(
        cls,
        *,
        model_name_or_path: str,
        extra_cli_args: tuple[str, ...],
    ) -> "VllmCompileIdentity":
        """Build an identity without run-specific chat-template file paths."""
        return cls(
            model_name_or_path=model_name_or_path,
            extra_cli_args=_stable_compile_cli_args(extra_cli_args),
        )


class _CacheIdentity(_CacheModel):
    schema_version: Literal[4] = ARCHIVE_SCHEMA_VERSION
    launcher: StrictStr
    compile: VllmCompileIdentity


class _Manifest(_CacheModel):
    schema_version: Literal[4] = ARCHIVE_SCHEMA_VERSION
    cache_identity: _CacheIdentity
    generation_id: _Digest
    archive_sha256: _Digest
    archive_size_bytes: _ArchiveSize
    source_size_bytes: _ExtractedSize
    entry_count: _EntryCount


class _LatestPointer(_CacheModel):
    generation_id: _Digest


class _TransferResult(_CacheModel):
    status: _TransferStatus
    reason: StrictStr | None = None
    generation_id: _Digest | None = None
    manifest: _Manifest | None = None


@dataclasses.dataclass(frozen=True, order=True)
class _FileEntry:
    relative_path: str
    size: int
    mode: int
    mtime_ns: int


@dataclasses.dataclass(frozen=True)
class _Archive:
    path: Path
    sha256: str
    archive_size: int
    source_size: int
    entry_count: int


@dataclasses.dataclass(frozen=True)
class _Extraction:
    entry_count: int
    size: int


@dataclasses.dataclass(frozen=True)
class _ManagedCache:
    local_temp_dir: Path
    local_cache_dir: Path
    remote_cache_dir: str
    cache_key: str
    identity: _CacheIdentity


_Model = TypeVar("_Model", bound=BaseModel)


def _stable_compile_cli_args(extra_cli_args: tuple[str, ...]) -> tuple[str, ...]:
    stable_args: list[str] = []
    index = 0
    while index < len(extra_cli_args):
        argument = extra_cli_args[index]
        if argument == _CHAT_TEMPLATE_ARG and index + 1 < len(extra_cli_args):
            stable_args.extend((_CHAT_TEMPLATE_DIGEST_ARG, _stable_chat_template(extra_cli_args[index + 1])))
            index += 2
            continue
        chat_template_prefix = f"{_CHAT_TEMPLATE_ARG}="
        if argument.startswith(chat_template_prefix):
            template = argument.removeprefix(chat_template_prefix)
            stable_args.append(f"{_CHAT_TEMPLATE_DIGEST_ARG}={_stable_chat_template(template)}")
        else:
            stable_args.append(argument)
        index += 1
    return tuple(stable_args)


def _stable_chat_template(template: str) -> str:
    path = Path(template)
    return sha256_for_path(path) if path.is_file() else template


def _canonical_model_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))


def _is_transient(path: Path) -> bool:
    return any(part.endswith(_TRANSIENT_SUFFIXES) for part in path.parts)


def _inventory(cache_dir: Path) -> tuple[_FileEntry, ...]:
    entries: list[_FileEntry] = []
    total_size = 0
    for path in sorted(cache_dir.rglob("*")):
        relative = path.relative_to(cache_dir)
        if _is_transient(relative):
            continue
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise ValueError(f"cache contains a symbolic link: {relative}")
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"cache contains an unsupported file: {relative}")
        entries.append(
            _FileEntry(
                relative_path=relative.as_posix(),
                size=path_stat.st_size,
                mode=stat.S_IMODE(path_stat.st_mode),
                mtime_ns=path_stat.st_mtime_ns,
            )
        )
        total_size += path_stat.st_size
        if len(entries) > _MAX_ENTRIES:
            raise ValueError(f"cache contains more than {_MAX_ENTRIES} files")
        if total_size > _MAX_EXTRACTED_BYTES:
            raise ValueError(f"cache contains more than {_MAX_EXTRACTED_BYTES} bytes")
    return tuple(entries)


def _current_entry(cache_dir: Path, entry: _FileEntry) -> _FileEntry:
    path = cache_dir / entry.relative_path
    path_stat = path.stat()
    return _FileEntry(
        relative_path=entry.relative_path,
        size=path_stat.st_size,
        mode=stat.S_IMODE(path_stat.st_mode),
        mtime_ns=path_stat.st_mtime_ns,
    )


def _create_archive(cache_dir: Path, destination: Path) -> _Archive | None:
    """Create a stable ZIP archive, or return ``None`` when the cache has no files."""
    before = _inventory(cache_dir)
    if not before:
        return None
    source_size = sum(entry.size for entry in before)
    archive_headroom = source_size + len(before) * 1024 + (1 << 20)
    if shutil.disk_usage(destination.parent).free < archive_headroom:
        raise ValueError(f"insufficient disk space to archive {source_size} cache bytes")

    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for entry in before:
            if _current_entry(cache_dir, entry) != entry:
                raise ValueError(f"cache file changed before archiving: {entry.relative_path}")
            path = cache_dir / entry.relative_path
            info = zipfile.ZipInfo(entry.relative_path, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | entry.mode) << 16
            with path.open("rb") as source, archive.open(info, mode="w", force_zip64=True) as target:
                shutil.copyfileobj(source, target)
            if _current_entry(cache_dir, entry) != entry:
                raise ValueError(f"cache file changed while archiving: {entry.relative_path}")

    if _inventory(cache_dir) != before:
        raise ValueError("cache inventory changed while archiving")
    archive_size = destination.stat().st_size
    if archive_size > _MAX_ARCHIVE_BYTES:
        raise ValueError(f"cache archive exceeds {_MAX_ARCHIVE_BYTES} bytes")
    return _Archive(
        path=destination,
        sha256=sha256_for_path(destination),
        archive_size=archive_size,
        source_size=source_size,
        entry_count=len(before),
    )


def _extract_archive(archive_path: Path, destination: Path) -> _Extraction:
    entry_count = 0
    extracted_size = 0
    names: set[str] = set()
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
                raise _CacheMiss(f"unsafe archive path: {member.filename}")
            if member.filename in names:
                raise _CacheMiss(f"duplicate archive path: {member.filename}")
            names.add(member.filename)
            mode = member.external_attr >> 16
            if member.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise _CacheMiss(f"unsupported archive entry: {member.filename}")
            entry_count += 1
            if entry_count > _MAX_ENTRIES:
                raise _CacheMiss(f"archive contains more than {_MAX_ENTRIES} entries")
            extracted_size += member.file_size
            if extracted_size > _MAX_EXTRACTED_BYTES:
                raise _CacheMiss(f"archive expands past {_MAX_EXTRACTED_BYTES} bytes")

            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, mode="r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(stat.S_IMODE(mode) & 0o777)
    return _Extraction(entry_count=entry_count, size=extracted_size)


def _remove_local_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove local vLLM compilation cache transfer file %s", path, exc_info=True)


def _read_metadata(path: StoragePath, model_type: type[_Model]) -> _Model:
    if path.size() > _MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds {_MAX_METADATA_BYTES} bytes")
    return model_type.model_validate_json(path.read_text())


def _write_result(path: str, result: _TransferResult) -> None:
    Path(path).write_text(result.model_dump_json(exclude_none=True))


def _restore_transfer_worker(remote_cache_dir: str, archive_path: str, result_path: str) -> None:
    try:
        remote = StoragePath(remote_cache_dir)
        latest_path = remote / LATEST_FILENAME
        if not latest_path.exists():
            _write_result(
                result_path,
                _TransferResult(status=_TransferStatus.MISS, reason="latest pointer not found"),
            )
            return
        latest = _read_metadata(latest_path, _LatestPointer)
        generation = remote / f"{GENERATIONS_DIR}/{latest.generation_id}"
        manifest_path = generation / MANIFEST_FILENAME
        remote_archive = generation / ARCHIVE_FILENAME
        if not manifest_path.exists() or not remote_archive.exists():
            _write_result(
                result_path,
                _TransferResult(status=_TransferStatus.MISS, reason="latest generation is incomplete"),
            )
            return
        manifest = _read_metadata(manifest_path, _Manifest)
        if remote_archive.size() != manifest.archive_size_bytes:
            raise ValueError("remote archive size does not match its manifest")
        remote_archive.download_to(archive_path)
        _write_result(
            result_path,
            _TransferResult(
                status=_TransferStatus.OK,
                generation_id=latest.generation_id,
                manifest=manifest,
            ),
        )
    except Exception as error:
        _write_result(result_path, _TransferResult(status=_TransferStatus.ERROR, reason=str(error)))


def _publish_transfer_worker(
    remote_cache_dir: str,
    generation_id: str,
    archive_path: str,
    manifest_json: str,
    result_path: str,
) -> None:
    try:
        remote = StoragePath(remote_cache_dir)
        generation = remote / f"{GENERATIONS_DIR}/{generation_id}"
        generation.mkdirs()
        (generation / ARCHIVE_FILENAME).upload_from(archive_path)
        (generation / MANIFEST_FILENAME).write_text(manifest_json)
        remote.mkdirs()
        latest = _LatestPointer(generation_id=generation_id)
        (remote / LATEST_FILENAME).write_text(latest.model_dump_json())
        _write_result(result_path, _TransferResult(status=_TransferStatus.OK))
    except Exception as error:
        _write_result(result_path, _TransferResult(status=_TransferStatus.ERROR, reason=str(error)))


def _read_transfer_result(path: Path) -> _TransferResult:
    return _TransferResult.model_validate_json(path.read_text())


def _run_transfer(
    target: Callable[..., None],
    args: tuple[Any, ...],
    result_path: Path,
    remote_cache_dir: str,
    *,
    timeout_seconds: int = _TRANSFER_DEADLINE_SECONDS,
) -> _TransferResult:
    """Return the transfer outcome, reporting remote deadline expiry as an error."""
    result_path.unlink(missing_ok=True)
    if StoragePath(remote_cache_dir).is_local:
        target(*args, str(result_path))
        return _read_transfer_result(result_path)

    context = multiprocessing.get_context(_TRANSFER_START_METHOD)
    process = context.Process(target=target, args=(*args, str(result_path)))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(_TRANSFER_PROCESS_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(_TRANSFER_PROCESS_GRACE_SECONDS)
        process.close()
        return _TransferResult(
            status=_TransferStatus.ERROR,
            reason=f"transfer exceeded {timeout_seconds}s deadline",
        )
    exit_code = process.exitcode
    process.close()
    if not result_path.exists():
        return _TransferResult(
            status=_TransferStatus.ERROR,
            reason=f"transfer helper exited with code {exit_code}",
        )
    return _read_transfer_result(result_path)


def _validate_manifest(
    manifest: _Manifest,
    *,
    expected_identity: _CacheIdentity,
    expected_generation_id: str,
    archive_path: Path,
) -> None:
    if manifest.cache_identity != expected_identity:
        raise _CacheMiss("cache identity mismatch")
    if manifest.generation_id != expected_generation_id:
        raise _CacheMiss("generation manifest does not match the latest pointer")
    if manifest.generation_id != manifest.archive_sha256:
        raise _CacheMiss("generation id does not match the archive digest")
    actual_size = archive_path.stat().st_size
    if actual_size != manifest.archive_size_bytes:
        raise _CacheMiss(f"archive size mismatch: expected {manifest.archive_size_bytes}, got {actual_size}")
    actual_sha256 = sha256_for_path(archive_path)
    if actual_sha256 != manifest.archive_sha256:
        raise _CacheMiss(f"archive digest mismatch: expected {manifest.archive_sha256}, got {actual_sha256}")


class VllmCompilationCache:
    """Compilation-cache lifecycle owned by a vLLM server handle."""

    def __init__(
        self,
        *,
        environment: dict[str, str],
        managed_cache: _ManagedCache | None = None,
    ) -> None:
        self._environment = environment
        self._managed_cache = managed_cache
        self._restored_sha256: str | None = None
        self._closed = False
        self._published = False

    @classmethod
    def prepare(
        cls,
        *,
        launcher_identity: str,
        compile_identity: VllmCompileIdentity,
        environment: dict[str, str],
        mode: VllmCompilationCacheMode = VllmCompilationCacheMode.MANAGED,
    ) -> "VllmCompilationCache":
        """Create and restore a managed cache, or preserve the caller environment when disabled."""
        child_environment = dict(environment)
        if mode is VllmCompilationCacheMode.DISABLED:
            return cls(environment=child_environment)

        local_temp_dir = Path(tempfile.mkdtemp(prefix="vllm_compilation_cache_"))
        local_cache_dir = local_temp_dir / "cache"
        local_cache_dir.mkdir()
        identity = _CacheIdentity(launcher=launcher_identity, compile=compile_identity)
        cache_key = hashlib.sha256(_canonical_model_json(identity).encode()).hexdigest()
        remote_cache_dir = str(
            StoragePath(marin_temp_bucket(_CACHE_TTL_DAYS, _CACHE_PREFIX)) / f"v{ARCHIVE_SCHEMA_VERSION}/{cache_key}"
        )

        child_environment.update(
            {
                "JAX_COMPILATION_CACHE_DIR": str(local_cache_dir / _XLA_CACHE_SUBDIR),
                "VLLM_XLA_CACHE_PATH": str(local_cache_dir / _XLA_CACHE_SUBDIR),
                "JAX_ENABLE_COMPILATION_CACHE": "1",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "-1",
                "VLLM_CACHE_ROOT": str(local_cache_dir / "vllm"),
                "TORCHINDUCTOR_CACHE_DIR": str(local_cache_dir / "torchinductor"),
                "TRITON_CACHE_DIR": str(local_cache_dir / "triton"),
                "CUDA_CACHE_PATH": str(local_cache_dir / "cuda"),
                "VLLM_COMPILE_CACHE_SAVE_FORMAT": "binary",
            }
        )
        cache = cls(
            environment=child_environment,
            managed_cache=_ManagedCache(
                local_temp_dir=local_temp_dir,
                local_cache_dir=local_cache_dir,
                remote_cache_dir=remote_cache_dir,
                cache_key=cache_key,
                identity=identity,
            ),
        )
        cache._restore()
        return cache

    def environment(self) -> dict[str, str]:
        """Return a copy of the environment for the managed vLLM child."""
        return dict(self._environment)

    def publish(self) -> None:
        """Publish a stable cache generation without failing an otherwise ready server."""
        managed = self._managed_cache
        if managed is None or self._published:
            return

        self._published = True
        started = time.monotonic()
        archive_path = managed.local_temp_dir / ARCHIVE_FILENAME
        try:
            archive = _create_archive(managed.local_cache_dir, archive_path)
            if archive is None:
                logger.info("Skipping empty vLLM compilation cache publication key=%s", managed.cache_key)
                return
            if archive.sha256 == self._restored_sha256:
                logger.info(
                    "Skipping unchanged vLLM compilation cache publication key=%s entries=%d bytes=%d",
                    managed.cache_key,
                    archive.entry_count,
                    archive.source_size,
                )
                return

            generation_id = archive.sha256
            manifest = _Manifest(
                cache_identity=managed.identity,
                generation_id=generation_id,
                archive_sha256=archive.sha256,
                archive_size_bytes=archive.archive_size,
                source_size_bytes=archive.source_size,
                entry_count=archive.entry_count,
            )
            result = _run_transfer(
                _publish_transfer_worker,
                (
                    managed.remote_cache_dir,
                    generation_id,
                    str(archive.path),
                    manifest.model_dump_json(),
                ),
                managed.local_temp_dir / "publish-result.json",
                managed.remote_cache_dir,
            )
            if result.status is not _TransferStatus.OK:
                logger.warning(
                    "Skipping vLLM compilation cache publication key=%s reason=%s",
                    managed.cache_key,
                    result.reason or "unknown transfer failure",
                )
                return
            self._restored_sha256 = archive.sha256
            logger.info(
                "Published vLLM compilation cache key=%s entries=%d source_bytes=%d " "archive_bytes=%d duration=%.1fs",
                managed.cache_key,
                archive.entry_count,
                archive.source_size,
                archive.archive_size,
                time.monotonic() - started,
            )
        except Exception as error:
            logger.warning(
                "Skipping vLLM compilation cache publication key=%s reason=%s",
                managed.cache_key,
                error,
                exc_info=True,
            )
        finally:
            _remove_local_file(archive_path)

    def close(self) -> None:
        """Remove the manager-owned local directory after the vLLM process group exits."""
        managed = self._managed_cache
        if managed is None or self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(managed.local_temp_dir)
        except OSError:
            logger.warning(
                "Failed to remove local vLLM compilation cache %s",
                managed.local_temp_dir,
                exc_info=True,
            )

    def _restore(self) -> None:
        managed = self._managed_cache
        if managed is None:
            return
        started = time.monotonic()
        archive_path = managed.local_temp_dir / ARCHIVE_FILENAME
        restored_cache_dir = managed.local_temp_dir / "restored-cache"
        try:
            result = _run_transfer(
                _restore_transfer_worker,
                (managed.remote_cache_dir, str(archive_path)),
                managed.local_temp_dir / "restore-result.json",
                managed.remote_cache_dir,
            )
            if result.status is _TransferStatus.MISS:
                logger.info(
                    "vLLM compilation cache miss key=%s reason=%s",
                    managed.cache_key,
                    result.reason or "no published generation",
                )
                return
            if result.status is not _TransferStatus.OK:
                raise _CacheMiss(result.reason or "unknown transfer failure")

            manifest = result.manifest
            if manifest is None or result.generation_id is None:
                raise _CacheMiss("generation manifest is not an object")
            _validate_manifest(
                manifest,
                expected_identity=managed.identity,
                expected_generation_id=result.generation_id,
                archive_path=archive_path,
            )

            required_space = archive_path.stat().st_size + manifest.source_size_bytes
            if shutil.disk_usage(managed.local_temp_dir).free < required_space:
                raise _CacheMiss(f"insufficient local disk for {required_space} cache bytes")

            restored_cache_dir.mkdir()
            extraction = _extract_archive(archive_path, restored_cache_dir)
            if extraction.entry_count != manifest.entry_count:
                raise _CacheMiss(
                    f"archive entry count mismatch: expected {manifest.entry_count}, got {extraction.entry_count}"
                )
            if extraction.size != manifest.source_size_bytes:
                raise _CacheMiss(
                    f"extracted cache size mismatch: expected {manifest.source_size_bytes}, got {extraction.size}"
                )
            managed.local_cache_dir.rmdir()
            restored_cache_dir.replace(managed.local_cache_dir)
            self._restored_sha256 = manifest.archive_sha256
            logger.info(
                "Restored vLLM compilation cache key=%s entries=%s source_bytes=%s " "archive_bytes=%s duration=%.1fs",
                managed.cache_key,
                manifest.entry_count,
                manifest.source_size_bytes,
                manifest.archive_size_bytes,
                time.monotonic() - started,
            )
        except Exception as error:
            shutil.rmtree(restored_cache_dir, ignore_errors=True)
            managed.local_cache_dir.mkdir(exist_ok=True)
            logger.warning(
                "vLLM compilation cache miss key=%s reason=%s",
                managed.cache_key,
                error,
            )
        finally:
            _remove_local_file(archive_path)
