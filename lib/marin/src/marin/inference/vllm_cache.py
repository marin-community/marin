# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent local compilation caches for managed vLLM subprocesses."""

import dataclasses
import gzip
import hashlib
import json
import logging
import multiprocessing
import re
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from rigging.filesystem import StoragePath, marin_temp_bucket

from marin.inference.config import VllmCompilationCacheMode

logger = logging.getLogger(__name__)

_ARCHIVE_SCHEMA_VERSION = 3
_CACHE_TTL_DAYS = 30
_CACHE_PREFIX = "vllm-compilation-cache"
_TRANSFER_DEADLINE_SECONDS = 120
_TRANSFER_START_METHOD = "forkserver"
_MAX_METADATA_BYTES = 1 << 20
_MAX_ARCHIVE_BYTES = 20 << 30
_MAX_EXTRACTED_BYTES = 20 << 30
_MAX_ENTRIES = 200_000
_GENERATION_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_SUFFIXES = (".lock", ".tmp")


class _CacheMiss(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class VllmCompileIdentity:
    """Inputs that conservatively identify one vLLM compilation namespace."""

    model_name_or_path: str
    extra_cli_args: tuple[str, ...]


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
    size: int
    source_size: int
    entry_count: int


class VllmCompilationCache:
    """Own a local compiler cache and synchronize it with Marin temporary storage."""

    def __init__(
        self,
        *,
        mode: VllmCompilationCacheMode,
        environment: dict[str, str],
        work_dir: Path | None,
        root: Path | None,
        remote_prefix: str | None,
        namespace: str | None,
        namespace_identity: dict[str, Any] | None,
    ) -> None:
        self._mode = mode
        self._environment = environment
        self._work_dir = work_dir
        self._root = root
        self._remote_prefix = remote_prefix
        self._namespace = namespace
        self._namespace_identity = namespace_identity
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
        """Create a local cache, restore its latest generation, and build the child environment."""
        child_environment = dict(environment)
        if mode is VllmCompilationCacheMode.DISABLED:
            return cls(
                mode=mode,
                environment=child_environment,
                work_dir=None,
                root=None,
                remote_prefix=None,
                namespace=None,
                namespace_identity=None,
            )

        work_dir = Path(tempfile.mkdtemp(prefix="vllm_compilation_cache_"))
        root = work_dir / "cache"
        root.mkdir()
        namespace_identity = json.loads(
            _canonical_json(
                {
                    "schema_version": _ARCHIVE_SCHEMA_VERSION,
                    "launcher": launcher_identity,
                    "compile": dataclasses.asdict(compile_identity),
                }
            )
        )
        namespace = hashlib.sha256(_canonical_json(namespace_identity).encode()).hexdigest()

        remote_prefix: str | None
        try:
            remote_prefix = str(
                StoragePath(marin_temp_bucket(_CACHE_TTL_DAYS, _CACHE_PREFIX))
                / f"v{_ARCHIVE_SCHEMA_VERSION}/{namespace}"
            )
        except Exception:
            remote_prefix = None
            logger.warning(
                "vLLM compilation cache has no remote prefix; using an ephemeral local cache",
                exc_info=True,
            )

        child_environment.update(
            {
                "JAX_COMPILATION_CACHE_DIR": str(root / "xla"),
                "VLLM_XLA_CACHE_PATH": str(root / "xla"),
                "JAX_ENABLE_COMPILATION_CACHE": "1",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "-1",
                "VLLM_CACHE_ROOT": str(root / "vllm"),
                "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
                "TRITON_CACHE_DIR": str(root / "triton"),
                "CUDA_CACHE_PATH": str(root / "cuda"),
                "VLLM_COMPILE_CACHE_SAVE_FORMAT": "binary",
            }
        )
        cache = cls(
            mode=mode,
            environment=child_environment,
            work_dir=work_dir,
            root=root,
            remote_prefix=remote_prefix,
            namespace=namespace,
            namespace_identity=namespace_identity,
        )
        cache._restore()
        return cache

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Disabled vLLM compilation caches do not have a local root")
        return self._root

    @property
    def remote_prefix(self) -> str:
        if self._remote_prefix is None:
            raise RuntimeError("This vLLM compilation cache does not have a remote prefix")
        return self._remote_prefix

    @property
    def namespace(self) -> str:
        if self._namespace is None:
            raise RuntimeError("Disabled vLLM compilation caches do not have a namespace")
        return self._namespace

    def environment(self) -> dict[str, str]:
        """Return a copy of the environment for the managed vLLM child."""
        return dict(self._environment)

    def publish_after_ready(self) -> None:
        """Publish a stable cache generation without failing an otherwise ready server."""
        if (
            self._mode is VllmCompilationCacheMode.DISABLED
            or self._published
            or self._remote_prefix is None
            or self._work_dir is None
            or self._root is None
            or self._namespace is None
            or self._namespace_identity is None
        ):
            return

        self._published = True
        started = time.monotonic()
        archive_path = self._work_dir / "publish.tar.gz"
        try:
            archive = _create_archive(self._root, archive_path)
            if archive is None:
                logger.info("Skipping empty vLLM compilation cache publication namespace=%s", self._namespace)
                return
            if archive.sha256 == self._restored_sha256:
                logger.info(
                    "Skipping unchanged vLLM compilation cache publication namespace=%s entries=%d bytes=%d",
                    self._namespace,
                    archive.entry_count,
                    archive.source_size,
                )
                return

            generation_id = archive.sha256
            manifest = {
                "schema_version": _ARCHIVE_SCHEMA_VERSION,
                "namespace": self._namespace,
                "namespace_identity": self._namespace_identity,
                "generation_id": generation_id,
                "archive_sha256": archive.sha256,
                "archive_size_bytes": archive.size,
                "source_size_bytes": archive.source_size,
                "entry_count": archive.entry_count,
            }
            result = _run_transfer(
                _publish_transfer_worker,
                (
                    self._remote_prefix,
                    generation_id,
                    str(archive.path),
                    _canonical_json(manifest),
                ),
                self._work_dir / "publish-result.json",
                use_subprocess=StoragePath(self._remote_prefix).is_remote,
            )
            if result.get("status") != "ok":
                logger.warning(
                    "Skipping vLLM compilation cache publication namespace=%s reason=%s",
                    self._namespace,
                    result.get("reason", "unknown transfer failure"),
                )
                return
            self._restored_sha256 = archive.sha256
            logger.info(
                "Published vLLM compilation cache namespace=%s entries=%d source_bytes=%d "
                "archive_bytes=%d duration=%.1fs",
                self._namespace,
                archive.entry_count,
                archive.source_size,
                archive.size,
                time.monotonic() - started,
            )
        except Exception as error:
            logger.warning(
                "Skipping vLLM compilation cache publication namespace=%s reason=%s",
                self._namespace,
                error,
                exc_info=True,
            )
        finally:
            _remove_local_file(archive_path)

    def close(self) -> None:
        """Remove the manager-owned local root after the vLLM process group exits."""
        if self._closed:
            return
        self._closed = True
        if self._work_dir is not None:
            try:
                shutil.rmtree(self._work_dir)
            except OSError:
                logger.warning("Failed to remove local vLLM compilation cache %s", self._work_dir, exc_info=True)

    def _restore(self) -> None:
        if self._remote_prefix is None or self._work_dir is None or self._root is None or self._namespace is None:
            return

        started = time.monotonic()
        archive_path = self._work_dir / "restore.tar.gz"
        staging = self._work_dir / "restored"
        try:
            result = _run_transfer(
                _restore_transfer_worker,
                (self._remote_prefix, str(archive_path)),
                self._work_dir / "restore-result.json",
                use_subprocess=StoragePath(self._remote_prefix).is_remote,
            )
            status = result.get("status")
            if status == "miss":
                logger.info(
                    "vLLM compilation cache miss namespace=%s reason=%s",
                    self._namespace,
                    result.get("reason", "no published generation"),
                )
                return
            if status != "ok":
                raise _CacheMiss(str(result.get("reason", "unknown transfer failure")))

            manifest = result.get("manifest")
            if not isinstance(manifest, dict):
                raise _CacheMiss("generation manifest is not an object")
            self._validate_manifest(manifest, str(result.get("generation_id")), archive_path)

            source_size = int(manifest.get("source_size_bytes", 0))
            required_space = archive_path.stat().st_size + source_size
            if shutil.disk_usage(self._work_dir).free < required_space:
                raise _CacheMiss(f"insufficient local disk for {required_space} cache bytes")

            staging.mkdir()
            entry_count, extracted_size = _extract_archive(archive_path, staging)
            if entry_count != manifest["entry_count"]:
                raise _CacheMiss(f"archive entry count mismatch: expected {manifest['entry_count']}, got {entry_count}")
            if extracted_size != manifest["source_size_bytes"]:
                raise _CacheMiss(
                    f"extracted cache size mismatch: expected {manifest['source_size_bytes']}, got {extracted_size}"
                )
            self._root.rmdir()
            staging.replace(self._root)
            self._restored_sha256 = str(manifest["archive_sha256"])
            logger.info(
                "Restored vLLM compilation cache namespace=%s entries=%s source_bytes=%s "
                "archive_bytes=%s duration=%.1fs",
                self._namespace,
                manifest.get("entry_count", "unknown"),
                manifest.get("source_size_bytes", "unknown"),
                manifest["archive_size_bytes"],
                time.monotonic() - started,
            )
        except Exception as error:
            shutil.rmtree(staging, ignore_errors=True)
            try:
                self._root.mkdir(exist_ok=True)
            except OSError:
                logger.warning("Failed to recreate local vLLM compilation cache root %s", self._root, exc_info=True)
            logger.warning(
                "vLLM compilation cache miss namespace=%s reason=%s",
                self._namespace,
                error,
            )
        finally:
            _remove_local_file(archive_path)

    def _validate_manifest(self, manifest: dict[str, Any], expected_generation_id: str, archive_path: Path) -> None:
        if manifest.get("schema_version") != _ARCHIVE_SCHEMA_VERSION:
            raise _CacheMiss("unsupported archive schema")
        if manifest.get("namespace") != self._namespace:
            raise _CacheMiss("namespace mismatch")
        if manifest.get("namespace_identity") != self._namespace_identity:
            raise _CacheMiss("namespace identity mismatch")
        generation_id = manifest.get("generation_id")
        if not isinstance(generation_id, str) or _GENERATION_PATTERN.fullmatch(generation_id) is None:
            raise _CacheMiss("invalid generation id")
        if generation_id != expected_generation_id:
            raise _CacheMiss("generation manifest does not match the latest pointer")

        expected_size = manifest.get("archive_size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0 or expected_size > _MAX_ARCHIVE_BYTES:
            raise _CacheMiss("invalid archive size")
        actual_size = archive_path.stat().st_size
        if actual_size != expected_size:
            raise _CacheMiss(f"archive size mismatch: expected {expected_size}, got {actual_size}")

        expected_sha256 = manifest.get("archive_sha256")
        if not isinstance(expected_sha256, str) or _GENERATION_PATTERN.fullmatch(expected_sha256) is None:
            raise _CacheMiss("invalid archive digest")
        if generation_id != expected_sha256:
            raise _CacheMiss("generation id does not match the archive digest")
        actual_sha256 = _file_sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise _CacheMiss(f"archive digest mismatch: expected {expected_sha256}, got {actual_sha256}")

        source_size = manifest.get("source_size_bytes")
        if not isinstance(source_size, int) or source_size < 0 or source_size > _MAX_EXTRACTED_BYTES:
            raise _CacheMiss("invalid extracted cache size")
        entry_count = manifest.get("entry_count")
        if not isinstance(entry_count, int) or entry_count < 0 or entry_count > _MAX_ENTRIES:
            raise _CacheMiss("invalid cache entry count")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_transient(path: Path) -> bool:
    return any(part.endswith(_TRANSIENT_SUFFIXES) for part in path.parts)


def _inventory(root: Path) -> tuple[_FileEntry, ...]:
    entries: list[_FileEntry] = []
    total_size = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
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


def _current_entry(root: Path, entry: _FileEntry) -> _FileEntry:
    path = root / entry.relative_path
    path_stat = path.stat()
    return _FileEntry(
        relative_path=entry.relative_path,
        size=path_stat.st_size,
        mode=stat.S_IMODE(path_stat.st_mode),
        mtime_ns=path_stat.st_mtime_ns,
    )


def _create_archive(root: Path, destination: Path) -> _Archive | None:
    before = _inventory(root)
    if not before:
        return None
    source_size = sum(entry.size for entry in before)
    archive_headroom = source_size + len(before) * 1024 + (1 << 20)
    if shutil.disk_usage(destination.parent).free < archive_headroom:
        raise ValueError(f"insufficient disk space to archive {source_size} cache bytes")

    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for entry in before:
                    if _current_entry(root, entry) != entry:
                        raise ValueError(f"cache file changed before archiving: {entry.relative_path}")
                    path = root / entry.relative_path
                    info = tarfile.TarInfo(entry.relative_path)
                    info.size = entry.size
                    info.mode = entry.mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                    if _current_entry(root, entry) != entry:
                        raise ValueError(f"cache file changed while archiving: {entry.relative_path}")

    if _inventory(root) != before:
        raise ValueError("cache inventory changed while archiving")
    archive_size = destination.stat().st_size
    if archive_size > _MAX_ARCHIVE_BYTES:
        raise ValueError(f"cache archive exceeds {_MAX_ARCHIVE_BYTES} bytes")
    return _Archive(
        path=destination,
        sha256=_file_sha256(destination),
        size=archive_size,
        source_size=source_size,
        entry_count=len(before),
    )


def _extract_archive(archive_path: Path, destination: Path) -> tuple[int, int]:
    entry_count = 0
    extracted_size = 0
    names: set[str] = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
                raise _CacheMiss(f"unsafe archive path: {member.name}")
            if member.name in names:
                raise _CacheMiss(f"duplicate archive path: {member.name}")
            names.add(member.name)
            if not (member.isfile() or member.isdir()):
                raise _CacheMiss(f"unsupported archive entry: {member.name}")
            entry_count += 1
            if entry_count > _MAX_ENTRIES:
                raise _CacheMiss(f"archive contains more than {_MAX_ENTRIES} entries")
            extracted_size += member.size
            if extracted_size > _MAX_EXTRACTED_BYTES:
                raise _CacheMiss(f"archive expands past {_MAX_EXTRACTED_BYTES} bytes")
            archive.extract(member, path=destination, filter="data")
    return entry_count, extracted_size


def _remove_local_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove local vLLM compilation cache transfer file %s", path, exc_info=True)


def _read_small_json(path: StoragePath) -> dict[str, Any]:
    if path.size() > _MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds {_MAX_METADATA_BYTES} bytes")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("metadata is not an object")
    return value


def _restore_transfer_worker(remote_prefix: str, archive_path: str, result_path: str) -> None:
    try:
        remote = StoragePath(remote_prefix)
        latest_path = remote / "latest.json"
        if not latest_path.exists():
            _write_result(result_path, {"status": "miss", "reason": "latest pointer not found"})
            return
        latest = _read_small_json(latest_path)
        generation_id = latest.get("generation_id")
        if not isinstance(generation_id, str) or _GENERATION_PATTERN.fullmatch(generation_id) is None:
            raise ValueError("latest pointer has an invalid generation id")
        generation = remote / f"generations/{generation_id}"
        manifest_path = generation / "manifest.json"
        archive_remote = generation / "cache.tar.gz"
        if not manifest_path.exists() or not archive_remote.exists():
            _write_result(result_path, {"status": "miss", "reason": "latest generation is incomplete"})
            return
        manifest = _read_small_json(manifest_path)
        archive_size = manifest.get("archive_size_bytes")
        if not isinstance(archive_size, int) or archive_size < 0 or archive_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("manifest has an invalid archive size")
        if archive_remote.size() != archive_size:
            raise ValueError("remote archive size does not match its manifest")
        archive_remote.download_to(archive_path)
        _write_result(
            result_path,
            {"status": "ok", "generation_id": generation_id, "manifest": manifest},
        )
    except Exception as error:
        _write_result(result_path, {"status": "error", "reason": str(error)})


def _publish_transfer_worker(
    remote_prefix: str,
    generation_id: str,
    archive_path: str,
    manifest_json: str,
    result_path: str,
) -> None:
    try:
        remote = StoragePath(remote_prefix)
        generation = remote / f"generations/{generation_id}"
        generation.mkdirs()
        (generation / "cache.tar.gz").upload_from(archive_path)
        (generation / "manifest.json").write_text(manifest_json)
        remote.mkdirs()
        (remote / "latest.json").write_text(_canonical_json({"generation_id": generation_id}))
        _write_result(result_path, {"status": "ok"})
    except Exception as error:
        _write_result(result_path, {"status": "error", "reason": str(error)})


def _write_result(path: str, value: dict[str, Any]) -> None:
    Path(path).write_text(_canonical_json(value))


def _run_transfer(
    target: Callable[..., None],
    args: tuple[Any, ...],
    result_path: Path,
    *,
    timeout_seconds: int = _TRANSFER_DEADLINE_SECONDS,
    use_subprocess: bool = True,
) -> dict[str, Any]:
    result_path.unlink(missing_ok=True)
    if not use_subprocess:
        target(*args, str(result_path))
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            return {"status": "error", "reason": "transfer helper returned an invalid result"}
        return result

    context = multiprocessing.get_context(_TRANSFER_START_METHOD)
    process = context.Process(target=target, args=(*args, str(result_path)))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()
        return {"status": "error", "reason": f"transfer exceeded {timeout_seconds}s deadline"}
    exit_code = process.exitcode
    process.close()
    if not result_path.exists():
        return {"status": "error", "reason": f"transfer helper exited with code {exit_code}"}
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        return {"status": "error", "reason": "transfer helper returned an invalid result"}
    return result
