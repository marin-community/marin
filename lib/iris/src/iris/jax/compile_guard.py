# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check distributed GPU compile inputs and executables before execution."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Protocol

logger = logging.getLogger(__name__)

# Large checkpoint loads can leave healthy ranks minutes apart at their first compile.
_COMPILE_FINGERPRINT_TIMEOUT_MS = 30 * 60_000
_COMPILE_FINGERPRINT_PREFIX = "iris/compile-fingerprint/v1"
_EXECUTABLE_TEXT_PREFIX = "iris/compile-executable-hash/v1"
_AUTOTUNE_COUNT_PREFIX = "iris/compile-autotune-count/v1"
_EXECUTABLE_TEXT_FAILURE = "extraction_failed"
# Warn once per process even if JAX installs a new guard after distributed reset.
_unsupported_text_warning_lock = threading.Lock()
_unsupported_text_warned = False


class _Coordinator(Protocol):
    def key_value_set(self, key: str, value: str, allow_overwrite: bool = False) -> None: ...

    def key_value_dir_get(self, key: str) -> list[tuple[str, str]]: ...

    def key_value_delete(self, key: str) -> None: ...

    def wait_at_barrier(self, barrier_id: str, timeout_in_ms: int, process_ids: Sequence[int]) -> None: ...


def check_distributed_compile_fingerprint(
    client: _Coordinator,
    process_ids: tuple[int, ...],
    process_id: int,
    sequence: int,
    fingerprint: str,
    timeout_ms: int = _COMPILE_FINGERPRINT_TIMEOUT_MS,
) -> None:
    """Fail before native GPU compilation if participating ranks lowered different modules."""
    group = "-".join(map(str, process_ids))
    prefix = f"{_COMPILE_FINGERPRINT_PREFIX}/{group}/{sequence}/"
    client.key_value_set(f"{prefix}{process_id}", fingerprint)

    try:
        client.wait_at_barrier(f"{prefix}ready", timeout_ms, process_ids)
    except Exception as exc:
        observed = _observed_fingerprints(client, prefix)
        raise RuntimeError(
            f"GPU compile fingerprint rendezvous failed on process {process_id}; "
            f"expected processes {process_ids}; observed {observed}"
        ) from exc

    observed = _observed_fingerprints(client, prefix)
    if set(observed) != set(process_ids) or len(set(observed.values())) != 1:
        raise RuntimeError(f"GPU compile fingerprint mismatch on process {process_id}: {observed}")

    client.wait_at_barrier(f"{prefix}read", timeout_ms, process_ids)
    if process_id == process_ids[0]:
        for peer in process_ids:
            client.key_value_delete(f"{prefix}{peer}")


def _observed_fingerprints(client: _Coordinator, prefix: str) -> dict[int, str]:
    return {int(key.removeprefix(prefix)): value for key, value in client.key_value_dir_get(prefix)}


@dataclass(frozen=True)
class _ExecutableRecord:
    text_sha256: str | None
    autotune_dir: str


def check_distributed_executable_text(
    client: _Coordinator,
    process_ids: tuple[int, ...],
    process_id: int,
    sequence: int,
    cache_key: str,
    executable_text: str | None,
    autotune_dir: str,
    extraction_error: Exception | None = None,
    timeout_ms: int = _COMPILE_FINGERPRINT_TIMEOUT_MS,
) -> None:
    """Fail before execution if participating ranks compiled different optimized HLO."""
    group = "-".join(map(str, process_ids))
    prefix = f"{_EXECUTABLE_TEXT_PREFIX}/{group}/{sequence}/"
    text_sha256 = hashlib.sha256(executable_text.encode("utf-8")).hexdigest() if executable_text is not None else None
    if extraction_error is not None:
        text_sha256 = _EXECUTABLE_TEXT_FAILURE
    record = _ExecutableRecord(text_sha256, autotune_dir)
    client.key_value_set(f"{prefix}{process_id}", json.dumps(asdict(record)))

    try:
        client.wait_at_barrier(f"{prefix}ready", timeout_ms, process_ids)
    except Exception as exc:
        observed = _observed_executables(client, prefix)
        raise RuntimeError(
            f"GPU executable text rendezvous failed on process {process_id}; "
            f"cache key {cache_key}; expected processes {process_ids}; observed {observed}"
        ) from exc

    observed = _observed_executables(client, prefix)
    if set(observed) != set(process_ids):
        raise RuntimeError(
            f"GPU executable text rendezvous incomplete on process {process_id}; "
            f"cache key {cache_key}; expected processes {process_ids}; observed {observed}"
        )

    failed = [
        rank for rank, peer_record in sorted(observed.items()) if peer_record.text_sha256 == _EXECUTABLE_TEXT_FAILURE
    ]
    if failed:
        raise RuntimeError(
            f"GPU optimized HLO text extraction failed on processes {failed}; cache key {cache_key}"
        ) from extraction_error

    hashes: dict[str, list[int]] = defaultdict(list)
    for rank, peer_record in sorted(observed.items()):
        if peer_record.text_sha256 is not None:
            hashes[peer_record.text_sha256].append(rank)

    if any(peer_record.text_sha256 is None for peer_record in observed.values()):
        _warn_unsupported_text()
    elif len(hashes) > 1:
        count_prefix = f"{_AUTOTUNE_COUNT_PREFIX}/{group}/{sequence}/"
        client.key_value_set(f"{count_prefix}{process_id}", json.dumps(_autotune_cache_file_count(autotune_dir)))
        client.wait_at_barrier(f"{count_prefix}ready", timeout_ms, process_ids)
        counts = {
            int(key.removeprefix(count_prefix)): json.loads(value)
            for key, value in client.key_value_dir_get(count_prefix)
        }
        groups = ", ".join(f"{digest}: {ranks}" for digest, ranks in sorted(hashes.items()))
        metadata_parts = []
        for rank, peer_record in sorted(observed.items()):
            file_count = "unavailable" if counts[rank] is None else str(counts[rank])
            metadata_parts.append(f"{rank}: {peer_record.autotune_dir!r} file_count={file_count}")
        metadata = ", ".join(metadata_parts)
        raise RuntimeError(
            f"GPU executable text mismatch on process {process_id}; cache key {cache_key}; "
            f"{len(hashes)} distinct executable_text_sha256 values: {groups}; autotune inputs: {metadata}"
        )

    client.wait_at_barrier(f"{prefix}read", timeout_ms, process_ids)
    if process_id == process_ids[0]:
        for peer in process_ids:
            client.key_value_delete(f"{prefix}{peer}")


def _observed_executables(client: _Coordinator, prefix: str) -> dict[int, _ExecutableRecord]:
    return {
        int(key.removeprefix(prefix)): _ExecutableRecord(**json.loads(value))
        for key, value in client.key_value_dir_get(prefix)
    }


def _warn_unsupported_text() -> None:
    global _unsupported_text_warned
    with _unsupported_text_warning_lock:
        if not _unsupported_text_warned:
            logger.warning("Optimized HLO text is unavailable; skipping distributed GPU executable comparison")
            _unsupported_text_warned = True


def _optimized_hlo_text(loaded: object, runtime_error_type: type[Exception]) -> str | None:
    get_hlo_text = getattr(loaded, "get_hlo_text", None)
    if get_hlo_text is not None:
        try:
            return get_hlo_text()
        except (NotImplementedError, runtime_error_type) as exc:
            if not isinstance(exc, NotImplementedError) and not str(exc).startswith("UNIMPLEMENTED"):
                raise

    hlo_modules = getattr(loaded, "hlo_modules", None)
    if hlo_modules is not None:
        try:
            return "\n\n".join(module.to_string() for module in hlo_modules())
        except (NotImplementedError, runtime_error_type) as exc:
            if not isinstance(exc, NotImplementedError) and not str(exc).startswith("UNIMPLEMENTED"):
                raise
    return None


def _autotune_cache_file_count(autotune_dir: str) -> int | None:
    if not autotune_dir:
        return 0
    try:
        with os.scandir(autotune_dir) as entries:
            return sum(entry.is_file() for entry in entries)
    except FileNotFoundError:
        return 0
    except OSError:
        return None


@dataclass
class _CompileGuardState:
    client: _Coordinator | None = None
    locks: dict[tuple[int, ...], threading.Lock] = field(default_factory=lambda: defaultdict(threading.Lock))
    sequences: dict[tuple[int, ...], itertools.count] = field(default_factory=lambda: defaultdict(itertools.count))
    mutex: threading.Lock = field(default_factory=threading.Lock)

    def check(self, client: _Coordinator, process_ids: tuple[int, ...], process_id: int, fingerprint: str) -> int:
        with self.mutex:
            if client is not self.client:
                self.client = client
                self.locks.clear()
                self.sequences.clear()
            lock = self.locks[process_ids]
        with lock:
            sequence = next(self.sequences[process_ids])
            check_distributed_compile_fingerprint(client, process_ids, process_id, sequence, fingerprint)
            return sequence


def install_gpu_compile_guard() -> None:
    """Guard JAX's shared compile entry point after Iris joins a distributed world."""
    import numpy as np  # noqa: PLC0415 - optional Iris dependency
    from jax._src import (  # noqa: PLC0415 - optional Iris dependency
        compilation_cache,
        compiler,
        config,
        distributed,
        profiler,
    )
    from jax._src.lib import xla_client as xc  # noqa: PLC0415 - optional Iris dependency
    from jax._src.lib.mlir import ir  # noqa: PLC0415 - optional Iris dependency
    from jax.errors import JaxRuntimeError  # noqa: PLC0415 - optional Iris dependency

    original = compiler.compile_or_get_cached
    if getattr(original, "_iris_compile_guard", False):
        return
    state = _CompileGuardState()

    @wraps(original)
    def guarded_compile_or_get_cached(
        backend: xc.Client,
        computation: ir.Module,
        devices: np.ndarray,
        compile_options: xc.CompileOptions,
        host_callbacks: Sequence[object],
        executable_devices: xc.DeviceList,
        pgle_profiler: profiler.PGLEProfiler | None = None,
    ) -> xc.LoadedExecutable:
        process_ids = tuple(sorted({device.process_index for device in devices.flat}))
        if backend.platform == "gpu" and len(process_ids) > 1:
            client = distributed.global_state.client
            if client is None:
                raise RuntimeError("multi-process GPU compile has no JAX coordinator")
            process_id = distributed.global_state.process_id
            fingerprint = compilation_cache.get_cache_key(
                computation,
                devices,
                compile_options,
                backend,
                config.remove_custom_partitioning_ptr_from_cache_key.value,
            )
            sequence = state.check(client, process_ids, process_id, fingerprint)
            logger.debug("GPU compile fingerprint matched across processes %s: %s", process_ids, fingerprint)

        loaded = original(
            backend, computation, devices, compile_options, host_callbacks, executable_devices, pgle_profiler
        )
        if backend.platform == "gpu" and len(process_ids) > 1:
            # Persistent cache hits load process 0's executable on every rank, so equality
            # here does not establish that every rank had identical compile inputs.
            extraction_error = None
            try:
                executable_text = _optimized_hlo_text(loaded, JaxRuntimeError)
            except Exception as exc:
                executable_text = None
                extraction_error = exc
            autotune_dir = compile_options.executable_build_options.debug_options.xla_gpu_per_fusion_autotune_cache_dir
            check_distributed_executable_text(
                client,
                process_ids,
                process_id,
                sequence,
                fingerprint,
                executable_text,
                autotune_dir,
                extraction_error,
            )
        return loaded

    guarded_compile_or_get_cached._iris_compile_guard = True
    compiler.compile_or_get_cached = guarded_compile_or_get_cached
