# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check distributed GPU compile identities before XLA's autotuner rendezvous."""

from __future__ import annotations

import itertools
import logging
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Protocol

logger = logging.getLogger(__name__)

# Large checkpoint loads can leave healthy ranks minutes apart at their first compile.
_COMPILE_FINGERPRINT_TIMEOUT_MS = 30 * 60_000
_COMPILE_FINGERPRINT_PREFIX = "iris/compile-fingerprint/v1"


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


@dataclass
class _CompileGuardState:
    client: _Coordinator | None = None
    locks: dict[tuple[int, ...], threading.Lock] = field(default_factory=lambda: defaultdict(threading.Lock))
    sequences: dict[tuple[int, ...], itertools.count] = field(default_factory=lambda: defaultdict(itertools.count))
    mutex: threading.Lock = field(default_factory=threading.Lock)

    def check(self, client: _Coordinator, process_ids: tuple[int, ...], process_id: int, fingerprint: str) -> None:
        with self.mutex:
            if client is not self.client:
                self.client = client
                self.locks.clear()
                self.sequences.clear()
            lock = self.locks[process_ids]
        with lock:
            sequence = next(self.sequences[process_ids])
            check_distributed_compile_fingerprint(client, process_ids, process_id, sequence, fingerprint)


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
            state.check(client, process_ids, process_id, fingerprint)
            logger.debug("GPU compile fingerprint matched across processes %s: %s", process_ids, fingerprint)

        return original(
            backend, computation, devices, compile_options, host_callbacks, executable_devices, pgle_profiler
        )

    guarded_compile_or_get_cached._iris_compile_guard = True
    compiler.compile_or_get_cached = guarded_compile_or_get_cached
