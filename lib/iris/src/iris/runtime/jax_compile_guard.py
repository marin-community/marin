# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check distributed GPU compile identities before XLA's autotuner rendezvous."""

import itertools
import logging
import threading
from collections import defaultdict
from collections.abc import Sequence
from functools import wraps
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_COMPILE_FINGERPRINT_TIMEOUT_MS = 60_000
_COMPILE_FINGERPRINT_PREFIX = "iris/compile-fingerprint/v1"
_exchange_locks: dict[tuple[int, ...], threading.Lock] = defaultdict(threading.Lock)
_sequences: dict[tuple[int, ...], itertools.count] = defaultdict(itertools.count)


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


def install_gpu_compile_guard() -> None:
    """Guard JAX's shared compile entry point after Iris joins a distributed world."""
    from jax._src import cache_key, compiler, distributed  # noqa: PLC0415 - optional Iris dependency

    original = compiler.compile_or_get_cached
    if getattr(original, "_iris_compile_guard", False):
        return

    @wraps(original)
    def guarded_compile_or_get_cached(
        backend: Any,
        computation: Any,
        devices: Any,
        compile_options: Any,
        host_callbacks: Any,
        executable_devices: Any,
        pgle_profiler: Any = None,
    ) -> Any:
        process_ids = tuple(sorted({device.process_index for device in devices.flat}))
        if backend.platform == "gpu" and len(process_ids) > 1:
            client = distributed.global_state.client
            if client is None:
                raise RuntimeError("multi-process GPU compile has no JAX coordinator")
            process_id = distributed.global_state.process_id
            fingerprint = cache_key.get(computation, devices, compile_options, backend)
            with _exchange_locks[process_ids]:
                sequence = next(_sequences[process_ids])
                check_distributed_compile_fingerprint(client, process_ids, process_id, sequence, fingerprint)
            logger.debug("GPU compile fingerprint matched across processes %s: %s", process_ids, fingerprint)

        return original(
            backend, computation, devices, compile_options, host_callbacks, executable_devices, pgle_profiler
        )

    guarded_compile_or_get_cached._iris_compile_guard = True
    compiler.compile_or_get_cached = guarded_compile_or_get_cached
