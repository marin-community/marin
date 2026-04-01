## Purpose
Provides distributed lease-based locking over GCS, S3, local filesystem, and arbitrary fsspec paths. An agent touches this module when coordinating exclusive access to a shared resource (e.g., a checkpoint directory) across multiple workers or hosts.

## Public API

### Factory
- `create_lock(lock_path: str, worker_id: str | None = None) -> DistributedLease` — Instantiates the correct lease implementation based on path prefix (`gs://`, `s3://`, local, or fsspec fallback). `distributed_lock.py:425`
- `default_worker_id() -> str` — Returns `hostname-thread_id` as a default unique holder identifier. `distributed_lock.py:55`

### Exceptions
- `LeaseLostError` — Raised by `refresh()` when the lock has been taken by another worker or the lock file has disappeared. `distributed_lock.py:37`

### Core Abstraction
- `DistributedLease` — Abstract base class; subclass for each storage backend. `distributed_lock.py:77`
  - `try_acquire() -> bool` — Atomically attempt to acquire; returns False if held by another active worker. `distributed_lock.py:110`
  - `refresh() -> None` — Extend lease timestamp; raises `LeaseLostError` if ownership was lost. `distributed_lock.py:138`
  - `release() -> None` — Release lock if held by this worker; idempotent. `distributed_lock.py:153`
  - `has_active_holder() -> bool` — Returns True if any worker holds a non-stale lease. `distributed_lock.py:162`

### Implementations
- `GcsLease(lock_path: str, worker_id: str | None = None)` — GCS-backed lease via generation-based conditional writes. `distributed_lock.py:181`
- `S3Lease(lock_path: str, worker_id: str | None = None)` — S3-backed lease via `If-None-Match`/`If-Match` conditional PutObject using botocore directly. `distributed_lock.py:237`
- `LocalFileLease(lock_path: str, worker_id: str | None = None)` — Local filesystem lease using `fcntl` exclusive locking. `distributed_lock.py:327`
- `FsspecLease(lock_path: str, worker_id: str | None = None)` — Best-effort lease for arbitrary fsspec filesystems; no atomic CAS guarantee. `distributed_lock.py:374`

### Data
- `Lease` — Dataclass holding `worker_id: str` and `timestamp: float`; `is_stale()` checks against `HEARTBEAT_TIMEOUT`. `distributed_lock.py:45`

## Dependencies
No cross-monorepo dependencies. This module is self-contained within `rigging`.

## Key Abstractions
- `DistributedLease` — Abstract lease holder; all locking protocol logic lives here; subclasses only implement storage read/write/delete.
- `Lease` — Serialized on-disk lock state; staleness is determined by comparing `timestamp` against `HEARTBEAT_TIMEOUT` at read time.
- `LeaseLostError` — Fatal signal to the holder that it must stop; raised by `refresh()` and must not be caught silently.
- `GcsLease` / `S3Lease` — Backends with true atomic CAS guarantees via storage-native conditional writes.
- `FsspecLease` — Best-effort fallback; uses write-then-readback with a `time.sleep(0.1)` race check — not safe under high contention.

## Gotchas
- `FsspecLease._write()` is **not atomically safe**: it writes unconditionally then reads back after a 100ms sleep. Under concurrent access it can silently fail to detect a race; prefer GCS or S3 paths when correctness matters.
- `refresh()` raises `LeaseLostError` if the lock file is **absent**, not just held by another worker — a missing file means another worker stole and released the lease, and this is treated as a fatal loss, not a retriable condition.
- `S3Lease._write()` asserts `self._last_etag is not None` when `if_generation_match != 0`; calling `_write` for an update without a preceding `_read_with_generation` in the same instance will panic.
- `S3Lease` reads `AWS_ENDPOINT_URL_S3` then `AWS_ENDPOINT_URL` from the environment to configure custom endpoints; the endpoint must support virtual-host style addressing, not path-style.
- `try_acquire()` returns `True` (idempotent re-acquire) if the current `worker_id` already holds the lock, even if the lease is stale — callers must not rely on `try_acquire` to detect their own expiry; use `refresh()` for that.
