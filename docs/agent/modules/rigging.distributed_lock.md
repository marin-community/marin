## Purpose
`rigging.distributed_lock` provides distributed mutual exclusion via lock files stored on pluggable backends (GCS, S3, local filesystem, or any fsspec-compatible store). An agent uses it to coordinate exclusive access to shared resources across multiple workers or processes, with lease-based heartbeating to detect and recover from dead holders.

## Public API

### Factory
`create_lock(lock_path, worker_id=None) -> DistributedLease` — Instantiates the correct backend lease class based on URI scheme (`gs://`, `s3://`, local path, or fsspec fallback).

### Lease Backends
`GcsLease(lock_path, worker_id=None)` — Atomic distributed lease backed by GCS conditional writes using object generation numbers.
`S3Lease(lock_path, worker_id=None)` — Atomic distributed lease backed by S3-compatible storage using ETag-based conditional PutObject.
`LocalFileLease(lock_path, worker_id=None)` — Local-filesystem lease using `fcntl` advisory locks for single-host serialization.
`FsspecLease(lock_path, worker_id=None)` — Best-effort lease for arbitrary fsspec filesystems; atomicity is weak (write-then-readback).

### Core Operations (on any `DistributedLease`)
`try_acquire() -> bool` — Non-blocking single attempt to acquire; steals stale leases; returns False if a live holder exists.
`refresh() -> None` — Heartbeat: updates the lease timestamp; raises `LeaseLostError` if ownership was lost.
`release() -> None` — Deletes the lock file if held by this worker; idempotent, silently ignores missing file.
`has_active_holder() -> bool` — Checks whether any worker currently holds a non-stale lease.

### Utilities
`default_worker_id() -> str` — Returns `'{hostname}-{thread_id}'` as a default unique worker identifier.

### Data Types
`Lease` — Dataclass holding `worker_id` and heartbeat timestamp; `is_stale()` checks against `HEARTBEAT_TIMEOUT`.
`LeaseLostError` — Exception raised when the current worker's lease has been stolen.

## Dependencies
- `google.cloud.storage` — GCS object reads/writes and generation-based conditional puts (used by `GcsLease`)
- `google.api_core.exceptions` — `NotFound` handling for missing GCS objects
- `botocore` — S3 client creation, conditional PutObject via event hooks, `ClientError` (used by `S3Lease`)
- `fsspec` — Filesystem abstraction and URL resolution (used by `FsspecLease`)
- `fcntl` — Advisory file locking for `LocalFileLease`
- `os` — `uname`, `makedirs`, `remove`, environment variable access
- `threading` — Thread ID for `default_worker_id`
- `time` — Heartbeat timestamp and staleness checks
- `abc` — Abstract base class machinery for `DistributedLease`

## Key Abstractions
`DistributedLease` — Abstract base implementing acquire/refresh/release protocol; subclasses supply atomic storage primitives.
`Lease` — Persisted lock state: who holds it and when they last heartbeated; staleness is determined by `HEARTBEAT_TIMEOUT`.
`LeaseLostError` — Fatal signal that another worker has stolen the lock; callers must stop operating on the shared resource immediately.
`HEARTBEAT_TIMEOUT` — Module-level constant controlling how long before an un-refreshed lease is considered stale and eligible to be stolen.

## Gotchas
- `FsspecLease` provides **weak atomicity** — it writes then reads back after a sleep to detect races; two workers can simultaneously believe they hold the lock on backends without conditional-write support.
- `refresh()` raises `LeaseLostError` rather than returning a status; callers that don't `try/except` it will propagate an unhandled exception when ownership is lost mid-operation.
- `try_acquire()` will **steal** a stale lease without warning — if a holder's heartbeat falls behind `HEARTBEAT_TIMEOUT`, another worker will take the lock even if the original holder is still alive but slow.
- `release()` is silently idempotent; it does **not** raise if the lock is held by a different worker — it simply does nothing, so a worker cannot use `release()` to verify it still holds the lock.
- `worker_id` defaults to `'{hostname}-{thread_id}'`; in containerized or forked environments where hostnames are identical and thread IDs may collide, callers should supply an explicit globally-unique `worker_id` to avoid false ownership matches.
