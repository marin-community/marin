## Module Index

### rigging
`rigging.distributed_lock` — Distributed mutual exclusion via heartbeated lease files on GCS, S3, local, or fsspec backends; coordinates exclusive resource access across workers.

---

## Dependency Edges

`rigging.distributed_lock -> google.cloud.storage` (GCS conditional writes)
`rigging.distributed_lock -> botocore` (S3 conditional PutObject)
`rigging.distributed_lock -> fsspec` (generic filesystem abstraction and URI dispatch)

---

## Entry Points

`rigging.distributed_lock.create_lock(lock_path, worker_id=None)` — Instantiate the correct backend lease for a given URI scheme; primary entry point for acquiring distributed locks.

---

## Conventions

**Config style:** Not applicable to this module; no draccus dataclasses present.

**Artifact paths:** Lock files are addressed by URI scheme: `gs://` routes to `GcsLease`, `s3://` routes to `S3Lease`, bare paths route to `LocalFileLease`, all others fall back to `FsspecLease`.

**Naming patterns:** Worker IDs default to `{hostname}-{thread_id}`; in containerized environments supply an explicit globally-unique ID to avoid collisions.

**Common patterns:** Lease acquire/refresh/release protocol — call `try_acquire()`, loop calling `refresh()` as a heartbeat, call `release()` on exit. `LeaseLostError` propagates from `refresh()` and must be caught at the operation boundary, not swallowed. Stale lease theft is automatic: any lease older than `HEARTBEAT_TIMEOUT` is eligible to be stolen by a competing `try_acquire()` call. `FsspecLease` provides only weak atomicity (write-then-readback); use GCS or S3 backends for strong mutual exclusion guarantees.
