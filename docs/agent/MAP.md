## Module Index

### rigging
`rigging.distributed_lock` — Distributed lease-based locking over GCS, S3, local, and fsspec backends for coordinating exclusive access across workers. → `docs/agent/modules/rigging.distributed_lock.md`

### marin
*(no summaries provided)*

### levanter
*(no summaries provided)*

### haliax
*(no summaries provided)*

### fray
*(no summaries provided)*

### iris
*(no summaries provided)*

### zephyr
*(no summaries provided)*

### dupekit
*(no summaries provided)*

---

## Dependency Edges

*(No cross-library import edges present in provided summaries.)*

---

## Entry Points

*(Summaries for marin.run, marin.execution, marin.training, marin.processing, and levanter.main were not provided. Populate this section when those module summaries are supplied.)*

---

## Conventions

- **Config style:** Draccus dataclasses; compose sub-configs via embedding, not inheritance; no `default_*` wrappers; force explicit critical parameters.
- **Artifact paths:** Rooted at `MARIN_PREFIX`; output paths constructed per-step; version hashing used to avoid collisions.
- **Execution patterns:** Steps described via `StepSpec`; remote execution dispatched through `RemoteCallable`; parallelism managed by Fray.
- **Locking:** Use `rigging.distributed_lock.create_lock(path, worker_id)` to acquire exclusive access to shared resources; call `refresh()` in a heartbeat loop and treat `LeaseLostError` as fatal.
- **Lock backend selection:** Path prefix determines backend — `gs://` → `GcsLease` (atomic CAS), `s3://` → `S3Lease` (atomic CAS), local path → `LocalFileLease`, other → `FsspecLease` (best-effort only).
- **No backward compat shims:** Update all call sites; no `hasattr` guards or fallback paths.
- **Imports:** All at top of file; no local imports except to break cycles; no `TYPE_CHECKING` guards.
