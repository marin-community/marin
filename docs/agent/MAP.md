## Package Index

### marin
`marin.processing.classification.deduplication` — Exact-paragraph, exact-document, and fuzzy-document dedup via Zephyr pipelines; writes attribute sidecar files marking duplicates. → `docs/agent/packages/marin.processing.classification.deduplication.md`

### levanter
*(no package docs loaded yet)*

### haliax
*(no package docs loaded yet)*

### fray
*(no package docs loaded yet)*

### rigging
`rigging.distributed_lock` — Distributed lease-based locking over GCS, S3, local, and fsspec paths; atomic CAS on GCS/S3, best-effort on fsspec. → `docs/agent/packages/rigging.distributed_lock.md`

### iris
`iris.cluster.controller` — Cluster control plane: task scheduling, worker heartbeats, VM autoscaling, WAL-mode SQLite state with GCS checkpoints, and gRPC RPC surface. → `docs/agent/packages/iris.cluster.controller.md`

### zephyr
*(no package docs loaded yet)*

### dupekit
*(no package docs loaded yet)*

---

## Dependency Edges

`marin -> zephyr` (dedup pipelines use `Dataset`, `ZephyrContext`)
`marin -> dupekit` (fuzzy dedup uses xxHash3 and MinHash/LSH primitives)
`marin -> fray` (worker/coordinator dispatch via `ResourceConfig`)
`marin -> rigging` (artifact storage reads/writes via fsspec)
`levanter -> haliax` (named tensors for JAX training)
`zephyr -> rigging` (dataset I/O over GCS/S3)

---

## Entry Points

`marin.processing.classification.deduplication.exact.dedup_exact_document(*, input_paths, output_path, text_field, max_parallelism, ...)` — Full-document exact dedup via xxHash3-128; writes `is_dup` sidecars.
`marin.processing.classification.deduplication.exact.dedup_exact_paragraph(*, input_paths, output_path, text_field, max_parallelism, ...)` — Paragraph-level exact dedup; writes `dup_spans` sidecars.
`marin.processing.classification.deduplication.fuzzy.dedup_fuzzy_document(*, input_paths, output_path, text_field, max_parallelism, ...)` — MinHash LSH + connected components fuzzy dedup; writes `is_dup` sidecars.
`marin.processing.classification.deduplication.connected_components.connected_components(ds, ctx, *, output_dir, max_iterations)` — Iterative Hash-to-Min CC; returns `(converged, vortex_paths)`.
`marin.processing.classification.deduplication.dedup_commons.group_files(files, num_groups)` — Round-robin file grouping to cap Zephyr shard count.
`marin.processing.classification.deduplication.dedup_commons.make_document_dedup_aggregator(*, idx_to_path, input_paths, output_path, counter_prefix)` — Factory for `group_by` reducer shared by exact-doc and fuzzy-doc dedup.
`rigging.distributed_lock.create_lock(lock_path, worker_id)` — Instantiate correct `DistributedLease` implementation for the given path prefix.
`rigging.distributed_lock.DistributedLease.try_acquire()` — Atomically attempt lock acquisition; returns `False` if held by another worker.
`rigging.distributed_lock.DistributedLease.refresh()` — Extend lease; raises `LeaseLostError` if ownership lost.
`rigging.distributed_lock.DistributedLease.release()` — Release lock; idempotent.
`iris.cluster.controller.main.run_controller_serve(cluster_config, *, host, port, checkpoint_path, ...)` — Daemon entry point; blocks until SIGTERM.
`iris.cluster.controller.controller.Controller.launch_job(request)` — Submit a job to the controller; returns `LaunchJobResponse`.
`iris.cluster.controller.autoscaler.route_demand(groups, demand_entries, timestamp)` — Two-phase priority demand routing; returns `RoutingDecision`.

---

## Conventions

- **Config style:** Draccus dataclasses; compose sub-configs by embedding, not inheritance. Force explicit specification of critical parameters; centralize defaults in one location.
- **Artifact paths:** Rooted at `MARIN_PREFIX`; output paths constructed per-step. Version hashing used to derive stable, content-addressable paths across runs.
- **Execution patterns:** Steps described via `StepSpec`; remote execution via `RemoteCallable`; workers dispatched through Fray `ResourceConfig` (cpu, memory). `max_parallelism` caps concurrent Zephyr workers.
- **Dedup output contract:** All dedup functions write attribute sidecar files marking duplicates — they do **not** produce filtered corpora. Downstream steps must apply sidecars to remove records.
- **No backward-compat shims:** Update all call sites on API changes; no `hasattr` guards or fallback paths.
