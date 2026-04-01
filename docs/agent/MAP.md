Here is the updated `docs/agent/MAP.md`:

---

## Package Index

### marin
`marin.processing.classification.deduplication` — Exact and fuzzy document/paragraph deduplication via Zephyr, producing Vortex sidecar attribute files. → `docs/agent/packages/marin.processing.classification.deduplication.md`

### levanter
*(no summaries provided)*

### haliax
*(no summaries provided)*

### fray
*(no summaries provided)*

### rigging
`rigging.distributed_lock` — Distributed lease-based locking over GCS, S3, local, and fsspec backends for coordinating exclusive access across workers. → `docs/agent/modules/rigging.distributed_lock.md`

### iris
*(no summaries provided)*

### zephyr
*(no summaries provided)*

### dupekit
*(no summaries provided)*

---

## Dependency Edges

`marin.processing.classification.deduplication` -> `zephyr` (ZephyrContext, Dataset, distributed dataflow)
`marin.processing.classification.deduplication` -> `dupekit` (MinHash LSH, native Rust thread pool)

---

## Entry Points

`dedup_exact_document(*, input_paths, output_path, text_field, max_parallelism, ...)` — Full-document exact dedup via xxHash-128; writes Vortex sidecars with `dup_doc: True`.
`dedup_exact_paragraph(*, input_paths, output_path, text_field, max_parallelism, ...)` — Paragraph-level exact dedup; sidecars carry `dup_spans` lists.
`dedup_fuzzy_document(*, input_paths, output_path, text_field, max_parallelism, ...)` — Near-duplicate dedup via MinHash LSH + connected components; writes CC metadata and sidecars.
`connected_components(ds, ctx, *, output_dir, max_iterations, preserve_singletons)` — Hash-to-Min label propagation; returns `(converged, vortex_paths)`.
`finalize_dedup(shard_results, mode, method, level)` — Aggregates shard counters, logs summary, finalizes W&B run, returns stats dict.

---

## Conventions

- **Config style:** Draccus dataclasses; compose sub-configs via embedding, not inheritance; no `default_*` wrappers; force explicit critical parameters (e.g. `max_parallelism` has no default).
- **Artifact paths:** Rooted at `MARIN_PREFIX`; output paths constructed per-step; version hashing avoids collisions; dedup writes to `{output_path}/data/` (sidecars) and `{output_path}/metadata/` (CC state).
- **Execution patterns:** Steps described via `StepSpec`; remote execution dispatched through `RemoteCallable`; parallelism managed by Fray; dedup entry points are direct function calls dispatched onto Zephyr workers.
- **Dedup output contract:** All dedup functions write attribute sidecars marking duplicates — they do not produce filtered corpora. Downstream steps must read `dup_doc`/`dup_spans` to filter.
- **Locking:** Use `rigging.distributed_lock.create_lock(path, worker_id)`; call `refresh()` in heartbeat loop; treat `LeaseLostError` as fatal. Path prefix selects backend (`gs://` → GCS CAS, `s3://` → S3 CAS, local → file lock).
- **No backward compat shims:** Update all call sites; no `hasattr` guards or fallback paths.
- **Imports:** All at top of file; no local imports except to break cycles; no `TYPE_CHECKING` guards.

---

**Notes on gaps:** The `marin`, `levanter`, `haliax`, `fray`, `iris`, `zephyr`, and `dupekit` sections in the Package Index and the Entry Points section for `marin.run`/`marin.execution`/`marin.training`/`levanter.main` will populate as those package summaries are supplied. Two cross-library dependency edges were inferred from the deduplication summary (`→ zephyr`, `→ dupekit`).
