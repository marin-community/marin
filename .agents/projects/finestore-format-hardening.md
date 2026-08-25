# finestore format hardening (before locking format_version = 1)

Driven by rjpower review of PR #7976. Goal: settle the expensive-to-change format-level
decisions before real archives exist. Schema-shape (`schema_version`) items are cheap later;
format-level (`format_version`) items are not.

## Decisions

1. **Schema machinery** — `finestore.arrow_schema(model_or_schema) -> pa.Schema`. Accepts a
   pydantic `BaseModel` subclass, a dataclass, or a `pa.Schema` (passthrough). Maps: scalars →
   arrow scalar; `T | None` → nullable `T`; `list[T]` → `list<T>`; `dict[K,V]` → `map<K,V>`;
   nested model/dataclass → `struct`; str/int enum → string/int64. All fields nullable (schema
   evolution). finestore's flush augments the caller schema with `_seq`/`_writer` stamps.
2. **Blobs stay a Parquet table keyed by name** (NOT one object per blob). Harbor writes many tiny
   blobs; one object each = a synchronous PUT per blob, the latency finestore exists to avoid. Keep
   `store.write(name, metadata, data)` appending to the reserved `blobs` table (merge_key = name), so
   writes batch through the background flusher (non-blocking). Reads are log-n: a compacted shard is
   name-sorted, so `read_blob(name)`'s `name ==` filter prunes to the one row group whose footer
   min/max brackets the name (verified: 50k blobs -> 5 row groups, lookup touches 1). Column
   projection reads names/metadata without the `data` payload. (Earlier draft made blobs objects; the
   maintainer corrected it -- reverted.)
3. **seal() = flush + compact each table + SEALED.** After seal: blobs-as-objects + one deduped
   Parquet per table, safe for a generic reader.
4. **Resume** — `CompositeReader.keys(table) -> set[tuple]` (deduped merge-key set). Harbor reads
   it to skip done trials. Stable writer ids already used (`harbor`/`evalchemy`).
5. **No shadowing** — dedup precedence `(seq, gen)` (seq-first; gen breaks the crash-mid-compaction
   exact-seq tie). Store resumes `_seq` above the table's max persisted `_seq` at registration, so a
   new write always outranks a prior one. Applies in reader `_deduplicate` AND compaction merge.
5b. **Cross-writer** — contract: writers own disjoint key spaces. No global clock; same-key writes
   from two uncoordinated writers are undefined (documented). Harbor=trials, Evalchemy=subtasks.

## Schema ownership
Eval models stay in marin (`EvalSample`/`StepRecord`). finestore ships `arrow_schema()` only.

## Additional eval-schema features (this commit, since it bumps schema_version)
- Pin `samples`/`steps` schemas via `arrow_schema`. `metrics: dict[str,float]` → `map<string,double>`
  (kills type drift AND the empty-struct landmine; read via `to_pylist(maps_as_pydicts="strict")`).
- Drop dead `exchange_uri`.
- (Consider) per-sample timing/token counts — batch in if cheap.
- Bump `SCHEMA_VERSION` 2 → 3.

## Ripple
- evaldash reads: materialize rows with `maps_as_pydicts="strict"` (metrics is now a map).
- `EvaluationStore.add_trajectory`: `store.write(name, raw)` (metadata param dropped).
- Redeploy evaldash to surface any new fields (graceful otherwise — pydantic ignores extras).

## Considered-and-deferred schema features
Held out of this commit on purpose (all cheap to add later via nullable evolution + a schema_version
bump, per the "minor schema change" walkthrough):
- Per-sample timing / token / cost totals: lm-eval emits none per sample; Harbor's live in the steps
  table already. Denormalizing totals onto the sample row is premature.
- The eval's driving YAML config, run timing: these are run-level, not per-sample. They belong in a
  future archive-level `run`/`meta` table, not duplicated across every sample row. Not sneaking a new
  table in without direction.
- `EvaluationStore.completed_keys()` ergonomic wrapper: the finestore primitive
  (`CompositeReader.keys`) + auto-seq-resume are in place; the Harbor-side skip integration that would
  call it is a separate change. No caller yet → no wrapper yet.

## Filesystem-factory gotcha (hit while wiring resume)
Adding a read (`max_seq`) to the write path exposed that reader.py bound `url_to_fs` at import, so the
harbor `[s3]` memory-fs monkeypatch (which patches `rigging.filesystem.factory.url_to_fs`) was bypassed
→ real S3 → NoCredentialsError. Fixed: reader.py + compaction.py now call `factory.url_to_fs` at call
time (matches shard_writer's StoragePath/atomic_rename pattern). Locked by a finestore-level test that
routes `s3://` to an in-memory store and round-trips write+seal+scan.

## Flush + row-group calibration (borrowed from finelog)
Maintainer steer: cap RAM to a reasonable amount + get decent row-group pruning; skip finelog's
1-flush/sec floor (eval write volume does not need it).
- Buffer flush trigger: `min(5s time ceiling, 100 MiB byte cap)`. Replaced the 20k-ROW cap with
  `DEFAULT_MAX_BUFFER_BYTES` (finelog's SEGMENT_TARGET_BYTES) — rows are a bad memory proxy once one
  row is a multi-MB blob. Cheap per-append `_estimate_bytes` (payload/text by length, recurse
  containers, flat scalar). This alone collapses the RL 1.3M-tiny-blob path to ~3 shards (per 100 MB).
- Row groups capped at `ROW_GROUP_ROWS = 16_384` (finelog's ROW_GROUP_SIZE) at BOTH L0 flush (via
  `write_table(row_group_size=)`) and compaction batch, so a big flush still prunes by row group.
- `max_seq` (resume) now reads footer `_seq` max stats only, never the column — finelog-style footer
  recovery; cheap however large the archive.
- NOT done: min-flush-interval floor, bloom filters (L0 point-lookup speed) — deferred as unneeded.

## Status: DONE. schema.py + finestore core (blobs/dedup/resume/seal/keys) + marin pinning +
## evaldash map reads all implemented and tested (finestore 29 + evaluation 91 green). Next: lint,
## commit, push to #7976, monitor CI. evaldash pulumi up still handed off to an authorized operator.
