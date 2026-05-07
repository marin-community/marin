# Finelog Compaction Refactor

**Status:** Draft v2 &nbsp;•&nbsp; **Author:** agent + russell &nbsp;•&nbsp; **Date:** 2026-05-05

One-line summary: Extract finelog's compaction logic into a `Compactor` class with a level-target merge policy, a single per-segment filename scheme, and a `level` column on the catalog. Compaction stays in DuckDB but inputs are bounded by the level scheme so spilling is a non-issue.

## Goals

- Cap finalized parquet segments at ~256 MiB so eviction granularity is bounded and remote round-trip costs stay predictable.
- Make compaction policy a first-class config object (`CompactionConfig`) — three knobs, all defaultable.
- Replace the implicit `tmp_*` / `logs_*` filename split with a single `seg_L<n>_<min_seq>.parquet` scheme + a `level` column on the catalog.
- Preserve the within-segment `(key_column, seq)` sort invariant so row-group `[min_key, max_key]` stays tight on `key`.

## Non-goals

- Changing the GCS offload path (already moved to `_OffloadWorker` in `duckdb_store.py`).
- Read-side row-group pruning improvements (orthogonal).
- Backwards compat for on-disk filenames — boot-time migration renames `tmp_*` → `seg_L0_*`, `logs_*` → `seg_L1_*` and rewrites the catalog atomically.
- Cross-namespace fairness beyond the bg loop's natural round-robin.

## Background

Compaction today lives inline on `DiskLogNamespace`. `_compaction_step` (`log_namespace.py:859`) snapshots all `tmp_*` segments, runs one DuckDB `COPY (SELECT … ORDER BY …) TO …` (`:949`) into a staging file, renames + swaps under the visibility lock. Triggered when `tmp_count > _max_tmp_segments_before_compact` (default 10) or every `compaction_interval_sec` (default 600 s). State is filename-encoded: `tmp_*` (post-flush) vs `logs_*` (compacted).

Production (one marin namespace): ~73–100 finalized parquets, 13–18 MiB each, ~1.3 GiB total. Files never grow above ~20 MiB because the policy is flat — every compaction merges all current tmps into one logs_ file, then never re-touches it.

The lock-during-COPY bug was fixed in commit `0acec62c0` so the COPY no longer blocks writers; that's a pre-condition for the design below.

## Proposed design

### 1. `Compactor` class

New module `lib/finelog/src/finelog/store/compactor.py`. Stateless. The namespace owns locks, catalog mutation, and offload submission; the Compactor owns planning + the merge SQL.

```python
@dataclass(frozen=True)
class CompactionConfig:
    # Promote L_n -> L_{n+1} when the longest contiguous run of L_n
    # segments has combined byte_size >= level_targets[n]. Terminal
    # level is len(level_targets); never re-compacted.
    level_targets: tuple[int, ...] = (64 * MiB, 256 * MiB)
    # Bg loop wakes this often to re-plan.
    check_interval_sec: float = 30.0
    # Promote L0 even if the size threshold isn't met, when the oldest
    # L0 has been sitting more than this long. Keeps low-volume
    # namespaces from leaking small files.
    max_l0_age_sec: float = 300.0


class Compactor:
    def __init__(self, config: CompactionConfig, *, schema: Schema): ...
    def plan(self, segments: Sequence[SegmentRow], now_ms: int) -> CompactionJob | None: ...
    def merge_sql(self, job: CompactionJob, *, staging_path: Path) -> str: ...

CompactionJob = NamedTuple("CompactionJob", [
    ("inputs", tuple[SegmentRow, ...]),
    ("output_level", int),
    ("output_min_seq", int),
    ("output_max_seq", int),
])
```

`level_targets` is the only policy knob. `level_fanout`, `max_segment_bytes`, `min_inputs_to_compact`, `target_row_group_bytes`, and `throughput_budget_mbps` were all derivable or unnecessary; they're gone.

### 2. Levels and policy

L0 is implicit — whatever flush emits (no target, no L0 compaction; flush writes directly into the L0 namespace).

Promotion rule: pick the longest contiguous-by-`min_seq` run of L_n segments whose summed `byte_size` is ≥ `level_targets[n]`. Merge them into one L_{n+1} segment. If a single segment alone meets the threshold (e.g. flush emitted 100 MiB and L0→L1 target is 64 MiB), short-circuit to a level bump only — `os.rename` + UPDATE level — no rewrite.

L_max (= `len(level_targets)`) is terminal. Eviction takes those over (§8).

**Why not binary-LSM (`_merge_chunks` style, "each chunk ≥ 2× the previous")?** It has no clean terminal state; every new flush eventually re-touches the largest file, rewriting ~3× the daily ingest volume. The level-target policy rewrites each byte exactly `len(level_targets) - 1` times (default: 2). Read fanout drops from ~73 files to `~level_targets[0]/L0_avg + ⌈total / level_targets[-1]⌉` ≈ 4 + 5 = 9 in steady state.

**Adjacency is naturally satisfied.** Levels are time-ordered: L_{N+1} segments cover seq ranges older than the live L_N tail because they were promoted earlier. Eviction is FIFO-by-`min_seq` (oldest first; never picks from the middle — see §8), so the local segment set is always a contiguous suffix of the global seq range. The planner picks the longest run of L_n segments whose summed `byte_size` ≥ `level_targets[n]`; a contiguity check (`prev.max_seq + 1 == next.min_seq`) is included as a defensive assertion but should never fail in practice.

**No splitting.** Removing `max_segment_bytes` removes the split-output path. A promotion is strictly many-to-one; one CompactionJob → one output segment. If the inputs are slightly lumpy and the output lands at, say, 280 MiB instead of 256, that's fine — it won't promote further (no level above) and won't be re-touched.

### 3. Merge implementation

**Use DuckDB COPY.** Each input is already sorted on `(key_column, seq)`. The Compactor builds a single SQL string:

```sql
COPY (
  SELECT <projected columns>
  FROM read_parquet([<input paths>], union_by_name=true)
  ORDER BY <key_column>, seq
) TO '<staging path>' (
  FORMAT 'parquet',
  ROW_GROUP_SIZE 16384,
  COMPRESSION 'zstd',
  COMPRESSION_LEVEL 1,
  WRITE_BLOOM_FILTER true
);
```

The earlier draft proposed pyarrow + k-way merge to avoid DuckDB's ORDER BY spilling. With the leveled scheme, inputs to any one job are bounded by `level_targets[n]`: at most ~256 MiB compressed (~1 GiB uncompressed). DuckDB's default 4 GiB memory_limit comfortably fits the sort; the multi-GB temp files we observed historically came from the flat policy where input was many GB. DDB stays. `union_by_name=true` continues to handle additive schema evolution as today.

Row-group sizing stays count-based at `_ROW_GROUP_SIZE = 16_384`. The earlier "switch to byte-based" was unnecessary churn.

### 4. Filenames: single scheme

Every segment is `seg_L<n>_<min_seq:019d>.parquet`. Level lives in both the filename (so disk-only boot recovery still works when the catalog is missing or stale) and the catalog `level` column (source of truth at runtime).

### 5. Catalog changes

**Migration `0003_segment_level.py`.**

```sql
ALTER TABLE segments ADD COLUMN IF NOT EXISTS level INTEGER NOT NULL DEFAULT 0;
UPDATE segments SET level = 0 WHERE state = 'tmp';
UPDATE segments SET level = 1 WHERE state = 'finalized';
CREATE INDEX IF NOT EXISTS segments_ns_level_minseq
  ON segments (namespace, level, min_seq);
```

The migration also renames on-disk `tmp_*` → `seg_L0_*` and `logs_*` → `seg_L1_*` inside a single transaction with the catalog `path` rewrites, then drops the now-unused `state` column. After this migration boots successfully there are no sentinels, no half-states — boot looks exactly like a fresh deploy with the new naming.

`SegmentState` is removed. The TMP/FINALIZED distinction was only used for visibility (now: every catalog row is visible because flush writes the row in lockstep with the file rename) and eviction (now: ordered by level, §8).

**Atomic swap is single-output.** `replace_segments(removed_paths, added)` already takes a sequence for `added`; with no splitting it's always called with a single-element list. The transaction is straightforward — no two-phase, no `level=-1` sentinel. A mid-rename crash leaves at most one `*.parquet.tmp` orphan, swept by a boot reaper.

### 6. Concurrency

**One bg thread per namespace** (today's structure). That thread handles flush, compaction, and eviction sequentially. Across namespaces, threads run in parallel — but each namespace's thread only ever touches its own namespace's files and catalog rows.

This makes the compaction-vs-eviction race impossible by construction: within a namespace they're back-to-back on the same thread; across namespaces the work sets are disjoint. No cross-thread coordination, no extra locks during the COPY.

**Eviction becomes per-namespace.** Today's `_evict_globally` walks all namespaces with a global cap (`max_local_segments=1000`, `max_local_bytes=100 GiB`); any namespace's bg thread can call it, and the eviction can target any other namespace's segments. That structure produces the cross-namespace race we're avoiding. Replacement: each namespace gets its own cap (`max_segments_per_namespace`, `max_bytes_per_namespace`), each bg thread evicts only its own segments at the tail of `_compaction_step`. Globals retained as deprecated defaults for one release. For finelog's typical single-dominant-namespace workload (the `log` namespace), per-namespace caps with the same numeric defaults are a strict relaxation, not a tightening.

**Locking pattern across phases:**

| Phase | `_insertion_lock` | `_query_visibility_lock` |
|---|---|---|
| Plan (input snapshot) | held | — |
| COPY (read inputs, write staging file) | — | — |
| Commit (rename + replace_segments + unlink + deque update) | held | **write** |
| Eviction (per-namespace, tail of compaction_step) | held briefly per victim | **write** |

Append/flush stay on the same per-namespace bg thread, serialized with compaction and eviction by virtue of being on one thread. New tmp segments arriving during a compaction are simply not in that compaction's input set; they're picked up by the next planning tick.

### 7. Failure handling

- **Mid-write crash:** COPY output is staged as `*.parquet.tmp`. Boot reaper deletes any `.parquet.tmp` not referenced in the catalog.
- **Mid-rename crash:** single-output renames are atomic at the OS level; either the new file is in place + catalog rewritten, or it isn't. Catalog reconciliation at boot drops rows for missing files.
- **GCS upload failure:** owned by `_OffloadWorker`, out of scope.

### 8. Eviction interaction

**Per-namespace, FIFO-by-`min_seq`.** Each namespace's bg thread evicts only its own segments, in oldest-first order, until that namespace is back under its caps. No global pass, no cross-namespace coordination.

The eviction query (per namespace):

```sql
SELECT path, byte_size FROM segments
WHERE namespace = ?
  AND level >= 1
  AND offloaded_at_ms IS NOT NULL
ORDER BY min_seq ASC
LIMIT 1
```

Eligibility:
- **`level >= 1`**: L0 is local-only and transient (awaiting promotion). Evicting it would lose data.
- **`offloaded_at_ms IS NOT NULL`**: migration `0004_segment_offloaded_at.py` adds the column; offload worker stamps it on upload success. Closes the current bug where a freshly-renamed L1 can be evicted in the window between rename and remote upload completion.

Because levels are time-ordered (L_max contains the oldest seqs), FIFO-by-seq naturally evicts oldest-largest first. No "prefer larger levels" heuristic.

**Caps move per-namespace.** New `CompactionConfig` fields (or a sibling `LocalStorageConfig`):

```python
max_segments_per_namespace: int = 1000
max_bytes_per_namespace: int = 100 * 1024**3
```

For a single-dominant-namespace deployment (today's marin `log` namespace) these defaults are a strict relaxation of the global caps, not a tightening. Multi-namespace deployments may want to tune.

Pinned tests:
- FIFO across mixed levels: stack L1 + L2, force eviction, assert popped `min_seq` is strictly ascending and never picks from the middle.
- L0 is never popped.
- Segments with `offloaded_at_ms IS NULL` are skipped until the offload completes.
- A long-running compaction in namespace A does *not* race eviction in namespace B (smoke-test by running the two on separate threads simultaneously and confirming neither breaks; expected behavior is they don't share files anyway).

### 9. Catalog interactions (end-to-end)

Every catalog touch the compactor performs:

1. **Plan input read.** `Compactor.plan` walks the in-memory `_local_segments` deque (which is kept in lockstep with the catalog under `_insertion_lock`). No SQL — the deque is authoritative at runtime.

2. **No catalog mutation during the COPY.** The merge SQL only reads parquet input files and writes the staging output; it never opens the catalog DB. The COPY runs lock-free. Concurrent eviction is impossible because eviction runs on the same per-namespace bg thread as the compaction that staged the COPY (§6).

3. **Atomic swap on commit.**
   ```python
   with self._query_visibility_lock.write():
       os.rename(staging_path, final_path)
       with self._insertion_lock:
           # Single transaction: DELETE inputs + INSERT output.
           self._catalog.replace_segments(
               namespace=self.name,
               removed_paths=[seg.path for seg in job.inputs],
               added=[output_row],
           )
           # Mirror the swap into the in-memory deque.
           self._local_segments = _splice(self._local_segments, job.inputs, output_seg)
       for seg in job.inputs:
           Path(seg.path).unlink(missing_ok=True)
       self._submit_offload(self.name, output_seg.path)
   ```
   `replace_segments` is already a transactional many-to-one swap (`catalog.py:210`) — no API change needed.

4. **Output row construction.** The output `SegmentRow` aggregates from inputs:
   - `level` = `inputs[0].level + 1` (all inputs share a level, enforced by `plan`).
   - `min_seq` = `min(input.min_seq for input in inputs)`.
   - `max_seq` = `max(input.max_seq for input in inputs)`.
   - `row_count` = `sum(...)`, `byte_size` = `staging_path.stat().st_size` (post-merge actual).
   - `min_key_value` / `max_key_value` = `_aggregate_key_bounds(inputs)` (already exists; landed in commit `ebe3f978b`).
   - `offloaded_at_ms` = `NULL` until the offload worker stamps it.

5. **Single-segment level bump.** If `len(inputs) == 1` and the lone input already meets `level_targets[n]`, the Compactor skips the COPY entirely. The namespace does:
   ```python
   new_path = _seg_filename(level=old_level + 1, min_seq=input.min_seq)
   os.rename(input.path, new_path)
   self._catalog.upsert_segment(replace(input_row, path=new_path, level=old_level + 1))
   ```
   No data rewrite, just a path + level bump. Same lock discipline.

6. **Boot reconciliation.** `reconcile_segments` (already exists) reads `seg_L<n>_*.parquet` from disk, re-derives `level` from the filename and `min_key_value` / `max_key_value` from the parquet footer, and rewrites the catalog rows to match disk. No special handling for in-progress compactions: any `*.parquet.tmp` orphan is unlinked.

7. **Eviction interaction.** Per-namespace eviction reads the catalog filtered to its own namespace, deletes rows + unlinks files, and pops the deque under `_insertion_lock`. All on the same per-namespace bg thread as compaction, so they're sequenced — no race possible.

The story for an external observer: at every commit boundary the catalog mirrors the on-disk parquet set exactly. Crash anywhere mid-compaction leaves at most one staging `.parquet.tmp` orphan; the next boot's reconciliation converges.

### 10. Migration plan

1. **Phase A — refactor only.** Extract `Compactor` with `level_targets = (∞,)` (single terminal level, no promotions beyond the equivalent of today's flat compaction) and `min_inputs_to_compact` matching the current threshold. Land migrations 0003 + the filename rewrite. Feature-equivalent to today; soak ~1 week.
2. **Phase B — tiered policy.** Flip the default to `level_targets = (64 MiB, 256 MiB)`. Land migration 0004 (`offloaded_at_ms`) in the same release so eviction stays safe.

Phase A is purely a refactor — small diff, easy to roll back. Phase B is the behavior change against a known-good baseline.

## Test plan

- Parametrize existing `test_persistence.py::test_compaction_*` over `CompactionConfig` fixtures (flat + tiered).
- New `test_compactor.py`: pure unit tests over `plan`. Cover the size-threshold case, the `max_l0_age_sec` case, and the single-segment-already-large case (level bump only).
- **Eviction FIFO test:** stack a namespace with mixed L1+L2 segments; force eviction repeatedly; assert popped `min_seq` is strictly ascending and never picks from the middle. Also assert L0 is never popped, and segments with `offloaded_at_ms IS NULL` are skipped until the offload completes.
- Crash injection: kill between rename and `replace_segments` commit; verify boot recovery converges.
- Eviction: L1 with `offloaded_at_ms IS NULL` is skipped; L2 with a value evicts first.
- Observability: bg-loop heartbeat logs per-level counts + bytes.

## Risks & open questions

- **Q1.** `level_targets` lives on the store-wide `CompactionConfig` (one set of values for all namespaces). Revisit if a namespace lands with a wildly different volume profile.
- **Q2.** When the L_max tier fills (steady state ≈ `max_local_bytes / level_targets[-1]` segments), eviction starts deleting them. We should add a `rows_local / rows_offloaded_only` gauge so on-call can see how much of the namespace is "remote only" without paging GCS.
- **Q3.** Phase A's `(∞,)` single-level mode reuses the new code path with the old policy. Verify the existing tests still pass under that config before we even ship Phase A — that's the rollback gate.

## References

- Current compaction: `lib/finelog/src/finelog/store/log_namespace.py:826` (`_compaction_step`)
- Catalog: `lib/finelog/src/finelog/store/catalog.py`
- Migrations: `lib/finelog/src/finelog/store/migrations/0001_init.py`, `0002_segment_key_value_bounds.py`
- Eviction: `lib/finelog/src/finelog/store/duckdb_store.py` (`_evict_globally`)
- Lock-during-COPY fix (precondition): commit `0acec62c0`
- Prior art: ClickHouse MergeTree (https://clickhouse.com/docs/en/development/architecture#merge-tree),
  Iceberg compaction (https://iceberg.apache.org/docs/latest/maintenance/#compact-data-files),
  RocksDB tiered compaction (https://github.com/facebook/rocksdb/wiki/Universal-Compaction).
