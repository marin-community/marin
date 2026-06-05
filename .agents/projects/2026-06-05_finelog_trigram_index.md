# Finelog trigram (substring) index — Part 1 of #6195

Status: design + measured prototype. Target branch `weaver/finelog-trigram-index`
(stacks on the Rust rewrite #6135; rebase onto `main` once #6135 lands).

## Problem

`contains(data, needle)` / `LIKE '%needle%'` over the log `data` blob is opaque to
the query engine. A prod FetchLogs

```sql
WHERE key = '/system/controller' AND contains(data, 'Bootstrap completed for TPU')
ORDER BY seq DESC LIMIT 8000
```

took ~8.6s. The `key =` predicate prunes to the controller key-band via parquet
min/max key stats, but `contains()` then forces a decode of `data` for every row
in that band. `data` is ~29.6 GB uncompressed across the namespace; bloom filters
prune 0 row groups for substring (they key on whole values), and a row-group-size
sweep showed no win. Only an index turns substring from `O(rows scanned)` into
`O(rows matched)`.

**Approach (per the issue):** a per-row-group **trigram (3-gram) presence**
structure. Decompose the needle into its 3-grams; a row group can be **skipped
unless it contains ALL of them** — conservative, never a false negative. Needles
< 3 chars fall back to scan. This plugs into DataFusion row-group pruning so the
decoder never touches skipped groups.

## The decision the user asked for first: ride-along vs sidecar

> Can the index ride along **inside** the parquet file (per-row-group trigram
> structure in the footer key-value metadata, arrow/parquet 58) instead of a
> separate sidecar file?

**Recommendation: sidecar.** Not because ride-along is conceptually wrong — its
atomic lifecycle is genuinely attractive — but because the only *expressible*
ride-along (footer KV) taxes the hot path, and the only *cheap* ride-along
(lazy, body-resident) is not expressible with the arrow-rs writer. Measured on
the real prod slice (`/home/power/finelog-bench/data/log`):

### The footer is read fully on every open — and it is already large

A representative L3 segment: **280 MB on disk, 27.5 M rows, 1679 row groups,
data = 4.2 GB uncompressed**. Its parquet footer (the thrift `FileMetaData`,
read in full on every file open because it holds the per-row-group, per-column
statistics) is **already 2.35 MB**.

A per-row-group trigram structure is, unavoidably, the same order of magnitude
as the existing per-row-group statistics. Measured distinct trigrams per row
group (60-rg sample): **median 4639, max 11231**. Encoded as a per-rg Bloom
filter, the index for *one* L3 segment is:

| encoding              | per-rg (median / max) | per-segment total |
|-----------------------|-----------------------|-------------------|
| Bloom @ 1% FPR        | 5.5 KB / 13.5 KB      | **9.3 MB** (max 22.6) |
| Bloom @ 10% FPR       | 2.8 KB / 6.7 KB       | 4.7 MB            |
| exact trigram set (3B)| 13.6 KB               | 23.4 MB           |

Embedding that in the footer KV grows it from 2.35 MB to **~11.6 MB (1% FPR)** —
a ~5× footer. The footer is fetched and parsed on every open of the file,
**including the dominant `key = … ORDER BY seq DESC LIMIT n` tail that never
calls `contains()`**. On the GCS-backed read path that is ~9 MB of extra ranged
read per segment per query for an index most queries never consult. Shrinking
the structure to "small enough to ride along" (a few hundred bytes/rg to keep
the footer under ~20% growth) drives the Bloom FPR to ~1.0 — i.e. it stops
pruning. *Small enough to ride along = useless.*

### "Lazy ride-along" collapses into a sidecar

The way to keep the index in-file *without* footer cost is to write it as a blob
in the parquet **body**, referenced by a tiny offset+length pointer in the footer
KV, and read it lazily only for `contains()` queries. This is not expressible
with arrow-rs's public writer:

- `ArrowWriter` exposes no hook to inject offset-referenced bytes into the data
  region between row groups; it writes row groups, then the footer.
- You cannot append the index *after* the trailing `PAR1` magic: standard
  readers locate the footer from the last 8 bytes, so trailing bytes break every
  reader. The **frozen read contract** (pyarrow reference reader + the Rust
  reader must keep opening these files unchanged) forbids that.

So a lazily-loaded in-file index is, in practice, *a second file* — which is a
sidecar with extra steps. Given that, the sidecar wins on every axis that
matters here, and we pay the one cost ride-along was meant to avoid:
lifecycle management. That cost is bounded (below).

### Why the sidecar's lifecycle risk is acceptable

The classic argument *for* ride-along is "no sidecar to orphan or keep
consistent." Here that risk is structurally small because **the index is a pure,
optional, derivable function of the segment's `data` column**:

- **Optional / never wrong.** A missing, stale, or corrupt sidecar is never a
  correctness bug — the prune just degrades to "scan this segment" (the current
  behavior). There is no read-contract coupling: the parquet is authoritative;
  the sidecar only ever *removes* row groups that provably cannot match.
- **Orphan = dead weight, not corruption.** An orphaned `.tgm` is cleaned up
  lazily; it never produces a wrong answer.
- **Lifecycle piggybacks on the existing segment swap.** Segment create / rename
  / unlink / GCS-sync are already centralized (see Integration). The sidecar
  rides those same seams as `seg_…parquet.tgm` next to its parquet, so it is
  never a separate bookkeeping problem.

### Net

| criterion                         | footer ride-along | **sidecar** |
|-----------------------------------|-------------------|-------------|
| footer load cost (hot non-contains path) | **+~9 MB/open, every query** | unchanged |
| index load cost (contains path)   | in-footer (eager) | lazy, only when needed |
| frozen read contract              | KV is ignorable, but body-blob not expressible | untouched |
| pruning power retained            | only if footer bloats | full |
| lifecycle atomicity               | best                | derivable ⇒ bounded |

The hot path is the keyed tail without `contains()`. Sidecar keeps that path at
zero added cost; ride-along cannot. Recommendation stands: **sidecar**.

## Does trigram pruning actually work on repetitive log text?

The issue flags "index size vs selectivity for highly repetitive log text" as
the gating open question. Measured prune rate over a 300-rg sample of the same
L3 segment (an *exact* per-rg trigram set; "survive" = row groups that pass the
all-trigrams-present test and must still be scanned):

| needle                          | prune % | precision % |
|---------------------------------|---------|-------------|
| `Bootstrap completed for TPU` (motivating) | **98.7** | 100 |
| `v6e-4`                         | 92.0    | 100 |
| `FDs open`                      | 91.3    | 100 |
| `Progress on:eval`              | 67.7    | 100 |
| `Lifecycle Leak!` (error spam)  | 58.7    | 100 |
| `disconnected unexpectedly` (absent) | 96.7 | 0 (no false negatives) |

The longer the needle, the more 3-grams must co-occur, the better the prune —
the motivating 27-char needle prunes **98.7%** of row groups. The worst case
(short, common substrings) still prunes ~59%. The absent-needle row shows the
conservative contract: ~3% of row groups survive on trigram co-occurrence but
**zero** false negatives. Feature is justified.

## Sidecar format (v1)

`seg_L{level}_{min_seq}.parquet.tgm`, sitting next to its parquet:

```
magic "FLTG" | version u8 | column id u8 | row_group_count u32
per row group: { m_bits u32, k u8, byte_len u32, bloom bytes }   # split-block / double-hash bloom
```

- One **per-row-group Bloom filter** over the byte-trigrams of `data`, sized per
  row group from its distinct-trigram count to hit a target FPR (default 1%).
  Bloom (not the exact set) keeps the sidecar ~9 MB for a 280 MB segment (~3%)
  and gives O(1) per-trigram probe; the ~1% extra survivors are immaterial next
  to a 59–99% prune.
- Row groups are **16384 rows** (`ROW_GROUP_SIZE`), and `ArrowWriter` splits
  strictly on row count with no byte cap, so per-rg index boundaries align
  deterministically with parquet row groups by global row index. A unit test
  asserts `index.row_group_count == parquet.num_row_groups`.
- Trigrams are raw 3-byte windows over UTF-8 `data` (byte-level, locale-free).
  The needle is tokenized identically. Indexed columns: `data` only for v1
  (`key` is already range-prunable; revisit if a `contains(key, …)` workload
  appears).

## Integration (the Part 2 pruning hook, scoped to Part 1)

DataFusion 53.1.0 gives the exact seam (the "advanced parquet index" pattern):
the parquet opener reads `PartitionedFile.extensions`, downcasts to a
`ParquetAccessPlan`, and honors per-row-group `RowGroupAccess::{Scan,Skip}`
(`datafusion-datasource-parquet-53.1.0/src/opener.rs:927`,
`access_plan.rs:90`). Absent an access plan it scans all row groups.

`NamespaceProvider::scan` (`query/provider.rs`):

1. If the pushed-down `filters` contain no `contains(indexed_col, needle≥3)`
   term, **delegate to `ListingTable::scan` exactly as today** — hot path
   untouched, zero new cost.
2. Otherwise, build a parquet scan by hand: for each snapshotted segment, load
   its `.tgm`, decompose each needle into trigrams, compute the row-group mask
   (skip rg unless every trigram probes present), and attach
   `PartitionedFile::with_extensions(Arc::new(access_plan))`. The `contains()`
   filter stays pushed down (`supports_filters_pushdown` remains `Inexact`) so
   the decoder still verifies survivors exactly — the index only ever *removes*
   provably-empty row groups. A missing/short-needle case yields `new_all`
   (scan everything) — identical results, just unpruned.

Write path:

- **L0 seal** (`store/segment.rs::write_segment*`) and **compaction output**
  (`store/compaction/executor.rs::write_merged_segment`) build the sidecar from
  the same in-order `data` values they write, chunked at `ROW_GROUP_SIZE`.
- **Lifecycle** (mapped to existing seams): rename the `.tgm` with its parquet on
  level-bump and unlink it with merged inputs in `namespace.rs::commit_swap`;
  unlink with `evict_segment`; upload/delete it alongside the parquet in
  `remote.rs::{upload,delete}` and `reconcile.rs`. `discover_segments` is
  unchanged (returns parquet paths); a missing sidecar is detected at adopt and
  simply disables pruning for that segment.

## Plan

1. `store/trigram.rs` — tokenizer, per-rg Bloom index, build-from-batches,
   serialize/deserialize, and `needle → row-group mask` against parquet metadata.
2. Wire the build into L0 seal + compaction output.
3. Wire the lifecycle (rename/unlink/sync) onto the existing segment seams.
4. Wire the prune into `NamespaceProvider::scan` behind the `contains()` check.
5. Tests: tokenize + serde round-trip; no-false-negative mask; rg-count
   alignment; end-to-end provider prune with correct rows + skipped row groups;
   compaction sidecar follow.
6. Benchmark: extend `lib/finelog/scripts/bench_dashboard_queries.py` with the
   controller `contains()` FetchLogs shape; report before/after p50/p95 on the
   real slice.

## Explicitly out of scope (Part 2)

The optimizer-rule relocation of `prefix`/`regexp`/`LIKE`→range rewrites, and
segment-level seq/epoch_ms pruning, are Part 2. This PR is Part 1 only: the
trigram index + its row-group pruning hook. The frozen protobuf contract is not
touched.
