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

### Alternative considered: a trigram *column* scanned before the body

The natural first question: rather than a sidecar, store the trigrams as an
extra **column** in the parquet, scan it first, then decode `data` only for the
rows that pass. The "scan a cheap column, then late-materialize the body" part is
real and we *do* use it (DataFusion `pushdown_filters`). The problem is what the
column would hold:

- **A parquet column is per-row; the presence we need is per-row-group.** To
  answer "does this row contain all the needle's trigrams" with no false
  negatives, a per-row column must store that row's whole trigram set (a list, or
  a per-row Bloom). Sized on the real data, a per-row Bloom is ~120 B/row →
  **~3.3 GB per L3 segment** vs `data`'s 4.2 GB — roughly doubling storage; shrink
  it and the FPR climbs to ~1.0 (every row says "maybe"). No win.
- **The win is skipping whole row groups, which a column can't do.** Even with
  late materialization, the engine reads the index column for *every* row in the
  band — O(rows scanned). Our measured win (98.7% of row groups never opened)
  comes from answering "can this 16 384-row block match at all?" *before* touching
  it, which is inherently per-row-group. Parquet has no per-row-group user column.
- **The ideal form is parquet's own per-row-group Bloom** — body-resident,
  offset-referenced, read lazily on probe — but it indexes whole values for
  *equality*, not substrings (hence "bloom filters prune 0 row groups" in the
  issue), and arrow-rs exposes no hook to write a custom trigram-seeded one.

So the sidecar *is* the "trigram index scanned first," realized at the row-group
granularity that makes it sub-linear and stored in the one place per-row-group
data can live.

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

## Measured end-to-end on the real prod slice

Built the sidecars and ran `SELECT count(*) … WHERE contains(data, needle)` over
the full pulled `log` namespace (54 segments, 12 319 row groups, 2.32 GB
parquet) once through a plain `ListingTable` (today's path) and once through the
trigram-pruned `NamespaceProvider`. Results match by construction; the figures
are wall-clock at the engine (`examples/bench_trigram.rs`):

| needle                          | matches | unpruned | pruned | speedup |
|---------------------------------|---------|----------|--------|---------|
| `Bootstrap completed for TPU` (motivating) | 3 124 | 10 589 ms | **633 ms** | **16.7×** |
| `Lifecycle Leak!` (50.9 M matches) | 50 940 663 | 10 341 ms | 2 765 ms | 3.7× |
| `disconnected unexpectedly` (absent) | 0 | 10 731 ms | **321 ms** | **33.4×** |

The motivating shape drops from ~10.6 s to **0.63 s**; an absent needle (the
common "search for a string that isn't there" dashboard case) is **33×**; even a
pathological substring matching 50.9 M rows is 3.7× (the index still skips the
row groups that lack it). **Index size: 78.7 MB = 3.39 % of the parquet** — the
~3 % the Bloom sizing predicted. Build cost: 324 s to index all 54 segments
cold; in production this is paid incrementally per merge in the background
compactor, not on the read or write-ack path.

## Sidecar format (v1)

`seg_L{level}_{min_seq}.parquet.tgm`, sitting next to its parquet:

```
magic "FLTG" | version u8 | column_name_len u8 | column_name | row_group_count u32
per row group: { k u8, m_words u32, m_words × u64 bloom words }   # double-hash bloom
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

`NamespaceProvider::scan` (`query/provider.rs`) uses **delegate-then-inject** —
not a hand-built scan — so it preserves every existing prune:

1. Delegate to `ListingTable::scan` exactly as today. ListingTable sets the
   parquet `predicate`, which is what drives the existing range / min-max /
   bloom row-group pruning (e.g. the `key =` controller-band prune the
   motivating query relies on). Building a scan from scratch would have *lost*
   that.
2. A cheap, no-I/O check: if no filter is a top-level
   `contains(INDEXED_COLUMN, <literal>)`, return the delegated plan untouched —
   the hot path pays nothing.
3. Otherwise, off the async worker (`spawn_blocking`), rewrite each
   `PartitionedFile` to carry a `ParquetAccessPlan` extension that *skips* the
   row groups the sidecar rules out. The opener composes our skips with its own
   range/stats/bloom pruning (`opener.rs:495`). The `contains()` filter stays
   `Inexact`, so a `FilterExec` re-checks survivors exactly — the index only ever
   *removes* provably-empty row groups; a missing/short-needle/misaligned case
   simply doesn't inject (scan everything, identical results).

Only top-level conjuncts are pruned on: a `contains()` under `OR` could drop rows
matching the other branch, so those are ignored.

Write path:

- **Build at the compaction merge output** (`compaction/executor.rs::apply_merge`)
  from the in-order merged `data` values. `kway_merge` already emits exactly
  16384-row chunks, so the index aligns 1:1 with the parquet row groups. L0→L1 is
  always a merge (small L0 flushes never hit the L1 byte target alone), so every
  queryable L1+ segment gets indexed. **L0 is intentionally unindexed** (small,
  short-lived) to keep the write-ack path fast.
- **Lifecycle** (existing seams in `namespace.rs`): carry the `.tgm` with its
  parquet on the single-input **level-bump rename** in `commit_swap` (a bump is a
  pure rename, no rewrite, so the index stays valid); **unlink** it with merged
  inputs (`commit_swap`) and on **eviction** (`evict_segment`).
- **Sidecars are local-only in v1.** They survive restarts (local files), are
  rebuilt by compaction, and self-heal after a full disk-loss recovery. This
  keeps the data-safety-critical GCS `sync_step` (with its phase-ordering
  durability invariant) untouched. Because the prod query path reads the
  compacting VM's *local* segments, the live dashboard win lands without remote
  sync. Syncing sidecars to GCS (for cross-VM / post-recovery pruning) is a safe
  follow-up — the index is optional, so a remote-restored segment without one is
  merely unpruned.

## Status (implemented in this PR)

1. ✅ `store/trigram.rs` — tokenizer, per-rg Bloom index, build-from-batches,
   serde, and `keep_mask(needle)`.
2. ✅ Build at the compaction merge output; carry on bump; unlink with parquet.
3. ✅ Prune injected in `NamespaceProvider::scan` behind the cheap `contains()`
   check, off the async worker.
4. ✅ Tests: serde round-trip; no-false-negative mask; rg-count alignment;
   end-to-end provider prune (correct rows + row group 0 skipped, hot path
   untouched); merge writes a working sidecar.
5. ✅ Benchmark: `examples/bench_trigram.rs` on the real prod slice (numbers
   above).

## Explicitly out of scope (Part 2 / follow-ups)

The optimizer-rule relocation of `prefix`/`regexp`/`LIKE`→range rewrites and
segment-level seq/epoch_ms pruning are Part 2. Remote (GCS) sidecar sync, and
indexing `key`, are follow-ups. This PR is Part 1 only: the trigram index + its
row-group pruning hook. The frozen protobuf contract is not touched.
