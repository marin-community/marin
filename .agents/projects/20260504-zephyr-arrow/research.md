# Research: Zephyr Arrow / Polars Migration

Companion to [design.md](design.md). Sources: PyArrow ≥19 (Arrow C++ ≥19) and
Polars ≥1.0 documentation as of mid-2026; Marin codebase as of late April 2026
(commit `c2dbcb6e`).

---

## 1. Where the Python time goes today

`writers.py`, `shuffle.py`, `plan.py`, `external_sort.py`, and `spill.py` all
operate on **`Iterator[dict]`**. Specifically:

- **`load_parquet`** ([readers.py:303](lib/zephyr/src/zephyr/readers.py:303))
  reads PyArrow row groups, then immediately calls `table.to_pylist()` to hand
  the worker stream a `dict` per row. Same for `load_vortex`. JSONL goes through
  `msgspec.json.Decoder().decode(line)` directly to dicts.
- **Map/Filter/FlatMap** ([plan.py:163-177](lib/zephyr/src/zephyr/plan.py:163))
  are pure-Python generators iterating one dict at a time.
- **`ScatterWriter`** ([shuffle.py:477](lib/zephyr/src/zephyr/shuffle.py:477))
  buffers per-target lists of Python dicts, sorts via `buf.sort(key=...)`, and
  serialises sub-batches with `cloudpickle.dump` inside a zstd frame.
- **Reduce** ([plan.py:602](lib/zephyr/src/zephyr/plan.py:602)) calls
  `heapq.merge` over chunk iterators of dicts, then `itertools.groupby`, and
  passes a Python iterator to `reducer_fn(key, items_iter)`.
- **Spill** ([spill.py:67](lib/zephyr/src/zephyr/spill.py:67)) wraps each item
  in `pickle.dumps(...)` and stores opaque bytes in a single `binary` Parquet
  column. Read path unpickles each row.
- **Sorted-merge join** ([plan.py:674](lib/zephyr/src/zephyr/plan.py:674)) tags
  every left/right item, `heapq.merge` by key, `groupby`, and calls a Python
  `combiner_fn(left, right)` per matched pair.

Every cross-stage boundary therefore round-trips through:

1. Arrow array → `to_pylist()` (allocates ~K dict objects, ~N string objects)
2. Python iterator transforms (refcount churn, GC pressure)
3. `cloudpickle.dump` into zstd (re-allocates everything as opaque bytes)
4. `cloudpickle.load` on the reducer side (re-allocates dicts)

The 50%–70% GC time in profiling corroborates step 1 dominating: PyArrow's
`to_pylist` materialises every cell as a fresh Python object. CPython's GC
walks every newly-allocated container. For a 1M-row scatter shard with 10
fields, that's 10M+ string objects per stage boundary.

The Arrow plan is to keep data as `pa.RecordBatch` / `pa.Table` from
`load_parquet` through scatter through reduce, only materialising Python
objects when (a) user code explicitly asks for items, or (b) the writer needs
JSONL/dict output.

---

## 2. PyArrow APIs the migration depends on

### IPC (the on-disk shuffle format)

PyArrow has two IPC formats. The choice matters for the scatter-file design:

| Format | Random access | Footer | Use |
|---|---|---|---|
| **Stream** (`pa.ipc.new_stream` / `open_stream`) | No — sequential read only | None; ends with EOS marker | Network/pipe transport |
| **File** (`pa.ipc.new_file` / `open_file`) | Yes — footer stores per-batch offsets | Yes — `RecordBatchFileReader` exposes `num_record_batches` and `get_batch(i)` | Random-access on disk |

The current `ScatterWriter` writes one binary file with **byte-range offsets in
a sidecar**. A direct port using IPC file format would let the IPC footer
itself carry per-batch offsets — but the design says the sidecar must still
identify *which target shard each batch belongs to*, so we still need a sidecar
mapping `target_shard → [batch_index]`.

**Compression**: Arrow IPC supports per-buffer compression via
`pa.ipc.IpcWriteOptions(compression='zstd', compression_level=N)`. This is
*not* whole-file zstd — it compresses each Arrow buffer (validity bitmap,
offsets, value buffer) independently inside the IPC frame. The file is a valid
IPC file that any Arrow consumer can read; the sidecar offsets remain valid
without an outer zstd wrapper. This is the cleanest match for the existing
"range-GET a chunk, decompress in memory, iterate" path.

Caveat: per-buffer compression is **less effective than whole-file zstd** on
small batches because each buffer compresses independently. For chunks ≪ 1 MB
the overhead can exceed savings. Need to benchmark; the current `ScatterWriter`
flushes by byte budget so chunk size is bounded by the cgroup memory fraction
(typically 64–256 MB) — that should compress well enough.

### Schema unification

`pa.unify_schemas([s1, s2])` widens compatible schemas (null → concrete,
missing fields). It **does not** unify `int` and `string` for the same field —
that raises `ArrowInvalid`. `_accumulate_tables`
([writers.py:164](lib/zephyr/src/zephyr/writers.py:164)) already handles this
for the writer side; reducers reading from N source-shard IPC files will
similarly need to unify schemas across files via `pa.concat_tables(...,
promote_options="permissive")`. Cost: O(num_source_shards) per reducer at
startup; cheap.

### Sort, group, and merge

PyArrow has:

- `pa.compute.sort_indices(table, sort_keys=[("col", "ascending")])` —
  in-memory single-table sort.
- `table.group_by(...).aggregate([(col, "agg_func")])` — vectorised aggregations
  (sum, count, count_distinct, hash_min, hash_max, list, ...). **Cannot run a
  Python reducer per group.**
- **No public k-way streaming sorted merge.** Acero has internal
  `SortMergeJoin` and `OrderByNode` but they're not exposed as a Python "merge
  N already-sorted record-batch streams" primitive. The closest exposed API is
  `Acero.declarations` for building plans, which is heavyweight and undocumented
  for this case.

This is the **single biggest gap**. The current `_merge_sorted_chunks`
([plan.py:602](lib/zephyr/src/zephyr/plan.py:602)) is a `heapq.merge` of
arbitrarily many chunk iterators. To stay in Arrow we need either:

a) Implement a custom Arrow-native k-way merge (RecordBatch peek + emit
   row-slices), keeping per-iterator memory bounded the same way. This is
   ~200–400 lines of new code, but it's the only way to preserve the current
   memory-bounded streaming semantics.

b) Concat all chunks for one target shard into a single table and sort. Loses
   the pre-sorted property (chunks are already individually sorted), and
   memory blows up for large shards — defeats the whole point of the
   external-sort fall-through.

c) Read each chunk into Python lists, use existing `heapq.merge`. No win.

### Hash and routing

`deterministic_hash` ([plan.py:559](lib/zephyr/src/zephyr/plan.py:559)) uses
`msgspec.msgpack.encode(obj, order='deterministic')` then xxh3_64. To stay
vectorised in Arrow we'd want column-wise hashing:

- `pa.compute.hash` exists but is **not stable across Arrow versions** (Arrow
  docs explicitly say so) and uses a different algorithm than xxh3.
- `xxhash` Python bindings can hash bytes; we'd need to materialise each row's
  key as bytes (e.g. via `pyarrow.compute.binary_join_element_wise`) which is
  awkward and expensive for nested keys.
- **Polars** exposes `Series.hash(seed)` using xxhash internally — but the seed
  semantics differ from our raw `xxh3_64_intdigest`, so cross-compatibility
  with the existing wire format requires care.

Practical answer: for the Arrow path, document that `key_fn` must be a
column-expression (e.g. `pl.col("user_id")` or a list of column names), and
compute the hash via Polars' `Series.hash()`. This **changes the partition
function from xxh3-of-msgpack to xxhash-of-polars-row** — not wire-compatible
with the existing scatter format. Acceptable if we're cutting a new format
version anyway, but worth calling out.

Per-row Python `key_fn` (the current API) requires materialising rows — same
GC cost we're trying to avoid. The migration only pays off when users opt into
column-based keys.

### Filter pushdown

Already integrated (`expr.py:to_pyarrow_expr`,
[readers.py:298](lib/zephyr/src/zephyr/readers.py:298)). No change needed; the
Arrow-native path inherits this for free.

---

## 3. Polars APIs and where they fit

Polars complements PyArrow rather than replacing it. Strengths relevant here:

- **`pl.DataFrame.partition_by(key, as_dict=True)`** — vectorised hash-partition
  into a `dict[key_value, DataFrame]`. Direct replacement for the per-target
  shard buffering loop in `ScatterWriter._flush`. Caveat: returns *materialised
  partitions* in memory; we still need our own byte-budget flushing.
- **`pl.LazyFrame.sort(...)` with streaming engine** — disk-spilling sort using
  Polars' streaming executor. Could replace `external_sort.py` for the
  single-shard case. Caveat: streaming engine in Polars 1.x is still labelled
  "experimental / breaking changes possible"; not all operations are streamable
  (notably user-defined functions break streaming). The blog post linked in
  the design ([Polars streaming joins](https://pola.rs/posts/streaming-joins/))
  describes the new merge-sort join, which **is a true streaming join** — but
  it requires both inputs as `LazyFrame`s with a common partition key. Mapping
  our `Shard → ScatterReader → join` flow into `LazyFrame.scan_ipc(...)` would
  work for the file-backed case.
- **`pl.from_arrow(table)` / `df.to_arrow()`** are zero-copy when the schema is
  Arrow-native (no Object dtype). Conversion overhead is negligible.
- **`pl.Series.hash(seed1, seed2, seed3)`** — xxhash-based vectorised hashing.
  See note above on cross-compatibility.
- **No equivalent of `reducer_fn(key, Iterator[items])`** — Polars
  `group_by().agg(...)` only takes vectorised expressions. User-defined
  per-group Python via `map_groups()` exists but materialises each group as a
  DataFrame and runs in Python — same GC cost we're trying to avoid.

**Where Polars genuinely helps**:

- Scatter routing (`partition_by` over a hashed key column).
- Streaming sort spill (replacement for `external_sort.py` pass-1).
- Streaming sorted-merge join (replacement for `_sorted_merge_join` if both
  sides are file-backed and the user combiner is expressible as a join).
- Vectorised aggregations when users opt into a Polars-expression API
  (replacement for `combiner_fn` in the common count/sum/agg case).

**Where Polars does not help**:

- Any user `key_fn` / `reducer_fn` / `combiner_fn` that's a Python lambda over
  per-item Python objects.
- Items that aren't representable in Arrow's typesystem (frozensets,
  arbitrary Python objects) — Polars' `Object` dtype exists but doesn't
  serialize to IPC.

---

## 4. Items not representable as Arrow

Test [test_shuffle.py:158](lib/zephyr/tests/test_shuffle.py:158)
(`test_scatter_handles_arbitrary_python_objects`) asserts that
`frozenset`, mixed `None`/`frozenset` in the same field, and other
non-Arrow-native types round-trip through scatter. The current code achieves
this via `cloudpickle.dump`. Arrow IPC has no equivalent — the schema must be
declared up-front and every batch must conform.

Options:

1. **Break the contract**. Document that scatter inputs must be Arrow-coercible
   dicts. Update the test. Requires auditing every `group_by` / `deduplicate`
   call site in the Marin tree. (Searched: most uses are dicts of strings/ints.
   Tokenization stages produce dicts of int arrays. The frozenset case appears
   only in tests. Likely acceptable but not zero-cost.)

2. **Fall back to a binary-payload column** for non-coercible items. Detect on
   first item; if unrepresentable, wrap as `{"_payload": cloudpickle.dumps(item)}`
   for that whole shard. Works but loses the perf win for any job that hits the
   fallback. Easier sell: keeps the test passing, no audit needed.

3. **Mixed schema per chunk**: write Arrow-native batches when possible, append
   pickle blobs as a separate batch when not. Rejected — too complex, too easy
   to corrupt.

The design doc currently says "Keep the API to Zephyr the same, so existing
jobs will continue to run." Option 2 is the only one consistent with that
literally. Option 1 is honest about the tradeoff.

---

## 5. The user-callable problem

Zephyr's API exposes:

- `MapOp.fn: T → R`
- `FilterOp.predicate: T → bool`
- `FlatMapOp.fn: T → Iterable[R]`
- `MapShardOp.fn: (Iterator[T], ShardInfo) → Iterator[R]`
- `GroupByOp.key_fn: T → Hashable`
- `GroupByOp.reducer_fn: (Hashable, Iterator[T]) → R | Iterator[R]`
- `GroupByOp.sort_fn: T → Hashable`
- `GroupByOp.combiner_fn: (Hashable, Iterator[T]) → Iterator[T]`
- `JoinOp.combiner_fn: (T | None, R | None) → object`

All are item-at-a-time Python callables. The migration's headline benefit —
"removing all the GC time" — only materialises if **the data never becomes
Python objects between scatter and reduce**. But the moment any of these
callables runs, we have to materialise rows back to dicts.

This means the migration's actual perf win depends on which stages have user
callables in them:

- **Pure-load → scatter → reduce → write** (e.g. dedup-by-id, count-by-key):
  data can stay in Arrow end-to-end. Real win.
- **Load → map(transform) → scatter → reduce → write** (e.g. tokenization,
  feature extraction): the map stage forces materialisation; scatter/reduce
  benefit unaffected, but the map cost dominates. Mid-sized win.
- **Load → group_by(key=lambda r: r['id'], reducer=lambda k, items: ...)**:
  reducer materialises per-group items as dicts. Scatter benefit only.

To hit the headline 2x–3x, we need to *also* expose Arrow-native variants:

- `MapTableOp(fn: pa.RecordBatch → pa.RecordBatch)` (or Polars equivalent)
- `GroupByOp(key=col_names, agg=pl.expr)` for vectorised aggregations
- A new way to declare combiners as Polars/Arrow expressions

These are additive — old API still works, just doesn't get the full speedup.
The design doc hints at this with "Expose an alternative Zephyr API that
allows for processing data directly in Arrow" but doesn't detail it.

---

## 6. K-way merge, in detail

The current scatter→reduce path produces **N×M sorted chunks** at a reducer
(N source shards × per-source flush count for this target shard). N×M routinely
hits the thousands; `external_sort_merge` exists precisely because
`heapq.merge` over thousands of open file iterators blows the memory budget.

For an Arrow-native rewrite:

**Pass-1 inside one reducer**:
- Each scatter chunk = one IPC RecordBatch (sorted by `merge_key`).
- Need to merge K sorted batches into one sorted stream of batches.
- Polars: would have to convert each batch to a `LazyFrame`, then
  `pl.concat([lf1, lf2, ...]).sort(merge_key)` — but this is a *full sort*, not
  a merge of pre-sorted inputs. Polars' streaming sort can spill, so it works,
  but at cost: re-sorting ignores the pre-sorted property and costs O(N log N)
  instead of O(N log K).
- PyArrow Acero exposes a `SortByKeyNode` and an internal `MergeNode`, but the
  Python bindings (`pyarrow.acero`) only document `OrderByNode` — same
  problem: full sort, not merge.
- Custom implementation: maintain a heap of `(current_row_key, batch_id,
  row_index)`, peek, emit slices. Doable but ~200 lines and trickier than the
  Python `heapq.merge` because we need to emit in batch-sized strides for
  efficiency.

**Best path**: Custom Arrow-native k-way merge that emits `RecordBatch`-sized
runs of monotonically increasing keys. Spell out the row-stride trick in the
implementation: pop the cheapest batch's first K rows where the K-th row's key
is still ≤ the next-cheapest batch's first key, emit them as a single
`batch.slice(0, K)`, advance, repeat. This amortises per-row dispatch across
batches when keys are unevenly distributed.

**Pass-2 (external sort fan-in)**: Same algorithm operates over spill-file
batch iterators. The current `SpillWriter`/`SpillReader` (Parquet binary
column) would be replaced by IPC files with the actual schema.

---

## 7. Sorted-merge join, in detail

`_sorted_merge_join` ([plan.py:674](lib/zephyr/src/zephyr/plan.py:674)) is a
streaming inner/left join over two pre-partitioned, pre-sorted streams, with a
**user-supplied `combiner_fn(left, right) → object`** called per matching pair
and a Cartesian fan-out within duplicate-key groups.

Polars 1.x added a streaming sort-merge join via the new streaming engine —
this matches the algorithmic shape exactly, and supports inner/left/full
joins. **But**:

- Polars' join produces a single combined row per match (column-merging,
  optionally with `suffix=`); it does not call a user combiner. The current
  `combiner_fn` API includes things like `lambda left, right: {**left,
  **right, "joined": True}` which have a vectorisable equivalent, but also
  arbitrary Python combiners that don't.
- For arbitrary-combiner cases, the current `_sorted_merge_join` semantics
  (Cartesian per-group + per-pair Python) is the only correct fallback. We'd
  use Polars' streaming join only when `combiner_fn` is `None` or a known
  vectorisable shape (dict-merge).

Practical recommendation: implement two paths, gated by combiner type. Add a
`Join.combiner_kind: Literal['merge_columns', 'callable']` field; the planner
chooses the Arrow-native path when it can.

---

## 8. Multithreading claim revisited

The design says "we can parallelize jobs using multithreading at the Zephyr
level, since the Python GIL will no longer be holding locks during processing."
This is **partially true**:

- PyArrow C++ kernels (sort, filter, take, hash, IPC read/write) release the
  GIL. Threading helps these.
- Polars operations release the GIL in its Rust core. Threading helps these.
- zstd encode/decode (via `python-zstandard`) releases the GIL.
- File I/O (fsspec/GCS) releases the GIL.

What does **not** release the GIL:
- Any user `key_fn` / `combiner_fn` / `reducer_fn` / `MapOp.fn` written in
  Python. These re-take the GIL.
- `pa.compute.cast`, `to_pylist`, `from_pylist` — the conversion boundary.

So the multithreading benefit follows the same rule as section 5: it
materialises only on stages that don't run user Python per row. Combined with
the per-stage worker types in
[20260430-zephyr-performance/design.md](../20260430-zephyr-performance/design.md),
the actual win comes from threading the *Arrow-native* stages while leaving
user-Python stages on subprocesses.

---

## 9. Spill format

`spill.py` stores `pickle.dumps(item)` as `binary` Parquet column values. The
Arrow migration suggests replacing this with native-schema IPC files, but:

- Spilled data passes through `_merge_sorted_chunks` then user `reducer_fn`.
- Mixed-type / non-Arrow items (section 4) hit spill the same way they hit
  scatter. Spill needs the same fallback strategy.
- Parquet → IPC for spill removes one Parquet write/read per spilled item but
  doesn't change the asymptotics. Switch is consistent with the rest of the
  migration but has small marginal impact.

---

## 10. Dependency footprint

- PyArrow ≥22 already in `lib/marin/pyproject.toml`. Iris uses ≥19. Zephyr's
  current minimum is whatever marin pulls in transitively. Worth pinning Arrow
  to ≥22 in Zephyr's `pyproject.toml` directly.
- Polars **is not currently a dependency** anywhere in the tree (verified with
  `grep -r polars lib/*/pyproject.toml`). Adding it pulls a substantial Rust
  binary (Polars wheel is ~30 MB). Worth confirming this is acceptable;
  Marin's worker images would gain ~30 MB.
- `python-zstandard` already in use.
- `cloudpickle` still required for the fallback path (and for shipping user
  closures to workers — separate concern, not on the data path).

---

## 11. Surprise areas worth a small benchmark before committing

These are the assumptions in the design that could turn out wrong, listed by
how cheaply they can be validated:

1. **Per-buffer zstd vs whole-file zstd** for chunk sizes typical of Marin
   shuffles (64 MB cgroup / 256 MB cgroup). Run `pa.ipc.write_table` with
   `compression='zstd'` on a real shuffle batch and compare bytes to the
   current `zstd.compress(cloudpickle.dumps(items))`. If per-buffer is >1.3x
   larger, consider keeping outer zstd wrapper.

2. **Custom k-way merge throughput** vs `heapq.merge` over `to_pylist()`'d
   batches. The custom implementation has to win to justify itself; if rows
   are tiny (e.g. `{"id": int}` for dedup) the Python overhead of heap
   maintenance dominates and the Arrow path may not win.

3. **`partition_by` cost** in the scatter writer. The current scatter does
   per-item dict-routing in Python — we'd replace it with a vectorised
   hash + partition. Should be a clear win, but `partition_by(as_dict=True)`
   on a 100k-row batch with 1024 target shards has not been benchmarked here.

4. **`pa.Table.concat_tables(promote_options='permissive')` cost across 1000s
   of source files** at reduce-side schema unification.

---

## 12. What stays the same

To prevent this list from looking like a teardown — most of the surrounding
machinery is fine and stays put:

- `ScatterReader` sidecar layout (target_shard → byte ranges) maps cleanly to
  IPC batch indices.
- The `from_sidecars` per-reducer parallel sidecar read.
- The `needs_external_sort` heuristic (just rewires `avg_item_bytes` from
  pickle bytes to Arrow bytes).
- `_compute_file_pushdown`, `LoadFileOp`, the existing reader paths.
- `_fuse_operations` and the planner's stage boundaries.
- The runner / coordinator / heartbeat machinery.

The actual surface area of code that *must* change is:

- `ScatterWriter` (~200 LOC) → IPC-based equivalent
- `ScatterFileIterator._iter_chunk` (~30 LOC) → IPC batch read
- `_merge_sorted_chunks` (~70 LOC) → custom k-way merge
- `_sorted_merge_join` (~50 LOC) → dispatch to Polars or fallback
- `spill.py` (~210 LOC) → IPC-based equivalent
- New `MapTableOp` / `GroupByExprOp` for the optional Arrow API

Everything else either compiles unchanged or adapts mechanically.
