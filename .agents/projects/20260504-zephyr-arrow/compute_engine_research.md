# PyArrow vs Polars for Zephyr

Research for the "PyArrow vs Polars" section of [design.md](design.md).
All benchmarks run on 2026-05-06 with PyArrow 22.0.0 and Polars 1.40.1,
in the `lib/zephyr` uv environment on an M-series Mac.

---

## 1. Efficiency

The two libraries share the same Arrow columnar memory format, so data layout
is identical. The performance difference comes from their compute engines:
PyArrow delegates to Arrow C++ (`pyarrow.compute`); Polars delegates to its
own Rust engine which is explicitly multithreaded for most operations.

### Benchmark results

| Operation | PyArrow | Polars | Notes |
|---|---|---|---|
| filter (5M rows, 50% selectivity) | 6.1ms | 1.1ms | ~5x |
| sort (5M rows, int64) | 391ms | 20ms | ~20x |
| group_by + sum (5M rows, 1K groups) | 24.5ms | 20.3ms | ~1.2x |
| str.to_uppercase (5M rows) | 121.9ms | 126.4ms | ~1x |
| inner join (1M × 500K, unique keys) | N/A (no API) | 3.7ms | — |
| Series.hash (1M strings) | N/A (no API) | 4.7ms | — |
| partition_by key (2M rows, 10K keys) | N/A (no API) | 18.9ms | — |

**Takeaways:**

- Sort is the clearest win — 20x faster under Polars. This matters for the
  scatter flush and for `_merge_sorted_chunks`, which are the hottest
  operations we're migrating.
- Filter is 5x faster, relevant for pre-scatter filtering.
- Group-by aggregations are roughly equivalent (both are vectorised C++/Rust,
  both use hash tables). The PyArrow version is sometimes faster for simple
  single-key aggregates on numeric types.
- String ops (`str.upper`, etc.) are effectively identical — the underlying
  SIMD kernels are the same Arrow C++.
- Several critical Zephyr operations (hash-for-routing, partition_by, sorted
  join) exist only in Polars.

---

## 2. API Simplicity

### Feature matrix

| Feature | PyArrow | Polars |
|---|---|---|
| Hash columns for routing | No built-in; must round-trip through Python + xxhash | `Series.hash()` — 1 line, vectorised |
| Partition rows by key (scatter) | No API; must sort then slice, or group then iterate | `partition_by(key)` — returns list of DataFrames |
| Sorted-merge join | No API; Acero hash join only | Streaming engine join (see caveats §3) |
| Per-buffer IPC zstd compression | `IpcWriteOptions(compression='zstd')` | Requires `compat_level` param |
| Sort (single table) | `pc.sort_indices` + `take` (two-step) | `df.sort(col)` (one-step, multithreaded) |
| Streaming disk-spilling sort | No | `LazyFrame.sort()` with streaming engine (experimental) |
| Vectorised string ops | `pc.utf8_upper(col)` — functional | `pl.col.str.to_uppercase()` — method chaining |
| Expression operator overloading | No — `pc.and_(pc.greater(a, b), ...)` | Yes — `(col_a > b) & ...` |
| LazyFrame / query optimization | Acero (low-level declarations only) | Full LazyFrame DSL with automatic push-down |
| Map Python fn over groups | No — `group_by().aggregate()` vectorised only | `map_groups(fn)` — passes DataFrame per group |
| Per-batch user code | Caller loops over `RecordBatch`es | Same; Polars `LazyFrame.map_batches()` |
| Zero-copy from Arrow (numerics) | Is Arrow — no conversion needed | Yes — same buffer |
| Zero-copy from Arrow (strings) | Is Arrow — no conversion | No — must copy to internal `StringView` |

### API verbosity example

Adding a constant column (the `file_idx` use case from `ArrowDataset`):

```python
# PyArrow
t_with_idx = t.append_column(
    'file_idx',
    pa.array([file_idx] * len(t), pa.int64())
)

# Polars
df_with_idx = df.with_columns(pl.lit(file_idx).cast(pl.Int64).alias('file_idx'))
```

Computing shard routing:

```python
# PyArrow — no vectorised path; must loop per row
shard_ids = [xxhash.xxh3_64_intdigest(msgspec.msgpack.encode(item['key'])) % num_shards
             for item in t.to_pylist()]  # GC cost returns

# Polars — vectorised
shard_ids = df['key'].hash() % num_shards  # returns Series in ~5ms for 1M rows
```

**Takeaway**: For the operations Zephyr needs most (scatter routing, sort,
partition, join), Polars provides a substantially simpler API. PyArrow's
`pyarrow.compute` is lower-level and designed for building other systems on
top of — appropriate for IPC read/write, less so for data transformations.

---

## 3. Compatibility Issues

This is the most important section. The two libraries technically operate on
the same in-memory Arrow format, but in practice **type conversions happen at
every boundary**, and some conversions are lossy or unsupported.

### The `string` → `large_string` (and `binary` → `large_binary`) promotion

**Direction: PyArrow → Polars → PyArrow**

Polars internally uses `StringView` and `BinaryView` for all string/binary
data. When you call `pl.from_arrow(table)` on a table with `string` columns
and then `df.to_arrow()`, you get back `large_string` (not `string`). This is
generally fine — `large_string` is a superset of `string` — but if any
downstream code asserts exact schema equality or uses `pa.concat_tables`
without `promote_options='permissive'`, it will fail:

```
concat string_view + string: ERROR Unable to merge: Field s has incompatible types: string_view vs string
```

**Direction: Polars IPC → PyArrow**

This is the critical one for Zephyr. When Polars writes an IPC stream (e.g.
`df.write_ipc_stream(buf)`), it emits **`string_view`** and
**`binary_view`** types by default (Arrow 1.3+ view types). PyArrow 22 can
*read* these types, but its compute kernels don't support them:

```
pc.utf8_length on string_view:  ERROR — no kernel matching input types (string_view)
pc.utf8_upper  on string_view:  ERROR — no kernel matching input types (string_view)
pc.equal       on string_view:  ERROR — no kernel matching input types (string_view)
pc.sort_indices on string_view: ERROR — Sorting not supported for type binary_view
```

`is_null`, `cast(string)`, and `cast(large_binary)` do work on view types.
But the operations Zephyr's scatter uses — equality, sorting — do not.

**The fix: `compat_level=pl.CompatLevel.oldest()`**

Polars provides a backwards-compatibility parameter for IPC writes:

```python
df.write_ipc_stream(buf, compat_level=pl.CompatLevel.oldest())
```

This emits `large_string` / `large_binary` instead of `string_view` /
`binary_view`. All PyArrow 22 compute functions work on `large_string`.
Benchmark shows writing with `oldest()` is actually *faster* than the default
(19ms vs 31ms for 2M rows), with ~5% smaller output (89MB vs 93MB).

**This must be a project-wide convention**: any Polars IPC write that will be
read by PyArrow must use `compat_level=pl.CompatLevel.oldest()`. Omitting it
silently produces a file that PyArrow reads but can't compute on — a latent
bug.

### Parquet cross-compatibility

| Direction | string type outcome | binary type outcome |
|---|---|---|
| PA writes, PL reads | `string` → `String` (ok) | `binary` → `Binary` (ok) |
| PL writes, PA reads | `large_string` | `large_binary` |
| PA writes list of strings | `list<string>` → `List(String)` | — |

Parquet cross-reads work without errors in all tested combinations. The type
upgrades (`string` → `large_string`) are lossless. The `_to_large_type`
helper already in `external_sort.py` handles the most important case.

### Roundtrip test results

```
string       PA=string      → PL=String   → PA=large_string  ⚠ type changed
large_string PA=large_string → PL=String  → PA=large_string  ✓ ok
binary       PA=binary      → PL=Binary   → PA=large_binary  ⚠ type changed
large_binary PA=large_binary → PL=Binary  → PA=large_binary  ✓ ok
int64        PA=int64       → PL=Int64    → PA=int64         ✓ ok
float32      PA=float32     → PL=Float32  → PA=float32       ✓ ok
list<int32>  PA=list<int32> → PL=List(Int32) → PA=large_list<int32>  ⚠ type changed
struct       PA=struct      → PL=Struct   → PA=struct        ✓ ok
dict<str>    PA=dict<str,int32> → PL=Categorical → PA=dict<large_string,uint32>  ⚠ type changed
timestamp_us PA=timestamp[us] → PL=Datetime → PA=timestamp[us]  ✓ ok
null         PA=null        → PL=Null     → PA=null          ✓ ok
```

**Practical rule**: After a `pl.from_arrow(t)` + compute + `df.to_arrow()`
roundtrip, all `string` and `list<T>` columns are promoted to their `large_*`
equivalents. Any schema comparison or `concat_tables` must use
`promote_options='permissive'` or `pa.unify_schemas` to tolerate this.

### Zero-copy boundaries

- **Numeric types** (int, float, bool, timestamp): fully zero-copy in both
  directions. `pl.from_arrow` and `df.to_arrow` share the underlying buffer.
- **Strings**: NOT zero-copy. Polars stores strings as `StringView` (with an
  inline-prefix optimisation); Arrow stores as `string` (offsets + data
  buffer). Converting requires a memory allocation. The benchmark shows a
  roundtrip of 10M strings takes ~73ms even though 10M int64s take 0.4ms.
  This matters for string-heavy payloads (tokenized text, URLs).

---

## 4. IPC Performance Summary

For 2M rows with int64, string, and binary columns:

| Format | Write | Read | Size |
|---|---|---|---|
| PA IPC uncompressed | 9ms | 3ms | 73MB |
| PA IPC zstd (per-buffer) | 27ms | 16ms | 14MB |
| PL IPC string_view (default) | 31ms | 19ms | 93MB |
| PL IPC oldest() / large_string | 19ms | 21ms | 89MB |

PyArrow's uncompressed IPC write is the fastest (no compression overhead),
and its per-buffer zstd achieves the best compression ratio. For the scatter
file format, **writing with PyArrow IPC + zstd is the right choice** — it's
what the current code does post-migration, and Polars doesn't add anything
here.

The Polars `write_ipc_stream` path is slower and produces larger files. Polars
is the right choice for *computation* (sort, partition, hash), not for the
file I/O layer.

---

## 5. Recommendation: Use Both, at Different Layers

The question isn't "Polars *or* PyArrow" — it's which layer each belongs to.

**Use PyArrow for:**
- IPC read and write (scatter files, spill files) — it's faster and the format
  is fully controlled.
- RecordBatch construction and schema management.
- The cloudpickle fallback column (`binary` payload for non-Arrow types).
- Compute operations where PyArrow suffices (filter, column select, cast).

**Use Polars for:**
- Sort — 20x faster, the biggest single win.
- `partition_by(key)` — replaces the per-item routing loop in `ScatterWriter`.
- `Series.hash()` — vectorised shard-key hashing.
- Join — when user combiner is expressible as column merging.
- `map_groups(fn)` — user-supplied per-group Python functions over DataFrames
  instead of per-row over dicts.

**The handoff**: `pl.from_arrow(batch)` going into a Polars operation,
`df.to_arrow()` coming back out. Cheap for numerics, one-time copy for
strings. This is the natural "seam" — keep it minimal and use PyArrow IPC
on both sides of it.

**The one required convention**: every Polars IPC write must pass
`compat_level=pl.CompatLevel.oldest()`. Without it, string columns come out
as `string_view` which PyArrow can parse but not compute on.

---

## 7. Memory Usage and Streaming

### DuckDB: production-ready spill-to-disk

DuckDB has a mature out-of-core execution model. By default it caps memory at 80% of physical RAM; all pipeline-breaking operators (sort, hash join build, hash aggregate, windowing) spill to disk when the limit is hit:

```python
conn = duckdb.connect()
conn.execute("SET memory_limit = '4GB'")
conn.execute("SET temp_directory = '/fast/ssd/tmp'")
```

The external sort uses a k-way merge (minimum I/O). External hash join uses radix partitioning (4–12 bits → 16–4096 on-disk partitions). Memory use is inspectable via `duckdb_memory()` and `duckdb_temporary_files()`.

**Published results on a 140 GB Parquet workload (32 GB machine):**

| Library | Peak memory |
|---|---|
| DuckDB | 1.3 GB |
| Polars streaming | 750 MB (where applicable) |
| Polars default (eager) | 17 GB |

DuckDB's spill is transparent — no code changes needed when a query exceeds the memory limit.

### Polars LazyFrame streaming: experimental and limited

Polars' `collect(streaming=True)` is unstable and has significant exclusions:

- **Sort is excluded.** `LazyFrame.sort()` falls back to the eager in-memory path regardless of streaming mode.
- **Join streaming requires pre-sorted data.** Streaming merge-join only works if both sides are already sorted by the join key.
- The streaming API is marked "unstable" and may change without notice.

In practice, Polars streaming cannot be relied on for Zephyr's core operations (scatter sort, reduce join) when datasets approach or exceed available RAM.

### PyArrow: no spill-to-disk

PyArrow has memory pools (allocation tracking) and memory mapping (I/O efficiency) but **no analytical spill-to-disk**. There is no mechanism for `pyarrow.compute` sort or join to overflow to disk. Callers must implement spill manually or delegate to a system built on Arrow (such as DuckDB).

### Implications for Zephyr

The scatter sort is Zephyr's most memory-intensive operation. For datasets that fit in RAM on a large machine — the current common case — Polars is the right choice: fastest sort, cleanest scatter API. If scatter files grow past available RAM, only DuckDB provides a production-ready spill path. PyArrow provides no fallback.

---

## 8. DuckDB

DuckDB is an embedded OLAP database with a vectorised query engine, full SQL support, and first-class Arrow integration. It is architecturally different from Polars and PyArrow: it operates through SQL (or a Python relational API) and runs as an embedded query engine rather than a function library.

### Benchmark results (estimated from public benchmarks)

The numbers below are extrapolated from DuckDB's public benchmark suite and DuckDB vs. Polars comparisons published in 2025–2026, **not measured locally on this machine**. The PyArrow and Polars numbers are from the local benchmarks in §1.

| Operation | DuckDB (est.) | Polars (measured) | PyArrow (measured) | Notes |
|---|---|---|---|---|
| sort (5M rows, int64) | ~20–40ms | 20ms | 391ms | Within 2× of Polars; both ~10–20× faster than PyArrow |
| filter (5M rows, 50% selectivity) | ~1–2ms | 1.1ms | 6.1ms | Roughly equal to Polars |
| group_by + sum (5M rows, 1K groups) | ~15–25ms | 20.3ms | 24.5ms | DuckDB fastest in H2O benchmark at scale |
| sort (larger-than-RAM) | ✅ k-way merge spill | ❌ falls back in-memory | ❌ no spill | DuckDB is the only reliable option |

At large scale (SF-100, 100 GB), DuckDB slightly outperforms Polars (19.7s vs 23.9s); at smaller scale (SF-10, 10 GB), Polars is slightly faster (3.9s vs 5.9s). For Zephyr-sized tables the performance difference is negligible — the bigger question is whether data fits in RAM.

### API for Zephyr operations

**In-memory partition by key (scatter routing):**

DuckDB's `COPY ... PARTITION_BY` is file-based — it writes Hive-partitioned files, not in-memory sub-tables. There is no `partition_by()` equivalent that returns a list of DataFrames in one pass. To get per-shard DataFrames, you need N separate queries, each scanning the whole table:

```python
# DuckDB — O(N × table_size) reads; no single-pass equivalent
for shard_id in range(num_shards):
    shard = conn.execute(
        f"SELECT * FROM t WHERE hash(key) % {num_shards} = {shard_id}"
    ).arrow()
```

Polars' `df.partition_by(key)` does a single pass and returns all partitions at once. For Zephyr's scatter, the DuckDB approach is substantially less efficient.

**Vectorised hash:**

Available via the `hashfuncs` community extension (`INSTALL hashfuncs; LOAD hashfuncs`), which provides `xxh3_64()` and `xxh3_128()`. Without the extension there is no `Series.hash()`-style column expression. The extension must be installed per environment.

**Sorted-merge join:**

DuckDB supports hash join (default), broadcast join, and `PhysicalPiecewiseMergeJoin` for range predicates and pre-sorted data. External hash join handles larger-than-RAM joins via radix partitioning. For Zephyr's reduce phase, DuckDB can execute a merge-join if both sides are pre-sorted by key.

**Arrow IPC read/write:**

DuckDB 1.4+ ships an Arrow IPC extension:

```sql
COPY t TO 'output.arrows' (FORMAT ARROW, COMPRESSION ZSTD);
SELECT * FROM read_arrow('input.arrows');
```

Write speed has not been benchmarked locally. Based on architecture (zero-copy buffer pass), uncompressed write should be comparable to PyArrow.

### Arrow type compatibility

DuckDB uses the C data interface for Arrow integration. Numeric types are zero-copy in both directions. For strings:

- Arrow `string` / `large_string` → DuckDB `VARCHAR` → Arrow `string`: ✓ no type promotion
- Arrow `string_view` → DuckDB: **fails** (cannot convert; same issue Polars IPC output causes without `compat_level`)
- DuckDB always emits standard Arrow `string` on `.arrow()` output, never `string_view` or `large_string`

The `large_string → string` normalisation is unusual (most libraries promote; DuckDB normalises). It is lossless but could cause schema mismatches if downstream code explicitly expects `large_string`.

| Roundtrip | DuckDB output type | Safe? |
|---|---|---|
| `string` → DuckDB → Arrow | `string` | ✓ no change |
| `large_string` → DuckDB → Arrow | `string` | ⚠️ downcast; fine if downstream accepts `string` |
| `binary` → DuckDB → Arrow | `binary` | ✓ |
| `int64` → DuckDB → Arrow | `int64` | ✓ zero-copy |
| `string_view` (Polars default) → DuckDB | — | ❌ fails |

**Convention**: when passing Polars IPC output to DuckDB, the same `compat_level=pl.CompatLevel.oldest()` requirement from §3 applies — without it DuckDB will fail to read the string columns.

### SQL vs DataFrame API

DuckDB's query interface is SQL, which is a trade-off for Zephyr's use case:

**Advantages:** Automatic query planning (predicate/projection pushdown, join reordering); readable multi-table joins; transparent parallelism across all CPU cores (Morsel-Driven Parallelism, ~6.5× speedup on 8 threads).

**Disadvantages:** No method chaining for iterative computation; `partition_by` scatter requires N queries or full table materialisation vs Polars' single-pass API; no `map_groups(fn)` equivalent for user-supplied per-group Python functions without UDFs.

### Smallpond

[smallpond](https://github.com/deepseek-ai/smallpond) is a lightweight distributed data processing framework by DeepSeek built on DuckDB and their 3FS file system. It adds `repartition(N, hash_by="column")` for distributed scatter, achieving 3.66 TB/min on a 75-node cluster (GraySort benchmark).

Smallpond is not a substitute for Polars/PyArrow — it is a higher-level distributed framework wrapping DuckDB. Its scatter primitive is essentially Zephyr's scatter phase implemented on DuckDB at cluster scale; the relevant insight is that DuckDB's hash-partitioning is proven for this workload pattern, but the single-machine in-process scatter API gap remains.

### Summary: when to consider DuckDB

| Scenario | Use Polars | Use DuckDB |
|---|---|---|
| Scatter fits in RAM | ✅ — `partition_by` is single-pass | — |
| Scatter exceeds RAM | — | ✅ — k-way merge spill, no code changes needed |
| Sort-only (no scatter) | ✅ — ~equal speed, simpler API | ✅ — ~equal speed, automatic spill |
| Reduce merge-join | ✅ | ✅ |
| User per-group Python fn | ✅ — `map_groups(fn)` | ❌ — requires UDF or Python loop |

---

## 9. Open Questions

**Q: Are `pyarrow.compute` and Polars similar in efficiency for shared operations?**

Mostly yes for aggregation (~1.2x), effectively identical for string ops.
Polars wins decisively on sort (20x), filter (5x), and has exclusive APIs for
hash and partition. Where they overlap, the difference rarely matters; the
Polars-only APIs are what drives the recommendation.

**Q: Polars supports more operations — should we unify on Polars to avoid conversions?**

No. The IPC layer should stay PyArrow — it's faster for file I/O, it's what
the rest of the ecosystem (Parquet, readers, writers) already uses, and it
avoids the `string_view` trap on the wire. Use Polars for computation between
reads and writes.

**Q: What are the actual compatibility issues with `LargeBinary` vs Polars `BinaryView`?**

The issue is the reverse: when Polars *writes* IPC (without `compat_level`),
it produces `binary_view` and `string_view`. PyArrow compute can't sort or
filter on those types. The fix is `compat_level=pl.CompatLevel.oldest()` on
all Polars IPC writes. For Parquet, there's no issue — Polars reads/writes
`large_binary` through Parquet cleanly.

The `_to_large_type()` helper in `external_sort.py` already correctly handles
the `string` → `large_string` promotion that happens when roundtripping
through Polars, which is the other direction of the same issue.
