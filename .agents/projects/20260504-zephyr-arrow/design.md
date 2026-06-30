# Zephyr Arrow / Polars Migration

Processing data directly in Python is slow. It always has been.

Profiling some of our jobs, the ones that use Python to do the processing spend 50% - 70% of their time doing GC. Even jobs that don't use Python as much but use group by, can spend a fair amount of time in the Zephyr internals merging and sorting Python records.

We should move to using Arrow as the internal format for Zephyr and expose Arrow via the Zephyr API.

## Why

- Exposing Arrow to Zephyr jobs should make Python-heavy jobs at least 2x - 3x faster by migrating the logic to Polars and thereby removing the majority of the existing GC time
- We can avoid the cost of serialization and deserialization with `cloudpickle`
- If we move user logic to Polars, we can parallelize jobs using multithreading at the Zephyr level, since the Python GIL will no longer be holding locks during processing

## Goals

- Migrate the internals of Zephyr to use Arrow for storage, including migrating the scatter format to use zstd compressed Parquet files.
- Keep the API to Zephyr the same, so existing jobs will continue to run (but without all the speed gains)
- Expose an alternative Zephyr API that allows for processing data directly in Arrow and enables the 2x - 3x speedups

## Design / Plan

- Replace `ScatterWriter` with a version the reads and writes `zstd` compressed parquet and processes it in Polars
- Update the necessary sections of `plan.py`: `_merge_sorted_chunks`, `_sorted_merge_join`
- Add a new functions to the `Dataset` builder that expose `PyArrow.BatchRecord` instead of Python objects.
- The main difference will be exposing a new version of `group_by` that allows for doing aggregations using `RecordBatch`s

### `Dataset` API
See discussion in [api.md](api.md) for the details of options considered.

The `Dataset` API will remain unchanged so existing code continues to run. Functions like `load_parquet` and `group_by` will have a `batch_mode` flag that allows for processing.

An example pipeline could look something like this:
```
import polars as pl
import pyarrow as pa

def flat_map(batch: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
    hashed = compute_paragraph_hashes(batch)
    return pl.from_arrow(hashesd).rename({"doc_id": "id"}).to_arrow().to_batches()

def annotate_dups(key: str, records: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
    df = pl.from_arrow(records)
    has_dups = df.height > 1
    cano_id = df["id"][0]

    is_dup = pl.lit(has_dups) & (pl.col("id") != cano_id)
    return df.select(
      pl.col("id"),
      is_dup.alias("is_dup"),
      pl.when(is_dup)
          .then(pl.col("paragraph_span").struct.field("span"))
          .otherwise(pl.lit([], dtype=pl.List(pl.Int64)))
          .alias("span"),
      pl.col("file_idx"),
    ).to_arrow().to_batches()

Dataset
  .from_files(files)
  .load_parquet(batch_mode=True)
  .flat_map(flat_map)
  .group_by(
    key=col("hash"),
    sort_by=col("id")
    reducer=annotate_dups,
  )
```

## Risks / Challenges

- **Reimplementing `_merge_sorted_chunks` is complex in Polars**.  There's not a perfect analog in Polars / PyArrow for this function, although there are options and we can benchmark them to figure out the best one. See [research §6](research.md#6-k-way-merge-in-detail)
- **Arrow restricts the processable types**.  `cloudpickle.dump` supports a much wider range of data types than Arrow. Our workload is predominantly in data types that Arrow supports, but we will need an escape hatch for types that are not supported. Something like  using `cloudpickle.dump` and storing the bytes in Arrow should work.
- **User-code needs to be migrated to Polars**. To get the bulk of the speedup, we'll need to migrate user-code in the pipelines to Polars. Representing things in Polars is more cumbersome than raw Python.
- **Change in the hashing function**. The Polars hash function is different than the one currently implemented. This means we can't change this while a job is running or read old scatter files, which should be totally fine (but Claude is very worried about).
- **Per-buffer zstd vs whole-file zstd compression ratio.** PyArrow IPC's `compression='zstd'` is worse than the current whole-batch `cloudpickle.dump → zstd.compress` flow. Could increase shuffle bytes 1.2-1.5x in the worst case. We should benchmark this. See [research §11.1](./research.md#11-surprise-areas-worth-a-small-benchmark-before-committing).
- **Polars and Arrow Compatibility** any Polars IPC write consumed by PyArrow must pass `compat_level=pl.CompatLevel.oldest()`. Without it, Polars emits `string_view`/`binary_view` types that PyArrow 22 cannot sort, filter, or compare on. (This is also faster — 19ms vs 31ms — and slightly smaller.) Also, `string` columns roundtripped through Polars come back as `large_string`; use `promote_options='permissive'` on any subsequent `pa.concat_tables`.

## Compute Engine (PyArrow vs Polars vs DuckDB)

From the research in [compute_engine_research.md](compute_engine_research.md) (summarized below), Polars remains the best fit for Zephyr's hot-path operations. Its built-in `partition_by` and `Series.hash()` have no efficient DuckDB equivalent for in-process in-memory scatter. DuckDB is architecturally superior for larger-than-RAM workloads — its sort and join have production-grade spill-to-disk, unlike Polars' experimental streaming engine. For Zephyr's current use case (scatter that fits in RAM on large machines), Polars wins on the hot path; DuckDB is worth revisiting if scatter files grow past available memory.


| Criterion                                     | PyArrow                        | Polars                                  | DuckDB                                 |
| --------------------------------------------- | ------------------------------ | --------------------------------------- | -------------------------------------- |
| Efficiency: sort, filter                      | 391ms / 6ms                    | ✅ 20ms / 1ms                            | ~20–40ms / ~2ms (est.)                 |
| Efficiency: group-by, string ops              | ✅ ~equal                       | ✅ ~equal                                | ✅ ~equal (fastest in H2O at scale)     |
| API: scatter routing (`partition_by`, `hash`) | ❌ no API                       | ✅ built-in, single-pass                 | ⚠️ file-based only; no in-memory split |
| API: join                                     | ❌ no sorted-merge join         | ✅ built-in                              | ✅ hash + merge join                    |
| API: expression ergonomics                    | functional (`pc.and_(...)`)    | ✅ operator overloading                  | SQL strings                            |
| API: per-group user Python fn                 | ❌                              | ✅ `map_groups(fn)`                      | ❌ requires UDF                         |
| IPC read/write speed                          | ✅ 9ms uncompressed / 27ms zstd | 19–31ms                                 | not yet benchmarked                    |
| IPC type safety (no `string_view` trap)       | ✅ always safe                  | ⚠️ needs `compat_level=oldest()`        | ✅ emits `string`, not `string_view`    |
| Zero-copy with Arrow (numerics)               | ✅ is Arrow                     | ✅ yes                                   | ✅ C data interface                     |
| Zero-copy with Arrow (strings)                | ✅ is Arrow                     | ❌ copies to StringView                  | ✅ (fails on `string_view` input)       |
| Memory: spill-to-disk for sort/join           | ❌ no spill                     | ⚠️ streaming merge_sorted experimental  | ✅ k-way merge, all operators           |
| Streaming larger-than-RAM ops                 | ❌                              | ⚠️ streaming / `LazyFrame` experimental | ✅ production-ready                     |
## Future work
Exposing sql is interesting in terms of sql's flexibility and ease for writing ad-hoc queries. Parsing sql to handle distributing it across multiple nodes is well outside the scope of the work we want to do, but there are a few future paths we can explore:
- Once `RecordBatch` objects are exposed, users can write sql transformations over them using DuckDB or DataFusion
- We can explore deploying [smallpond](https://github.com/deepseek-ai/smallpond) for ad-hoc (or possibly data processing) queries in a more always-on fashion
