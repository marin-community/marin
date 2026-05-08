# Zephyr Arrow / Polars Migration

Processing data directly in Python is slow. It always has been.

Profiling some of our jobs, the ones that use Python to do the processing spend 50% - 70% of their time doing GC. Even jobs that don't use Python as much but use group by, can spend a fair amount of time in the Zephyr internals merging and sorting Python records.

We should move to using Arrow as the internal format for Zephyr and expose Arrow via the Zephyr API.
## Why
- Exposing Arrow to Zephyr jobs should make Python-heavy jobs at least 2x - 3x faster by migrating the logic to Polars and thereby removing the majority of the existing GC time
- We can, likely, use more [efficient merging algorithms](https://pola.rs/posts/streaming-joins/) for the scatter / reduce phases
- We can avoid the cost of serialization and deserialization with `cloudpickle`
- Again, if we move to user logic to Polars, we can parallelize jobs using multithreading at the Zephyr level, since the Python GIL will no longer be holding locks during processing
## Goals
- Migrate the internals of Zephyr to use Arrow for storage, including migrating the scatter format to use zstd compressed IPC files (IPC is the native Arrow format that doesn't incur any serialization or deserialization cost to write to disk).
- Keep the API to Zephyr the same, so existing jobs will continue to run (but without all the speed gains)
- Expose an alternative Zephyr API that allows for processing data directly in Arrow and enabled the 2x - 3x speedups
## Design / Plan
- Replace `ScatterWriter` with a version the reads and writes `zstd` compressed Arrow IPC
- Update the necessary sections of `plan.py`: `_merge_sorted_chunks`, `_sorted_merge_join`
- Update `spill.py` to use Arrow IPC as well
- Add a new `ArrowDataset` builder that has a very similar API to `Dataset` but provides a `PyArrow.BatchRecord` to the functions.
	- The main difference will be exposing a new version of `GroupBy` that allows for [Polars-level aggregations](https://docs.pola.rs/user-guide/expressions/aggregation/)

### `ArrowDataset` example
```
    def pyarrow_flat_map_fn(path: str) -> Iterator[pa.RecordBatch]:
        file_idx = path_to_idx[path]

        for batch in _load_batches(path):
            col_names = batch.schema.names
            col_values = list(batch.columns)
            col_values.append(pa.array([file_idx] * batch.num_rows, type=pa.int64()))
            col_names.append("file_idx")

            yield pa.RecordBatch.from_arrays(col_values, names=col_names)

    def polars_flat_map_fn(path: str) -> Iterator[pa.RecordBatch]:
        """Same as ``pyarrow_flat_map_fn``, but using Polars."""
        file_idx = path_to_idx[path]

        for batch in _load_batches(path):
            df = pl.from_arrow(batch)
            df = df.with_columns(pl.lit(file_idx).cast(pl.Int64).alias("file_idx"))
            table = df.to_arrow()
            yield from table.to_batches()


    def polars_flat_map_fn(path: str) -> Iterator[pa.RecordBatch]:
        """Same as ``pyarrow_flat_map_fn``, but falling back to Python."""
        file_idx = path_to_idx[path]

        for batch in _load_batches(path):
            py_batch = [{"file_idx": file_idx, **hash_record} for item in batch.to_pylist()]
            yield pa.RecordBatch.from_pylist(py_batch)

    def pyarrow_reducer(key: Any, items: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Reduce a group of ``pa.RecordBatch``s into a single ``pa.RecordBatch``."""
        pass

    dataset = (ArrowDataset.from_files(input_files)
        .flat_map(pyarrow_flat_map_fn)
        .map(compute_paragraph_hashes)
        .group_by(
            key_columns=["hash"],
            sort_by_columns=["id"],
            reducer=pyarrow_reducer,
            arrow=True,
        ))
    results = ctx.execute(dataset).results
```

## Risks / Challenges

- **Reimplementing `_merge_sorted_chunks` is complex in Polars**.  There's not a perfect analog in Polars / PyArrow for this function, although there are options and we can benchmark them to figure out the best one. See [research §6](research.md#6-k-way-merge-in-detail)
- **Arrow restricts the processable types**.  `cloudpickle.dump` supports a much wider range of data types than Arrow. Our workload is predominantly in data types that Arrow supports, but we will need an escape hatch for types that are not supported. Something like  using `cloudpickle.dump` and storing the bytes in Arrow should work.
- **User-code needs to be migrated to Polars**. To get the bulk of the speedup, we'll need to migrate user-code in the pipelines to Polars. Representing things in Polars is more cumbersome than raw Python.
- **Change in the hashing function**. The Polars hash function is different than the one currently implemented. This means we can't change this while a job is running or read old scatter files, which should be totally fine (but Claude is very worried about).
- **Per-buffer zstd vs whole-file zstd compression ratio.** PyArrow IPC's `compression='zstd'` is worse than the current whole-batch `cloudpickle.dump → zstd.compress` flow. Could increase shuffle bytes 1.2-1.5x in the worst case. We should benchmark this. See [research §11.1](./research.md#11-surprise-areas-worth-a-small-benchmark-before-committing).
- **Polars and Arrow Compatibility** any Polars IPC write consumed by PyArrow must pass `compat_level=pl.CompatLevel.oldest()`. Without it, Polars emits `string_view`/`binary_view` types that PyArrow 22 cannot sort, filter, or compare on. (This is also faster — 19ms vs 31ms — and slightly smaller.) Also, `string` columns roundtripped through Polars come back as `large_string`; use `promote_options='permissive'` on any subsequent `pa.concat_tables`.

## PyArrow vs Polars


From the research in [pyarrow_vs_polars.md](pyarrow_vs_polars.md) (summarized in the table below) Polars seems like a better fit for our use case, since the bulk of the time is spent processing data versus reading and writing it. The Lazy API also opens up some possible improvements for streaming data as well.

| Criterion | PyArrow | Polars |
|---|---|---|
| Efficiency (sort, filter) | 391ms / 6ms | ✅ 20ms / 1ms |
| Efficiency (group-by, string ops) | ✅ ~equal | ✅ ~equal |
| API: scatter routing (`partition_by`, `hash`) | ❌ no API | ✅ built-in |
| API: join | ❌ no sorted-merge join | ✅ built-in |
| API: expression ergonomics | functional (`pc.and_(...)`) | ✅ operator overloading |
| IPC read/write speed | ✅ faster (9ms uncompressed) | slower (19–31ms) |
| IPC type safety (no `string_view` trap) | ✅ always safe | ⚠️ needs `compat_level=oldest()` |
| Zero-copy with Arrow (numerics) | ✅ is Arrow | ✅ yes |
| Zero-copy with Arrow (strings) | ✅ is Arrow | ❌ copies to StringView |
## Future work
Once this is complete, we can return to [zephyr-performance](../20260430-zephyr-performance/design.md) and implement the parallelization improvements there and actually realize the benefit, since the GIL will not limit multithreading in *Arrow-native* stages.
