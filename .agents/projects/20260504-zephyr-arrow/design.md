# Zephyr Arrow / Polars Migration

Processing data directly in Python is slow. It always has been.

Profiling some of our jobs, the ones that use Python to do the processing spend 50% - 70% of their time doing GC. Even jobs that don't use Python as much but use group by, can spent a fair amount of time in the Zephyr internals merging and sorting Python records.

We should move to using Arrow as the internal format for Zephyr and expose Arrow via the Zephyr API.
## Why
- Exposing Arrow to Zephyr jobs should make Python-heavy jobs at least 2x - 3x faster by migrating the logic to Polars and thereby removing all the GC time
- We can, likely, use more [efficient merging algorithms](https://pola.rs/posts/streaming-joins/) for the scatter / reduce phases
- We can avoid the cost of serialization and deserialization with `pickle`
- Again, if we move to user logic to Polars, we can parallelize jobs using multithreading at the Zephyr level, since the Python GIL will no longer be holding locks during processing
## Goals
- Migrate the internals of Zephyr to use Arrow for storage, including migrating the scatter format to use zstd compressed IPC files (IPC is the native Arrow format that doesn't incur any serialization or deserialization cost to write to disk).
- Keep the API to Zephyr the same, so existing jobs will continue to run, but without all the speed gains
- Expose an alternative Zephyr API that allows for processing data directly in Arrow and enabled the 2x - 3x speedups
## Design / Plan
- Replace `ScatterWritter` with a version the reads and writes ztsd compressed Arrow IPC
- Update the necessary sections of `plan.py`: `_merge_sorted_chunks`, `_sorted_merge_join`
- Update `spill.py` to use Arrow IPC as well
- Add a new `ArrowDataset` builder that has a very similar API to `Dataset` but provides a `PyArrow::BatchRecord` to the functions.
	- The main difference will be exposing a new version of `GroupBy` that allows for [Polars-level aggregations](https://docs.pola.rs/user-guide/expressions/aggregation/)
## Risks / Challenges

- **Reimplementing `_merge_sorted_chunks` might be tricky**  There's not a perfect analog in Polars / PyArrow for this function, although there are option and we can benchmark them to figure out the best one. See [research §6](research)
- **Arrow restricts the processable types** Technically `cloudpickle.dump` supports a much wider range of data types than Arrow, creating a breaking change. I reality, we only use data types supported by Arrow, so should be fine.
- **User-code needs to be migrated to Polars** To get the speedups, we'll need to migrate all the user-code in the pipelines to Polars. Representing things in Polars is more cumbersome than raw Python.
- **Change in the hashing function** I'm not sure why we'd want to read old scatter files, but we won't be able to after this.
- **Per-buffer zstd vs whole-file zstd compression ratio.** PyArrow IPC's `compression='zstd'` is worse than the current whole-batch `cloudpickle.dump → zstd.compress` flow. Could increase shuffle bytes 1.2-1.5x in the worst case. We should benchmark this. See [research §11.1](research).
## Future work
Once this is complete, we can return to [zephyr-performance](../20260430-zephyr-performance/design) and implement the parallelization improvements there and actually realize the benefit, since the GIL will not limit multithreading in *Arrow-native* stages.
