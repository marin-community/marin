# API Design

## Constraints

To get the maximum benefit from Zephyr's Arrow/Polars internals, users need to do two things:

1. Use Polars as much as possible in `map` / `flat_map` to avoid doing data processing in Python.
2. Use the new `group_by` and `reduce` functions so Zephyr can avoid creating Python objects in the reducer and combiner. This requires the preceding step to produce `DataFrame`s (not Python dicts).

---

## API Options
### Option A: Separate `ArrowDataset` class

Create two distinct dataset types:
1. `Dataset` — unchanged; existing jobs continue to run with Python `dict` items.
2. `ArrowDataset` — same pipeline API (`from_files`, `map`, `flat_map`, `group_by`, `reduce`) but `group_by` and `reduce` receive `pa.RecordBatch`.

### Option B: New `group_by_arrow / `reduce_arrow` methods on `Dataset`

Leave `Dataset` as-is and add suffixed variants.

### Recommendation: Option A (`ArrowDataset`)

`AGENTS.md` says "Use separate classes over boolean flags for variant behavior" — the same principle applies to `_polars`-suffixed method variants, which are just boolean dispatch with extra typing overhead.

The separate class makes the "you're committing to Arrow/Polars processing" contract explicit at construction time. It make it less likely to have pipelines that interleave Python and Arrow processing functions in ways that are confusing or incorrect.

The existing `Dataset` stays as-is for now. Per `AGENTS.md`, no backward-compat shims — update all call sites if/when we want to rename it.

## `group_by` and `reduce` signatures

### Proposed signatures

```python
def group_by(
    self,
    key_column: str, # Name of the column to use as the key
    *,
    reducer: Callable[[str, Iterator[RecordBatch]], Iterator[RecordBatch]],
    sort_by_column: str | None = None,  # Name of the column to sort by
    num_output_shards: int | None = None,
    combiner: Callable[[str, Iterator[RecordBatch]], Iterator[RecordBatch]] | None = None,
) -> ArrowDataset[RecordBatch]:

def reduce(
    self,
    local_reducer: Callable[[Iterator[RecordBatch]], RecordBatch],
    global_reducer: Callable[[Iterator[RecordBatch]], RecordBatch] | None = None,
) -> ArrowDataset[RecordBatch]:
```

## Open Questions

1. **ArrowDataset strictness** The current proposal only changes `group_by` and `reduce`, but to make the API as coherent as possible, `map` and `flat_map` should operate on `RecordBatch` objects as well. That said, many pipelines today do a small `flat_map` step on paths and utilize that information (to map back to original files, for example) that would make this hard. We could bake in the common patterns to things like `Dataset.from_files` and have that return `RecordBatch` objects with the needed columns. Using the `ArrowDataset` API does make this evolution easier, should we choose to do it.

2. **Multi-column keys and sort.** There is some loss of expressivity with the single `key_column: str`, but that column can be a struct (computed in the previous `map` phase).

3. **RecordBatch or DataFrame/LazyFrame** Polars operates on `DataFrame` (or `LazyFrame` for streaming) and since the internals are Polars and Polars provides an easier API for writing pipelines it could be nice to expose `DataFrame` instead of `RecordBatch`. Things like `dupekit` already operate on `RecordBatch` and so having both in a pipeline could be confusing. The conversion from one to the other is a single function call either way.

   One middle ground: keep `RecordBatch` as the public type (consistent with `dupekit`, stable, not tied to Polars internals) but provide a `polars_reducer` decorator that handles the conversion boilerplate:

   ```python
   def polars_reducer(fn: Callable[[Any, pl.DataFrame], pl.DataFrame]):
       def wrapped(key: Any, batches: Iterator[RecordBatch]) -> Iterator[RecordBatch]:
           df = pl.from_arrow(pa.Table.from_batches(list(batches)))
           result = fn(key, df)
           yield from result.to_arrow().to_batches()
       return wrapped

   # Usage
   @polars_reducer
   def my_reducer(key: str, df: pl.DataFrame) -> pl.DataFrame:
       return df.group_by("doc_id").agg(pl.col("score").sum())
   ```

   This keeps the core API type-stable while making Polars the easy path. The decorator can live in `zephyr.polars_utils` or similar.
