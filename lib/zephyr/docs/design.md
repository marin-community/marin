# zephyr - Marin light dataset library

We have a _lot_ (like 100) redundant data processing patterns in the Marin code
base.  Most of them involve ad-hoc use of Ray + HF datasets + manual processing.
Since most of these are working on smaller datasets, we don't have a lot of
concern about completion or stalled jobs. The lack of consistency though, means
we need to know 10 different ways to run jobs, understand how errors are
handled, and deal with a wide variety of "how do I load a jsonl" file. Given our
desire to abstract away from Ray for our cluster management, we also want to
consolidate our usage of Ray as much as possible.

To that end, we've identified a set of common patterns in our code we can
consolidate using some standard data processing patterns. We're calling this
"zephyr".

## Outline

To use Zephyr, like Beam/Flume/Ray-Data etc, you define a pipeline of operations against a Dataset:

```python
dataset = (Dataset
  .from_files("gs://my-cool-data/*.jsonl")
  .flat_map(load_jsonl) # load each file, and yield individual records in the file
  .map(convert_to_dolma_format)
  .write_jsonl("gs://ml-cool/output/processed-{shard%05d}-of-{total%05d}")
)

for filename in backend.execute(dataset):
  print(filename)
```

You focus on the individual operations, and the dataset & backend handle the
scaling and load-management for you. To process a dataset, you need a `Backend`:
we provide simple multi-processing and synchronous backends for small datasets,
as well as a Ray backend which runs tasks on a cluster. A launcher script
handles creating the backend for you and running your code against a Draccus compatible API:

```python
# experiments/my_processor.py
def my_entry_point(conf: Config, backend: zephyr.Backend):
  dataset = ...
  backend.execute(dataset)
```

```bash
uv run zephyr experiments/my_processor.py --backend=ray --max_tasks=100 --memory=1GB
```

## Datasets

Users primarily interact with zephyr via the Dataset class. It exposes a
standard set of constructors and transformations for construction,
transformation, and serialization.

`from_{files,list,iterator}`

Construct a dataset from a list or glob.

`flat_map(fn)`

Given a function from `item` -> `list[new_item]`, return a new `Dataset` which
contains `new_item`. A common use case is for loading files: given a dataset of
filenames, `ds.flat_map(load_jsonl)` returns a dataset which contains the
individual JSON lines.

`map(fn)`

Map the given function over the dataset, transforming the elements.

`filter(fn)`

Return a new dataset containing only elements where `fn(item)` returns True-ish.

`batch(window_size)`

Given a dataset of `item`, return a dataset of `list[item]` where each list is a
distinct set of `window_size` elements. If the dataset is sharded, batching is
independently performed on each shard.

`write_{jsonl|parquet}(file_pattern)`

Serialize the dataset to disk at the given location. The output expects a
pattern for serializing sharded datasets, such as
"gs://path-to-data/corpus-{{shard:05d}}-of-{{total:05d}}.jsonl.gz". The `shard`
and `total` formatters are automatically replaced by the backend when
serializing. If there is only one shard being processed, the shard & total
patterns may be omitted.

`group_by(key, reducer, num_output_shards=None)`

Group items by a key function and apply a reducer to each group. The operation
is implemented as a two-phase shuffle:

1. **Local grouping**: Each shard hashes items by `key` and redistributes them
   into `num_output_shards` buckets
2. **Shuffle & reduce**: Items with the same key are brought together and the
   `reducer(key, Iterator[items])` is called to produce a single result

This is commonly used for aggregations like counting, deduplication, or
computing statistics per group. If `num_output_shards` is not specified, it
defaults to the current number of shards.

```python
ds = (Dataset
  .from_files("gs://data/*.jsonl")
  .flat_map(load_jsonl)
  .group_by(
    key=lambda x: x["user_id"],
    reducer=lambda key, items: {"user": key, "count": sum(1 for _ in items)}
  )
)
```

`deduplicate(key, num_output_shards=None)`

Remove duplicate items based on a key function.

```python
ds = (Dataset
  .from_files("gs://data/*.jsonl")
  .flat_map(load_jsonl)
  .deduplicate(key=lambda x: x["id"])
)
```

`reduce(local_reducer, global_reducer=None)`

Reduce the entire dataset to a single value using two-phase reduction:

1. **Local reduction**: Apply `local_reducer` to each shard independently
2. **Global reduction**: Pull all shard results to the controller and apply
   `global_reducer` (defaults to `local_reducer` if not specified)

This is useful for computing dataset-wide statistics like totals, min/max, or
custom aggregations.

```python
# Count total items
total = (Dataset
  .from_list(range(1000))
  .reduce(local_reducer=lambda items: sum(1 for _ in items))
)

# Custom aggregation
stats = (Dataset
  .from_files("gs://data/*.jsonl")
  .flat_map(load_jsonl)
  .reduce(
    local_reducer=lambda items: {"count": sum(1 for _ in items), "sum": sum(x["value"] for x in items)},
    global_reducer=lambda shards: {
      "total_count": sum(s["count"] for s in shards),
      "total_sum": sum(s["sum"] for s in shards)
    }
  )
)
```

`sorted_merge_join(right, left_key, right_key, combiner=None, how="inner")`

Perform a streaming merge join between two datasets that are already sorted and
co-partitioned. Preconditions:

- Both datasets must have the same number of shards
- Corresponding shards (left[i], right[i]) must contain the same key ranges
- Items within each shard must be sorted by their join key

These preconditions are typically met when both datasets come from `group_by`
with the same key and `num_output_shards`.

Only supports inner and left joins (no right or outer joins).

```python
# Both datasets grouped by the same key
docs = (Dataset
  .from_files("gs://docs/*.jsonl")
  .flat_map(load_jsonl)
  .group_by(key=lambda x: x["id"], reducer=keep_first, num_output_shards=100)
)

attrs = (Dataset
  .from_files("gs://attrs/*.jsonl")
  .flat_map(load_jsonl)
  .group_by(key=lambda x: x["id"], reducer=keep_first, num_output_shards=100)
)

# Sorted merge join - no shuffle needed
joined = docs.sorted_merge_join(
  attrs,
  left_key=lambda x: x["id"],
  right_key=lambda x: x["id"],
  how="inner"  # or "left"
)
```

The default combiner merges dictionaries: `{**left, **right}`. For custom
combining or handling nulls in left joins, provide a custom combiner function.

## Backends

A `backend` takes a dataset and runs it, yielding the results from the final
stage. We provide local backends for testing and small datasets, as well as a
Ray backend which runs operations in parallel across a Ray cluster.

You can create a backend manually and specify the requirements for running
individual tasks:

```
backend = RayBackend(memory=1_000_000_000, cpu=1)
backend.execute(ds)
```

Most usage will use the `zephyr` script to handle creating an appropriate
backend for you and calling your entry point function:

```
uv run zephyr src/my/script.py --backend=ray --cluster=xyz
```

### Optimizations

By default, the Ray backend creates a separate _shard_ for each input item in a
cluster - operations like `map` are run in parallel across shards, but
sequentially within a shard.  Naive execution of a dataset would result in each
individual stage being "materialized" - this can result in memory blow-ups and
slow execution. For example, in:

```
Dataset.from_files().flat_map(load_jsonl).map(x).filter(y).map(z).write_jsonl()
```

We want to avoid computing the intermediate representation of the dataset if
possible. Fortunately for us, we only rely on trivial transformations, thus we
can easily fold these operations into a single combined map against each file
shard:

```
Dataset.from_files().map_shard(load_jsonl | x | y | z)
```

## Alternatives Considered

Obvious alternatives to our own library are Beam and Ray-data. Both of these
support basic dataset transformations and would be reasonable fits for some of
our use cases. We elected to make our own abstraction for a few reasons:

* We're unclear about our long-term usage of Ray, and didn't want to embed it everywhere in the code base at this point.
* The only fully implemented Beam backend is on GCP, and the execution model is very disjoint from e.g., our Ray cluster. This makes it suitable for big standalone jobs, but we want a dataset pattern that works for smaller jobs and is more easily tracked as well.

While we decided to make our own abstraction, there is no reason we can't add
Beam or Ray-Data as an additional _backend_ to our system. The cost of defining
our own dataset layer is quite trivial and thus we feel it's worth it to buy us
additional flexibility.


## Implementation Notes

### Repartitioning

Zephyr provides `reshard(num_shards)` to change the parallelism of a dataset
without changing the underlying order. This is useful for increasing or
decreasing the amount of parallel work. For example, a pipeline like
`Dataset.from_files([".../corpus.jsonl"]).flat_map(load_jsonl)` will only run on
a single worker by default. Using `.reshard(num_shards=20)` forces the runtime
to redistribute the data across 20 shards, allowing subsequent operations to run
in parallel.

The implementation is best-effort and redistributes chunks across shards in
round-robin fashion without splitting individual chunks.
