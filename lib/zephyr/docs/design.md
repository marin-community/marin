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


## Future Work

zephyr is missing 2 common operations familiar from existing distribution
toolkits: shuffle, and repartition. As our existing code does not rely on these
patterns, we have deferred work on them in the original design.

_Repartitioning_ changes the parallelism of the dataset, without attempting to
change the underlying order. It is commonly used to increase or reduce the
amount of parallel work occurring on a dataset. For example, a pipeline like
`Dataset.from_files([".../corpus.jsonl]).flat_map(load_jsonl)` will only run on
a single worker by default. Repartitioning forces the runtime to explicitly
split up the underlying files into separate execution shards, allowing map
operations to occur in parallel.

_Shuffle_ or _group_ operations re-order the dataset, usually by grouping
related items by a key. This is commonly used in tasks like de-duplication,
where we want to keep a single example from a group of near-identical documents.
