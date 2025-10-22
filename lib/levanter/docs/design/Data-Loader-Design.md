# Data Loader Design

## Context

Levanter, like any LM training framework, needs to read (usually text) data to feed it to the model. This
process involves reading lots of raw text, tokenizing it, and splitting it up into model-sized chunks.
Unlike many other ML workloads, the mapping from raw data to model-sized chunks is not 1:1, but in general
many-to-many. This is because we typically take a moving window of tokens from a list of documents.

Levanter is designed to be completely deterministic, meaning that if you run the same code on the same data on
the same hardware, you should get the same results. This is important for debugging and for reproducibility.
In order to guarantee determinism, our data loading pipeline must be deterministic as well.
Moreover, to the extent possible, we want deterministic batch order even if the number of machines changes.

Data is usually stored in compressed shards, each morally equivalent to an iterator over a list of documents.
In particular, we don't usually have random access. This implies that we need to produce a cache of processed
documents that does allow for random access. Random access is important for resuming training quickly,
as well as for shuffling.

Early on in Levanter's development, we made the decision to support "quick start" training, where we can start
training while we are still building the cache. This is helpful when iterating on the data pipeline
and removes a step from the training process. This implies that we need to support simultaneous reading and writing
of the cache.

Levanter also wants to support dynamic mixtures of data, where we reweight different datasets on the fly. To do so,
we need separate caches for each dataset.

In practice, even for the relatively "small" examples one has in LM training (compared to vision, for example),
we also want to do sharded loading.

## Goals

To summarize:

* **Deterministic batches**: For any cluster size during training, we want the same batches to be
  generated in the same order.
* **Instant Resume**: We want training to be able to resume training quickly, without losing too much progress.
* **Quick Start**: Unless it is logically impossible (e.g. for shuffling), we want to be able to start training
  while we are still building the cache.
* **Random Access**: We want to be able to jump around the dataset, for shuffling and for resuming.

## Cache Design

### Terminology

* **Document**: A document is a single datum that is fed into the model. Documents are typically tokenized and
  preprocessed. For LM training, a document is just a string, but for other tasks, it might be more complex. For example,
  there might be a "prompt" and a "response".
* **Shard**: A shard is a list of *raw* documents that not been tokenized/preprocessed.
* **Cache**: A cache is a list of *processed* documents that have been tokenized/preprocessed. These are stored as
a group of [TensorStore](https://google.github.io/tensorstore/) arrays structured to behave like a column store. These
arrays are stored as Zarray arrays, typically compressed.
* **Reader**: A reader is a process that reads from the cache. Typically, there is one reader per machine.
* **Writer**: A writer is a process that writes to the cache. Typically, there is one writer per *cache*.
* **Global ordering**: Each document in a cache is assigned a global index. This index is deterministic, but
a bit hard to compute a priori.
* **Processor** or **Tokenizer**: A function that takes a batch of raw documents and returns a batch of processed documents.
* **Example**: A single datum that is fed into the model. Examples are typically composed of fragments of documents.
  For example, we might take a moving window of tokens from the concatenation of a list of preprocessed documents.
* **Ledger**: A ledger is a list of metadata about the cache. This includes the number of documents in each shard
as well as some information to make it less likely that you accidentally reuse a cache.


### Cache Structure

A cache is a [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) of [levanter.store.JaggedArray][]s, each
representing a different field of the processed documents. Each JaggedArray is a group of either two or three arrays:

* **Data**: The actual data, stored as a Zarray array. All the "tokens" for a given field for all documents are stored in a single flat array.
* **Offsets**: The offsets into the data array for each document. This is a 1-D array of length N+1, where N is the number of documents.
* **Shape** (optional): The shape of the data for each document. This is only present for fields that are not 1-D.

For tokenized documents, a cache looks like this:

```
cache
├── train
│   ├── input_ids
│   │   ├── data
│   │   │   ├── c
│   │   │   │   └── 0
│   │   │   └── zarr.json
│   │   └── offsets
│   │       ├── c
│   │       │   └── 0
│   │       └── zarr.json
│   ├── shard_ledger.json
```

(Typically there's a lot more files in the `c` directories, but I've omitted them for brevity.)

The stuff under `input_ids/data` is the actual data, and the stuff under `input_ids/offsets` is the offsets.

In code, this is modeled in [levanter.store.TreeStore][].

### Cache Construction

We use Ray to handle the construction of the cache. There are 4 types of processes/actors that we create using Ray:

- `_TreeStoreCacheBuilder`: This actor is responsible for building the cache. It forks off actors for reading
  shards and processing documents. It acts as a callback for these processes.
- `_OrderedCacheWriter`: This actor is responsible for writing to the cache. It is responsible for writing the
  processed documents to the cache in the correct order.
- `WorkQueueDispatcherActor`: This actor is responsible for reading batches of documents from a group of shards. It dispatches
  documents to a group of processors, which are responsible for processing the documents.
- `_BatchProcessorQueue`: This actor is responsible for managing the queue of batches of documents to be processed. It
  actually calls the processors to process the documents and then forwards the results to the writer.

The basic flow is that the builder forks off a bunch of `WorkQueueDispatcherActor`s, which read from the shards and
dispatch the documents to the processors. The processors process the documents and send the results to the writer,
which writes them to the cache.

The writer is responsible for writing the documents to the cache in the correct order. In particular, fix a batch
size B. The writer writes the documents in batches of size B, round-robin from the shards. Once a shard is exhausted,
it is removed from the list of shards.

The writer maintains a "ledger" of the cache, which has the number of documents processed in each shard, as well as
whether or not the shard is done. This ledger is used for resuming cache construction.

## Datasets and the Data Loader

Along with the cache, we introduce interfaces and classes for working with the cache. The main classes are:

- [levanter.data.AsyncDataset][]: This is the base class for all datasets. The main method it exposes is
  `get_batch(indices: Sequence[int])` which (asynchronously) returns a batch of documents for the given indices.
- [levanter.data.DataLoader][]: This is a class that wraps a dataset and provides an iterator over the dataset. It prefetches
  the parts of batches that each machine needs. It has an iterator and supports "seeking" to a particular batch.
- [levanter.store.TreeCache][]: This is an AsyncDataest that wraps a cache and exposes a `get_batch` method that returns
  a batch of documents for the given indices.
- [levanter.data.TokenSeqDataset][]: This is an async dataset that does the chunking of documents into examples. It
  takes a cache and a `max_seq_len` and returns examples of length `max_seq_len`.
- [levanter.data.PermutationDataset][]: This is a dataset that permutes the indices of another dataset. It is used for shuffling.
- [levanter.data.EraShufflingDataset][]: This is a dataset that emulates the behavior of a shuffle buffer, while
  still support random access. It is used for shuffling while still building the cache.
- [levanter.data.MixtureDataset][]: This is a dataset that mixes together multiple datasets with different weights.

### [levanter.data.PermutationDataset][]

The PermutationDataset is a dataset that permutes the indices of another dataset. It is backed by a pseudo-random
permutation (PRP). PRPs give you random access to a permutation with O(1) time and memory.

### [levanter.data.EraShufflingDataset][]

The EraShufflingDataset is a dataset that emulates the behavior of a shuffle buffer, while still supporting random access.
It works by defining an "era" length, which is the number of samples that are shuffled together. After an era is exhausted,
the dataset shuffles the next era.


### [levanter.data.MixtureDataset][]

We implement "stable mixtures" where the number of samples from each domain for each batch is fixed. This acts
as a kind of variance reduction, while also enabling random access and sampling without replacement.

Note: I believe it's impossible to sample without replacement and have random access with sampled batches.
This is because for each item `i`, you sample a domain `d_i`, but you can't know which indices in the domain have
been sampled. With replacement is easy so long as you know how big each domain is ahead of time, which means
you can't do streaming.


## Performance

### Reading from the Cache

TensorStore can sustain high throughput but has pretty terrible latency (when hitting GCS).
The latency can be on the order of a second. We mitigate this by prefetching the data in the DataLoader.

With prefetching we can sustain about a million tokens per second per host, wihch is sufficient.
In particular, when training a GPT-2 small model on a v3-32, loading is able to keep up with training.
However, 3/4 of evaluation time is spent blocked on loading data, so we could potentially speed up evaluation.
(However it's still twice as fast as with the old cache and data loader.)

### Writing to the Cache

Writes are also slow, but we also batch up the writes, typically writing 8K documents at a time.
