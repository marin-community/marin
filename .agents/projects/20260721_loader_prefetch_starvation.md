# DataLoader prefetch starvation reproduction

## Scope

This investigation follows the open loader observation in
[#7012 comment 5007009232](https://github.com/marin-community/marin/issues/7012#issuecomment-5007009232): a
512-device training run was compute-healthy while fed, but reportedly stalled after its input prefetch buffer drained
at step 38 with Datakit and step 43 with Slim. The loader observation does not yet have its own issue under
[#6710](https://github.com/marin-community/marin/issues/6710).

This note reproduces one mechanism in the current loader: an initially full queue can hide a sustained producer-rate
deficit, after which the consumer blocks at all-or-nothing prefetch boundaries. It does **not** identify whether the
production producer deficit came from the dataset implementation, object storage, network behavior, or a slow host.

## Current data path

`experiments/grug/moe/train.py` constructs `levanter.data.loader.DataLoader` with its defaults:

- `prefetch_size=32`
- `max_buffered_batches=64`

`DataLoaderIterator._produce_batches` describes 32 training steps, flattens all local indices for those steps, and
awaits one `data_store.get_batch(...)`. Only after that entire call completes does it batchify and enqueue the 32
results. It does not enqueue partial results from a fetch window.

For the reported d5120/L48/E64/top4 run, the global batch was 8,192 sequences of length 4,096 across 128 hosts. Each
host therefore requested 64 sequences (262,144 tokens) per training step. At about 12.1 seconds per step, the
sustained per-host requirement was about 5.29 sequences/s (21,665 tokens/s). A default 32-step fetch window contains
2,048 sequences (8,388,608 tokens) on each host and must complete within about 387 seconds to sustain that rate. The
global consumer waits for the slowest host.

The two datasets share this loader and eventually converge on the same `TokenSeqDataset`/`TreeCache` read path, but
their transforms and cache layout differ. The available run evidence cannot distinguish backend throughput from
host/network tail latency.

## Bounded reproduction

Run on one CPU:

```bash
uv run --package marin-levanter \
  python lib/levanter/scripts/bench/bench_loader_prefetch_starvation.py
```

The script uses a synthetic infinite dataset whose `get_batch` takes 0.4 seconds and returns four training batches
at once. Its sustained capacity is therefore 0.1 seconds/batch (10 batches/s). Both scenarios use an eight-batch
buffer and a 1.3-second warmup so the buffer is full before consumption.

The default run on 2026-07-21 produced this boundary:

| Consumer interval | Requested rate | Empty queue | Stall steps (>=0.1s) | Result |
| --- | ---: | --- | --- | --- |
| 0.05s/batch | 20 batches/s | yes | 16, 20 | starvation reproduced |
| 0.125s/batch | 8 batches/s | no | none | sustained |

In the failing case, queue occupancy fell from eight to zero, then each fetch released another four-batch burst. The
two measured waits were 0.150 and 0.201 seconds. In the passing case, all 24 `next()` calls completed below the
0.1-second threshold. The script asserts both outcomes, verifies every backend request contains exactly four training
batches, and prints each step's wait time and queue occupancy.

The smallest behavioral boundary is independent of buffer depth:

```text
consumer_interval < fetch_delay / prefetch_size  -> eventual drain and starvation
consumer_interval > fetch_delay / prefetch_size  -> sustainable after warmup
```

A deeper buffer moves the first stalled step but does not change that sustained-rate boundary. No accelerator,
multi-host runtime, dataset cache, or network is needed to reproduce it.

## Interpretation and next measurement

The reproduction confirms that a warm prefetch queue can make an under-rate producer look healthy for many steps,
and that the current 32-step all-or-nothing fetch makes starvation visible in bursts. It does not establish that
smaller fetch windows would increase sustained throughput; they would only change latency and queue shape unless
they expose additional backend concurrency or reduce tail amplification.

The next production measurement should record, per host and per fetch window:

- `get_batch` start/end time and number of examples;
- queue occupancy over time, not only after a stall;
- median, p95, p99, and maximum host fetch latency;
- dataset/cache identity and bytes read;
- host/network error and retry counters.

That measurement should be compared at independent host counts, `prefetch_size` values, buffer depths, and datasets.
GPU profiling is not required until the producer behavior has been separated from the input backend.

## Related production evidence

[#7344](https://github.com/marin-community/marin/issues/7344) records a later 500-step Datakit run with transient
loader warnings and a separate hard hang at step 145. That issue attributes the hard hang to a collective, not the
loader. Slim also completed a 100-step run. Those observations are compatible with transient loader starvation but
do not support calling every later training hang a loader failure.
