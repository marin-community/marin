# Zephyr shuffle: cut GCS I/O while keeping preemption resilience

Status: design (2026-06-21). Companion to research logbook
`.agents/logbooks/zephyr-shuffle-perf.md`.

## Problem (data-established)

On the datakit normalize benchmark (shuffle-heavy, the agreed signal), a clean
sequential per-shard A/B and three cluster profiles show the normalize shard is
**~50% irreducible GCS I/O**, not shuffle-serialization-bound:

- scatter shard ≈ 54% of total; ~48% of scatter is the **GCS upload** of the
  shuffle chunk (≈26% of total shard time).
- reduce shard ≈ 46%; its *read* is mostly CPU (merge+decompress+pickle.loads,
  GCS-read only 0.44s/shard); its *output write* (the normalized deliverable,
  ~12GB) is ~28% of total shard and necessary.
- CPU wins already landed: vectorized whitespace (−16% scatter), pickle proto-5,
  native sort-key, codec corrected lz4→**zstd** (lz4 was a measured −27% I/O
  regression). Net **−9.7%** end-to-end. A full columnar/RecordBatch reduce adds
  only ~−6% (removes pickle round-trip) → ~−15-18%, still short of 20%.

The one lever big enough for ≥20% is the **scatter→GCS→reduce round-trip**
(≈26GB written + read per run; the scatter upload alone ≈26% of shard time).
GCS gives durability (=preemption resilience) but at ~35 MB/s contended.

## Proposal: local-disk shuffle with direct reducer fetch + recompute-on-loss

Model the shuffle on Spark/MapReduce external shuffle, adapted to Iris actors:

1. **Scatter writes chunks to mapper-LOCAL disk** (NVMe ~GB/s) instead of GCS.
   Same zstd parquet format + sidecar; just a local path.
2. **Reducers fetch chunks directly from mapper actors** via an RPC
   (`fetch_shard_chunk(mapper, target_shard)`), streamed over the intra-cluster
   network (fast, not GCS). Predicate-pushdown/row-group selection happens
   mapper-side so only the target shard's bytes cross the wire.
3. **Resilience = recompute-on-loss (no GCS for the happy path, no
   non-preemptible VMs).** The coordinator already tracks shard completion; it
   additionally tracks *map-output availability* (which mapper actor holds which
   scatter output). If a mapper is preempted (local disk lost), the coordinator
   re-queues that **map shard** (deterministic) before its dependent reducers
   proceed — exactly today's transient-failure re-queue, applied to map output.
   This preserves the zephyr assumption "preempted work is re-runnable"; it only
   changes the *storage* of the intermediate, which the user explicitly allowed
   ("change how shuffle works completely … a shuffle service, as long as it's
   resilient to preemption without non-preemptible VMs").

### Why this clears 20%

- Eliminates the scatter GCS upload (~26% of shard) — replaced by a local-disk
  write (~10× faster) + a network fetch (faster than the GCS read it replaces).
- Expected ≈ −20-25% of shard time on top of the CPU wins, on a normally-loaded
  preemptible pool.

### Resilience tradeoff (explicit)

- Today: map output is GCS-durable → never recomputed for shuffle-loss, but the
  GCS round-trip is paid **always**.
- Proposed: pay a map **recompute** only when a mapper is actually preempted
  before its reducers have fetched — rare on the interactive band; the saved I/O
  on every non-preempted run dominates. Worst case (pathological preemption)
  degrades toward today's cost, never worse than re-running a map shard.
- Optional hardening: async, best-effort GCS backup of map output so a late
  preemption can fetch from GCS instead of recomputing — strictly better, at the
  cost of restoring (bounded) background GCS writes. Make it a policy knob.

## Scope / phasing (this is a multi-session build, not a one-shot)

1. Local chunk store + `fetch_shard_chunk` RPC on the worker actor; reducer
   fetches from mappers instead of `pl.scan_parquet(gcs)`. Keep GCS as a
   fallback path behind a flag.
2. Coordinator map-output-availability tracking + recompute-on-loss wiring into
   the existing re-queue path; barrier waits for required map outputs to be
   live, not just "completed".
3. Flow control / fetch concurrency; spill very large chunks to local disk
   (bounded), backpressure on reducer fetch.
4. Bench gate: the same per-shard `done in Xs` A/B harness in
   `lib/zephyr/bench/` (parse_shard_times.py) on normalize tier1; target ≥20%.

## Risks

- Mapper longevity: mappers must outlive their reducers' fetches (or recompute).
  Persistent workers already do; needs explicit lifecycle in the coordinator.
- Skewed fetch hot-spots (one mapper, many reducers) — needs fetch fan-out /
  rate limiting.
- Correctness: deterministic recompute requires deterministic map (normalize is;
  `deterministic_hash` + xxh3 id are). Non-deterministic maps would need the GCS
  backup path.

## Already-landed wins (independent of this design; ship now)

- `lib/marin/.../normalize.py`: vectorized batched whitespace compactor
  (byte-exact, 4.1×).
- `lib/zephyr/.../shuffle.py` + `external_sort.py`: pickle proto-5 (+cloudpickle
  fallback), native `pl.struct` sort-key, codec confirmed zstd.
- Net −9.7% normalize shard-time, tested + lint-clean.
