---
topic: zephyr-memory-store
issue: https://github.com/marin-community/marin/issues/6854
description: Read-only Zephyr memory store and verified fuzzy-dedup experiments
---

# Zephyr memory store: Task Logbook

## Current TL;DR

- The approved design is a context-owned, read-only actor store over an
  existing partitioned Dataset. The caller supplies the stable key hash; store
  construction preserves shards and validates their routing instead of adding
  a `group_by`.
- Fuzzy-dedup candidate clusters remain candidate discovery. Every removal
  must pass PR #7831's direct comparison with the canonical; cluster sampling
  may only retain a cluster early.
- The generic store passes its focused local and Iris tests. A pickled handle
  served later Zephyr pipelines, and a lookup blocked through a forced owner
  preemption and returned the original value after reconstruction. Dedup parity
  and medium-scale CoreWeave runs remain open.

## Scope

- Goal: Add a picklable, restartable Zephyr `MemoryStore` and evaluate it as a
  full-text lookup path for fuzzy-dedup verification.
- Primary metrics: marker parity, construction CPU/time, lookup throughput,
  actor peak RSS and balance, shuffle bytes, and recovery behavior.
- Constraints: Read-only; reuse existing partitioning; no hidden `group_by`;
  no child-job federation pinning; regional S3 only on CoreWeave; no cluster
  restart; per-document similarity remains the deletion boundary.
- Coordinating issue: [#6854](https://github.com/marin-community/marin/issues/6854)
- Design: [Weaver artifact](https://loom.oa.dev/s/xi8wu13a/artifacts/design)
- Experiment prefix: `ZKV`

## Current Baseline

- Code refs: PR #7145 at `a55236a5661ac15aed1bb5b5abf5971b0754f282`;
  PR #7831 at `1a392275f64965db5bb5ae59ef35d25b7e06ee5e`.
- PR #7831 100B verifier: 1,513,510 candidate members, 1,007,455
  comparisons, 27,179 accepted markers, 20m14s, 6,587.96 worker CPU-seconds,
  9.67 GB peak worker memory, and 31.75 billion candidate-text characters.
- Largest candidate cluster: 104,490 members.

## Hypothesis Queue

### Active

- `ZKV-004`: Store-backed fuzzy verification matches PR #7831 exactly while
  reducing text in the cluster shuffle. Next test: compose the verifier branch
  and establish synthetic parity.
- `ZKV-005`: A deterministic large-cluster sample can reject known false
  clusters cheaply without creating false removals. Next test: compare its
  retained-marker subset and rated-pair recall; keep disabled by default.

### Blocked

- `ZKV-004` medium run: blocked until synthetic store-backed verifier parity.
- `ZKV-005`: blocked until unsampled store parity is established.

### Falsified / Dead End

- None.

### Promoted

- `ZKV-001`: Four existing source shards routed correctly to two actors;
  construction rejected a mismatched hash and did not add a shuffle stage.
- `ZKV-002`: A cloudpickled handle served a later Zephyr map pipeline on both
  local and local-Iris backends.
- `ZKV-003`: A lookup remained pending while its owner reconstructed after a
  forced Iris preemption, then returned the original value from that owner.

## Decision Log

- 2026-07-31: Use `ZephyrContext.load_memory_store()`, not
  `Dataset.cache(mode=MEMORY)`. Evidence: the API owns execution, actor, and
  backing-storage lifetime and exposes indexed lookup rather than cached scans.
- 2026-07-31: Require `hash_key` from the caller and validate existing shards;
  do not repartition during store construction. Evidence: design-review
  feedback and the common case of already partitioned stage outputs.
- 2026-07-31: Do not set `target_cluster` on child jobs. Evidence: Iris keeps
  every child on its federated parent's peer.
- 2026-07-31: Sampling cannot authorize deletion. Evidence: similarity
  thresholds are not transitive across connected components.
- 2026-07-31: Actors execute the input's shard-local reader/map plan directly;
  the store does not write an intermediate Parquet copy. Evidence: direct
  loading passed list-source, Parquet-source, and local-Iris behavior tests.

## Negative Results Index

- Word shingles alone still measured a 52.149% semantic false-positive rate in
  the issue #6854 audit.
- Per-source canonical selection did not prevent a large false cluster whose
  canonical already came from the affected source.
- Longest-first canonical selection measured only 51.91% precision in PR
  #7831's rated-pair audit.

## Entry Log

### 2026-07-31 20:55 UTC - ZKV-000 design gate and implementation base

- Hypothesis: A persistent shared Zephyr context makes a read-only actor store
  feasible without changing fuzzy candidate formation.
- Commit Hash: `a55236a5661ac15aed1bb5b5abf5971b0754f282`
- Command: `git merge --ff-only origin/pr/7145`
- Config: PR #7145 shared-pool head; caller-owned hash routing; inherited
  federation placement.
- Result: Design artifact revision 2 approved. The research branch now points
  at the latest PR #7145 head with no merge commit.
- Interpretation: Implement the generic store and its observable behavior
  before composing the open fuzzy-verification branch.
- Next action: Inspect current plan/shard APIs and write the smallest public-API
  tests for partition validation, lookup order, and pickle reuse.

### 2026-07-31 21:30 UTC - ZKV-001/002 generic store contract

- Hypothesis: Existing physical shards can become actor-owned lookup
  partitions without a store shuffle, and the resulting handle can cross a
  later Zephyr execution by value.
- Commit Hash: `a55236a5661ac15aed1bb5b5abf5971b0754f282` plus uncommitted store changes.
- Commands:
  - `uv run pytest tests/test_memory_store.py -q --tb=short` from `lib/zephyr`
  - `uv run pytest tests -q` from `lib/fray`
  - `uv run pytest tests -q` from `lib/zephyr`
  - isolated rerun of `test_simple_map_integration[iris]`
- Config: Two store actors; stable `key[0]` routing; list and Parquet readers;
  deterministic msgspec keys; cloudpickle values; 1 MiB encoded-data budget.
- Result: The focused store suite passed 7/7. Fray passed 83/83. The full
  Zephyr suite passed 367 tests, with three local-Iris integration timeouts
  under full-suite load; the first failed case passed alone in 27.79 seconds.
  The multi-backend store test cloudpickled its handle and queried it from a
  later Zephyr pipeline on local and Iris backends.
- Negative result: The first Iris fixture used top-level helpers from
  `tests.test_memory_store`, which the task workspace could not import. A
  by-value lambda isolated the product behavior. The next run found lookup
  fan-out resolving handles inside fresh driver threads; starting Fray's
  native async calls in the caller context fixed that real context-loss bug.
- Interpretation: A store actor can replay a shard-local Dataset plan directly.
  No backing prefix, manifest, or intermediate Parquet write is required.
  Shuffle and join outputs must be persisted and reloaded before construction.
- Next action: Revise the Weaver design to the direct-load model, snapshot the
  generic implementation, then compose PR #7831 for synthetic verifier parity.

### 2026-07-31 21:43 UTC - ZKV-003 owner recovery

- Hypothesis: A lookup can wait through reconstruction of its owning actor
  without routing to another partition or converting the outage to `KeyError`.
- Commit Hash: `a55236a5661ac15aed1bb5b5abf5971b0754f282` plus uncommitted store changes.
- Command: `uv run pytest tests/test_memory_store.py -q --tb=short` from
  `lib/zephyr`.
- Config: Two local-Iris store actors; actor 0 forced to `PREEMPTED`; its
  replacement constructor held on a cross-process sentinel while a lookup was
  issued; 15-second store recovery timeout.
- Result: 10 focused tests passed in 39.74 seconds. The lookup stayed pending
  until the reconstruction gate opened, then returned `zero`. The task attempt
  ID increased. A separate actor-boundary fake confirmed that
  `ActorUnavailableError` is retried while `ValueError` propagates immediately.
- Interpretation: Endpoint-name handles and typed transient retry cover the
  intended immutable actor restart model. Context shutdown also removes local
  endpoints, and partial local actor-group construction now rolls back actors
  that had already registered.
- Next action: Commit the generic store checkpoint and compose PR #7831 for
  store-backed verifier parity.
