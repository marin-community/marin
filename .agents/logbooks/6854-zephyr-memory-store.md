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
  preemption and returned the original value after reconstruction.
- The store-backed fuzzy verifier passes all 90 safe dedup tests. It keeps only
  candidate text in actors, shuffles metadata without text, and uses bounded
  reducer lookups. Medium-scale CoreWeave runs remain open.

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

- `ZKV-005`: A deterministic large-cluster sample can reject known false
  clusters cheaply without creating false removals. Next test: compare its
  retained-marker subset and rated-pair recall; keep disabled by default.

### Blocked

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
- `ZKV-004`: The store-backed verifier preserved all synthetic marker,
  canonical, exact-copy, empty-shard, and counter behavior across batch size
  one and multiple worker counts. All 90 safe dedup tests passed.

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

### 2026-07-31 21:58 UTC - ZKV-004 local verifier parity

- Hypothesis: Candidate documents can stay in a shard-preserving actor store
  while the cluster shuffle carries metadata only, without changing any fuzzy
  removal decision.
- Commit Hash: `d103673ad` plus uncommitted store-backed verifier changes.
- Commands:
  - `uv run pytest tests/processing/classification/deduplication/test_verify_fuzzy_dups.py -q`
  - `uv run pytest tests/processing/classification/deduplication -q`
  - targeted `./infra/pre-commit.py` over the eight changed verifier and call-site files
- Config: `(file_idx, id)` keys hashed by `file_idx`; two local actors; one-row
  lookup batches in behavior tests; explicit actor byte and recovery budgets.
- Result: The focused verifier passed 7/7 and the safe dedup suite passed
  90/90 with five integration cases excluded by repository defaults. Expected
  sparse marker rows, canonical choice, equal-ID delegation, empty outputs,
  and validation failures were unchanged. Store counters reported three
  candidate items over two actors in the representative behavior test.
- Interpretation: Full text is not needed in the cluster shuffle. Fetching the
  canonical once and members through ordered `get_many()` batches preserves the
  verifier's direct-comparison boundary.
- Next action: Snapshot and push the treatment revision, then compare it with
  PR #7831 on the same medium CoreWeave inputs using per-stage finelog metrics.

### 2026-07-31 22:32 UTC - ZKV-004 federated correctness smoke

- Hypothesis: The pickled store handle and metadata-only verifier preserve the
  baseline result when their actors and Zephyr workers are child jobs of a
  federated Iris entrypoint.
- Commit Hashes: baseline `e2902dc9e`; treatment `25214d424`.
- Jobs:
  - `/loom/zephyr-kv-baseline-0p1b-20260731-v2`
  - `/loom/zephyr-kv-treatment-0p1b-20260731-v1`
- Config: CoreWeave `cw-us-east-02a`; 0.1B Datakit sample; shared persisted
  MinHash/candidate prefix; 32 Zephyr workers; 32 store actors; 2 CPU and 8 GiB
  per worker/actor; 128-key lookup batches.
- Result: Both jobs succeeded without retries or preemptions. The independent
  inspection matched all eight comparisons over 13 candidate members in five
  clusters; both revisions emitted zero verified markers. Every Zephyr stage
  processed identical item and input-byte counts. Baseline worker CPU was
  35.38 seconds and treatment worker CPU was 41.19 seconds; this tiny input is
  intentionally a correctness smoke, and store startup/RPC overhead dominates.
  Peak worker RSS fell from 471,683,072 to 409,092,096 bytes.
- Negative result: The first control entrypoint failed before pipeline startup
  because scoped environment setup omitted `marin-dupekit`. Explicitly syncing
  `marin-core` and `marin-dupekit` fixed the launch.
- Interpretation: Federation inheritance works without child pinning. All 32
  store actors loaded on the federated peer, served the later worker job, and
  terminated with the context. The 0.1B sample is too sparse for a performance
  claim.
- Next action: Run the same A/B on the 100B candidate population and require a
  complete marker-by-marker equality check against the control artifact.

### 2026-07-31 22:34 UTC - ZKV-005 100B control launched

- Hypothesis: Removing candidate text from the verification shuffle reduces
  worker CPU and peak RSS on the 1.5M-candidate workload while preserving every
  output marker.
- Commit Hash: baseline `5b875ae27`.
- Job: `/loom/zephyr-kv-baseline-100b-20260731-v1`.
- Config: CoreWeave `cw-us-east-02a`; 100B Datakit sample; 64 workers; complete
  inspection disabled in favor of an exact persisted-marker comparison after
  the treatment.
- Result: Running. The discovered 100B prefix held prior verifier results but
  not the current testbed's full MinHash/candidate layout, so the control is
  materializing that shared discovery cache before verification. Those stages
  are excluded from the verifier A/B, and the treatment will consume the exact
  resulting candidate artifact.
- Next action: Monitor discovery through a clean terminal verifier execution,
  then launch the treatment on the same prefix.

### 2026-07-31 22:43 UTC - ZKV-005 control stopped to share workers

- Observation: `StepRunner(max_concurrent=8)` started each source's MinHash
  `ZephyrContext` without an advertised host pool. PR #7145 therefore gave each
  concurrent context its own coordinator `*-pool` and `*-workers-a0` job rather
  than packing their tasks onto one worker group.
- Result: The user authorized stopping
  `/loom/zephyr-kv-baseline-100b-20260731-v1` after 8 minutes 16 seconds. Iris
  reported `killed`, zero execution failures, and no completed verifier result.
- Interpretation: The jobs were independent pipeline attempt zeroes, not
  retries. This layout multiplies environment startup, coordinator, and idle
  worker overhead, so it is unsuitable for the performance comparison.
- Change: The testbed now scopes `StepRunner` inside one 64-worker
  `PoolMode.HOST` context. All MinHash, candidate, and verifier contexts inherit
  its coordinator through the current Iris job environment; federation needs
  no explicit child pinning.
- Next action: Apply the same orchestration change to both A/B revisions and
  relaunch the 100B control under a fresh output prefix.

### 2026-07-31 23:01 UTC - ZKV-005 shared coordinator sizing

- Hypothesis: One standing pool can serve the eight concurrent MinHash stages
  without creating a coordinator and worker group for every source.
- Commit Hash: baseline `fd3322a1d`.
- Job: `/loom/zephyr-kv-baseline-100b-20260731-v2`.
- Result: Iris showed exactly one `zephyr-fuzzy-verification-testbed-pool` and
  one 64-task worker group. Multiple MinHash execution IDs made progress and
  completed source steps on those workers. After 6 minutes 57 seconds, the
  coordinator's default 1 GiB task was OOM-killed with exit 137 while eight
  pipelines were active; it had no preemption or worker failure beforehand.
- Interpretation: Sharing removes repeated worker environments but concentrates
  the active pipelines' plans, task queues, results, counters, and worker RPC
  state in one coordinator. The lightweight single-pipeline default is not an
  adequate request for this eight-pipeline testbed.
- Change: The testbed now requests a non-preemptible coordinator with 1 CPU and
  4 GiB RAM. The standing-pool documentation now calls out aggregate
  coordinator sizing and shows that request in its Datakit example.
- Recovery: The failed pool was terminal, so the parent was stopped to release
  its step locks. Successful MinHash artifacts remain under `candidates-v2` and
  will be reused by the corrected run.
- Next action: Relaunch the control on the same candidate prefix, verify the
  larger coordinator remains healthy, and continue through baseline verifier
  completion.
