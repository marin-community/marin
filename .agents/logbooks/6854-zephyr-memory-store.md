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

### 2026-07-31 23:21 UTC - ZKV-005 bounded shared-pool finalization

- Hypothesis: The shared pool's OOM is amplified by one 32-thread final-result
  executor per concurrently completing pipeline, rather than only by durable
  coordinator state.
- Job: `/loom/zephyr-kv-baseline-100b-20260731-v3` at baseline commit
  `b93305102`.
- Observation: Coordinator thread samples oscillated from about 43 durable
  threads to 303-317 threads while eight pipelines materialized results. At 14
  minutes 29 seconds, Iris reported 317 threads and 1.55 GiB RSS. The corrected
  4 GiB coordinator remained healthy with zero failures or preemptions, more
  than twice the failed run's 6 minute 57 second lifetime.
- Change: Commit `8d22aab14` gives the coordinator one reusable 32-thread
  final-result executor. All concurrent pipelines share that bound, and the
  executor is closed with the coordinator. The same patch is baseline commit
  `2a8a8a691`.
- Validation: The coordinator execution and shared-context suite passed 89/89
  tests. The regression test saturates all 32 materialization threads twice and
  verifies that the second pipeline reuses the exact first thread set.
- Interpretation: Pool reuse is still the efficient topology, but its shared
  coordinator must bound aggregate concurrency and request memory for all live
  pipelines. The explicit 4 GiB request remains useful headroom even after the
  thread bound.
- Next action: Let the recovery run finish populating the candidate cache, then
  run both verifier revisions with the bounded executor and identical pool
  resources.

### 2026-08-01 00:05 UTC - ZKV-006 reuse the reported 100B candidates

- Observation: The issue reproduction already has a complete candidate
  artifact at
  `s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1/baseline/dedup`:
  115 sources, 1,513,510 candidate members, and 505,876 clusters.
- Change: The testbed accepts that artifact directly. Its importer normalizes
  legacy absolute source paths and rewrites the nested v1 candidate schema to
  the current flat schema without rerunning MinHash. The same-region migrated
  artifact is durable at
  `s3://marin-us-east-02a/marin/user/loom/datakit/zephyr-kv-6854/100b/baseline-direct-v3/candidate-artifact`.
- Validation: Three importer regressions cover source normalization, collision
  rejection, and v1 schema flattening.
- Interpretation: Both verifier arms can now consume exactly the candidates
  reported in issue #6854. Candidate discovery is outside the measured A/B.

### 2026-08-01 00:30 UTC - ZKV-007 external-sort memory accounting

- Observation: Direct verifier attempts v3-v7 repeatedly OOM-killed the two
  hottest reduce shards. Increasing worker containers from 8 GiB to 16 GiB did
  not fix the failure. A configured 4 GiB task budget was ignored because
  Zephyr planned shuffle fan-in from the container cgroup limit.
- Change: Integration commit `43a8def14` makes shuffle planning and external
  sort use the active task memory budget, falling back to the cgroup limit when
  no task budget is advertised. The baseline equivalent is `c0ea24b4a`.
- On-cluster validation: The corrected reducer logged a 4.3 GB task budget and
  fan-in 31 instead of a 17.2 GB container budget and fan-in 126. With a 1 GiB
  diagnostic budget, pass 1 reduced fan-in to 9 and exposed a second defect in
  pass 2: 66 concurrent runs containing approximately 356,637-byte serialized
  records computed a safe single-digit read batch, then clamped it upward to
  100 and exceeded its own 0.3 GB buffer budget.
- Change: Integration commit `0adc81786` and baseline commit `5af4f21f1` remove
  the unsafe 100-item floor. One item per run is now the minimum. The regression
  uses the production run count and record width and fails under the old floor.
- Validation: The focused shuffle suite passed 23/23 tests; the combined
  shuffle, execution, and memory-store suite passed 102/102 tests. Targeted
  lint, formatting, and type checks passed.
- Resource decision: The 1 GiB task budget was diagnostic, not a target
  production allocation. At the user's direction, the measured A/B now uses
  64 GB worker containers with a 60 GB verifier task budget, reserving 4 GB for
  the worker process and runtime. Integration commit `45c8e0130` and baseline
  commit `e2544ff01` carry identical sizing.
- Next action: Complete the 100B baseline and treatment under those identical
  resources, verify exact persisted-marker equality, and compare finelog CPU
  and memory metrics.

### 2026-08-01 01:24 UTC - ZKV-008 100B direct verifier A/B

- Jobs:
  - baseline `/loom/zephyr-kv-baseline-100b-direct-20260801-v8`, execution
    `20260801-003704-09c1a674`;
  - treatment `/loom/zephyr-kv-treatment-100b-direct-20260801-v1`, execution
    `20260801-005433-f35a303d`.
- Result: Both jobs succeeded with zero failures and preemptions. The treatment
  verified 27,179 duplicates from 1,513,510 candidate members in 505,876
  clusters, then compared every persisted marker and logged `Verified output
  matches 27179 reference markers` against the baseline artifact.
- Topology: Each arm used one shared 64-worker Zephyr pool. The treatment added
  one intentional 32-task MemoryStore actor job; these are store shards rather
  than additional Zephyr pools. The actors directly reused the 768 source
  partitions through the caller-supplied file-index hash.
- Store load: The 32 actors loaded all 1,513,510 documents and 31,955,129,644
  serialized bytes in parallel. Load CPU summed to 2,399.51 seconds, the
  largest actor held 1,365,131,612 serialized bytes, and the slowest actor was
  ready in 96.43 seconds.
- Stage finelog comparison:
  - map/scatter: baseline 2,622.24 CPU-seconds and 63.31 seconds elapsed;
    treatment 34.64 CPU-seconds and 8.86 seconds elapsed because document I/O
    moved to the store actors;
  - cluster reduce/scatter: baseline 1,642.26 CPU-seconds and 627.91 seconds;
    treatment 2,174.67 CPU-seconds and 583.85 seconds;
  - output reduce: baseline 47.47 CPU-seconds and 11.18 seconds; treatment
    48.61 CPU-seconds and 11.31 seconds.
- Accounting: The baseline stages used 4,311.97 CPU-seconds. Treatment stages
  plus the one-time actor load used 4,657.43 CPU-seconds, an 8.0% increase.
  The hot reduce critical path fell 7.0%, while complete root-job time rose
  from 800.71 to 857.97 seconds (+7.2%) because store startup was not hidden.
- Memory: The completed hot-reduce stat fell from 13,420,138,496 to
  1,303,760,896 bytes. Live cgroup inspection found about 17 GB on the baseline
  hot reducer and 866,050,048 bytes on the corresponding treatment reducer.
  These are directional rather than final peak figures: the periodic sampler
  used a ContextVar that is absent in its child thread, so only the final
  main-thread sample populated resource counters.
- Zephyr fix: Commit `6a503a78d` writes periodic samples through the explicit
  shard context. The runner regression now requires positive RSS in a RUNNING
  finelog row; 9 tests passed with one expected xfail. Baseline commit
  `6b8480e2e` carries the identical fix for the next measured pair.
- Lookup analysis: 474,388 clusters contain 2-3 documents, while only 248
  exceed 127 documents. Commit `145f8a404` co-fetches the representative with
  the first member batch, removing a serial lookup round for almost every
  cluster, and raises the skew testbed batch from 128 to 1,024 for the
  104,490-document tail. The focused verifier suite passed 11/11 tests.
- Next action: Measure treatment v2 at
  `/loom/zephyr-kv-treatment-100b-direct-20260801-v2`; if batching improves the
  reduce stage without changing markers, rerun the corrected-telemetry baseline
  and final treatment pair.

### 2026-08-01 01:40 UTC - ZKV-009 lookup batching result

- Job: `/loom/zephyr-kv-treatment-100b-direct-20260801-v2`, execution
  `20260801-012609-7722e0db`.
- Result: The job succeeded in 13 minutes 23.65 seconds with zero failures or
  preemptions. It verified 27,179 duplicates and matched every persisted marker
  from the baseline artifact.
- Comparable work: All three stages matched treatment v1 exactly on items and
  bytes. The cluster reduce processed 27,947 output items and 12,752,368 bytes
  in both runs.
- Batch result: Co-fetching the canonical document with the first 1,024-record
  batch shortened the noisy reduce critical path from 583.85 to 567.97 seconds,
  but increased primary CPU work from 2,174.67 to 2,299.26 CPU-seconds (+5.7%).
  Corrected periodic telemetry measured a 5,140,877,312-byte peak and
  974,127,181-byte average, versus the incomplete v1 peak sample.
- Decision: Do not use a 1,024-document production batch. Keep the canonical
  co-fetch, but retain the bounded 128-document batch for the final local-
  representative comparison. The observed 5.14 GB peak also confirms that the
  final 64 GB worker / 60 GB task budget has ample headroom.
- Next action: Finish porting the store beneath the current local-representative
  verifier and run the corrected baseline/treatment pair against its 27,203-
  marker reference output.

### 2026-08-01 05:55 UTC - ZKV-010 current verifier A/B and PR split

- Code refs: The corrected combined implementation is revision `15b1ba9e4`.
  The review branches are #7872 for the Zephyr/Fray/Iris prerequisites, #7873
  for the generic memory store, and #7874 for Datakit integration. Worker reuse
  from #7145 is not included in those branches.
- Executions: Control `20260801-015637-790a9637`; encoded memory-store
  treatment `20260801-021947-4fbda315`; corrected output validation
  `20260801-054058-69da3970` under root job
  `/loom/zephyr-kv-current-treatment-100b-20260801-v4`.
- Comparable work: Both measured arms read 1,513,510 candidate members and
  emitted the same 27,203 markers. Every comparable finelog stage matched on
  items and bytes processed.
- Result: Moving candidate text out of the shuffle reduced peak cluster-reducer
  RSS from 16.25 GB to 2.56 GB (-84.2%) and comparable stage CPU from 6,131.44
  to 4,268.61 seconds (-30.4%). The 32 encoded store actors added 2,414.70
  CPU-seconds, making single-consumer total CPU 6,683.31 seconds (+9.0%).
- Correctness: The corrected run completed in 22m03s with no failures or
  preemptions. An in-region record comparison found 27,203 expected markers,
  no unexpected or missing markers, and no changed records.
- Decision: Keep the memory-store data path for its reducer-memory reduction,
  but remove eager msgspec/cloudpickle encoding before presenting it as an
  efficient generic primitive. Split the implementation into the three PRs
  requested by the user and keep worker reuse independent.

### 2026-08-01 21:58 UTC - ZKV-011 native-object rerun and conclusion

- Code refs: PR #7874 revision `4f62e48f`; benchmark-only revision
  `ae2931eae`, which adds the Iris listener fix later merged as #7880.
- Job: `/loom/zephyr-kv-native-treatment-100b-20260801-v2`, execution
  `20260801-212729-d53e91d7`, output
  `s3://marin-us-east-02a/marin/user/loom/datakit/zephyr-kv-6854/100b/native-treatment-v2`.
- Config: 32 memory-store actors, 64 Zephyr workers, 64 GB worker requests,
  60 GB stage budgets, 768 source shards, and one worker pool reused across all
  three stages.
- Correctness: All 32 actors became ready and loaded their assigned partitions.
  The pipeline verified 1,513,510 candidate members, emitted 27,203 markers,
  and matched all 27,203 reference records without failures or preemptions.
- Finelog command:

  ```bash
  uv run finelog query marin --format table "
  SELECT execution_id, stage_name, cpu_time_total, items, bytes_processed,
         mem_peak_bytes_max, mem_bytes_avg, cpu_pct_avg, elapsed
  FROM \"zephyr.stage\"
  WHERE status = 'END'
    AND execution_id IN (
      '20260801-054058-69da3970',
      '20260801-212729-d53e91d7'
    )
  ORDER BY stage_name, execution_id
  "
  ```
- Stage result: With identical item and byte counts, encoded versus native CPU
  was 3,105.89 versus 3,088.21 seconds for map/scatter (-0.6%), 61,496.32
  versus 59,232.82 for cluster reduce/scatter (-3.7%), and 33,147.45 versus
  30,963.39 for final reduce (-6.6%). Total stage CPU fell from 97,749.66 to
  93,284.42 seconds (-4.6%). The encoded baseline used the earlier combined
  #7145 stack, so this is directional rather than an isolated serialization
  A/B.
- Store startup: Both implementations loaded 1,513,510 rows. Aggregate load
  CPU changed from 2,399.33 to 2,389.60 seconds (-0.4%), mean actor wall time
  from 78.68 to 78.91 seconds, and maximum wall time from 84.72 to 88.22
  seconds. Parquet scanning dominates distributed startup; native retention
  removes the encoded-memory copy but does not shorten that phase. The isolated
  200,000-row benchmark remains 0.906 versus 0.173 CPU-seconds (-81%).
- Tail diagnosis: Wall time remained 22m03s. A live worker profile put the last
  shard in `verify_prepared_candidate()` at
  `member.ngrams & representative.ngrams`, not in store, RPC, network, or memory
  waits. Shard 495 took 603.68 seconds versus 593.43 seconds in the encoded run.
  Giant token n-gram intersection is the next performance target.
- Incident: The first rerun selected the same released ephemeral port for two
  host-networked actors. #7880 keeps the listener reserved through Uvicorn
  startup and propagates bind/readiness failures. The incident is recorded at
  https://echo.oa.dev/wiki/66; the clean rerun is its 32/32 live validation.
- Publication: PR #7873 and #7874 bodies and comments contain the performance
  result. Weaver design artifact rev 10 contains the complete API and source
  ledger at https://loom.oa.dev/s/xi8wu13a/artifacts/design.
- Cleanup: All benchmark jobs are terminal. Obsolete task worktrees and virtual
  environments were removed; the shared volume recovered from full to 9% used
  with 867 GB available.
- Conclusion: Native objects remove avoidable serialization work and preserve
  exact output, while the distributed load and wall-time bottlenecks are now
  clearly Parquet scan and giant-set similarity computation, respectively.
