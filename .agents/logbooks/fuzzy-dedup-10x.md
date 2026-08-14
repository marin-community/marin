---
topic: fuzzy-dedup-10x
description: Fuzzy-verification memory-store load optimization
---

# Fuzzy-verification memory-store load: task logbook

## Scope

- Goal: reduce fuzzy-verification memory-store creation time by 10x before the full Datakit run.
- Primary metrics: memory-store load CPU, maximum actor load elapsed, end-to-end verifier elapsed, and peak worker RSS.
- Correctness gates: identical candidate count, stored key count, decompressed text bytes, and verified output records.
- Scale gates: local 10M documents, then 100M, 1B, and the production corpus.
- Constraints: keep independent per-document compressed frames for random lookup; preserve worker recovery from immutable source shards; keep data movement region-local.

## Current TL;DR

- The local sample contains exactly 10,000,000 normalized Focus Crawl documents in 152 Parquet files and occupies 27.56 GB. See [FD10X-001](#2026-08-13-1912---fd10x-001-local-corpus).
- Profiling showed that token folding, splitting, and n-gram construction dominated verification. The selected implementation moves those operations and set intersection into `dupekit_native` and computes Jaccard inputs beside the compressed document store.
- Each Zephyr worker supervises independently addressable store subprocesses. Stage tasks call each child actor port directly; the supervisor recreates and reloads a failed child from immutable source partitions.
- The wheel-backed 99.9M-document gate loaded 32,151,573 retained documents across eight workers and 64 child stores in 1,258.31 CPU-seconds and 261.29 seconds wall. Resident memory was 104.61 GiB. CoreWeave object-store throughput was impaired during the run.
- At 1 GB/s per node, a 10 TB source scan has a 156-second transfer floor on 64 nodes. The current planning range is 30–60 minutes end to end for 20B source documents; a clean 1B gate is still needed to narrow reducer and write time.

## Hypothesis Queue

### Active

- `FD10X-H1`: eight row-group load partitions per worker reduce maximum load elapsed by at least 4x. Next test: sweep 1, 2, 4, and 8 concurrent partitions per worker on the 10M sample.
- `FD10X-H2`: Arrow-native ID filtering removes most Python work from non-candidate rows and reduces load CPU by at least 2x. Next test: compare the current row iterator with `pyarrow.compute.is_in` and Polars prefiltered scans.
- `FD10X-H3`: `ZstdCompressor.multi_compress_to_buffer` reduces selected-text compression CPU by at least 2x while retaining one frame per document. Next test: compare frame bytes and decompressed payloads against the current compressor.

### Blocked

- Full verifier timing needs candidate and MinHash attributes aligned to the selected normalized shards. Resume after the local candidate fixture is materialized from the production candidate lineage.

### Falsified / Dead End

- Reusing compressed Parquet page bytes as document values: the local source is Snappy Parquet, pages hold encoded multi-row column data, and independent keyed lookup requires document boundaries.

### Promoted

- `FD10X-H4`: worker-supervised subprocess stores expose independent CPU parallelism without tying store shard count to Zephyr worker count.
- `FD10X-H5`: native token preparation and store-local intersection remove the dominant Python reducer path.

## Baseline

- Date: 2026-08-13
- Code ref: `6cf9d5c6ef3d97b98db74770b8c1ba7e5df359a9`
- Prior 100B result: 1,513,510 candidate texts, 32 actors, 2,389.60 aggregate load CPU-seconds, 88.22 seconds maximum actor load elapsed. Source: PR #7873 and the predecessor `6854-zephyr-memory-store` logbook.
- Local baseline: pending candidate fixture and first timed load.

## Entry Log

### 2026-08-13 19:12 - FD10X-001 local corpus

- Hypothesis: hash-partitioned normalized shards form a representative local corpus while preserving production Parquet layout.
- Commit Hash: `6cf9d5c6ef3d97b98db74770b8c1ba7e5df359a9`
- Command: `MARIN_PREFIX=s3://marin-us-east-02a/marin .venv/bin/python /tmp/copy_fuzzy_dedup_sample.py`
- Config: Focus Crawl normalized artifact `common-crawl-focus-2026-22_17ce32f9`; SHA-256 shard selection seed `fuzzy-dedup-10m-v1`; target 10,000,000 rows; 16 copy workers.
- Result: selected 152 of 333 normalized shards. The copy contains 9,979,497 rows in 151 byte-identical files plus 20,503 rows in one boundary file, for exactly 10,000,000 rows and 27,556,447,994 bytes. Transfer elapsed was 253.25 seconds. The original files use Snappy; the boundary file uses Zstd level 0.
- Validation: 152 Parquet files, no partial files, exact row sum 10,000,000, schema preserved. A sampled production file has six row groups with globally ascending IDs and disjoint ascending ID statistics.
- Interpretation: row groups are a valid immutable load partition and provide about 912 physical partitions across this sample, enough to test eight load partitions per worker.
- Next action: materialize aligned production candidate IDs, publish the design artifact, and time the control.

### 2026-08-13 19:23 - FD10X-002 candidate fixture and concurrency sweep

- Hypothesis: the existing `load_concurrency` knob can approach the proposed eight-way load speedup without changing the loader.
- Commit Hash: `6cf9d5c6ef3d97b98db74770b8c1ba7e5df359a9`
- Commands: `/tmp/align_focus_candidates.py`; `/tmp/benchmark_current_store_load.py --files 16 --workers 2 --load-concurrency {1,2,4,8}`.
- Config: first 16 selected normalized shards, 1,056,642 documents, 584,125 legacy production candidate members, two independent worker processes, identical current `_candidate_documents` implementation.
- Result:

  | Load concurrency | Max actor elapsed | Aggregate load CPU | Max actor RSS | Elapsed speedup |
  |---:|---:|---:|---:|---:|
  | 1 | 17.71s | 34.86s | 1.27 GB | 1.00x |
  | 2 | 11.86s | 41.20s | 1.36 GB | 1.49x |
  | 4 | 10.32s | 53.39s | 1.69 GB | 1.72x |
  | 8 | 11.31s | 64.59s | 2.21 GB | 1.57x |

- Correctness: every arm loaded 584,125 keys and 1,216,222,871 compressed bytes.
- Interpretation: four concurrent loads are the local latency knee. Eight threads regress elapsed time while using 85% more CPU and 74% more peak RSS than concurrency one. The current Python row and per-document compression loop cannot turn eight-way subsharding into an eight-way speedup.
- Next action: benchmark Arrow-native candidate filtering, one-document native multi-compression, and cluster-aligned multi-document frames before changing the persistent API.

### 2026-08-13 22:05 - FD10X-003 local Iris E2E sweep and CPU profile

- Hypothesis: memory-store loading and serialization dominate verifier wall time after adding independent store shards.
- Commit Hash: `6cf9d5c6ef3d97b98db74770b8c1ba7e5df359a9`
- Config: real local Iris controller, 16 normalized/candidate/MinHash shards, 1,056,642 source documents, 584,125 candidate members, and 298,158 reducer groups. Each arm produced 853 output files with identical record counts.
- Worker sweep: two, four, and eight Zephyr workers completed in 214.94s, 175.93s, and 180.12s. Four workers is the local E2E knee even though eight workers halve maximum store-load latency from 9.58s to 4.98s.
- Fetch sweep at four workers: batch sizes 32, 128, and 512 completed in 178.27s, 175.93-176.31s, and 174.34s. Fetch size is a minor control; 512 is 1% faster than 128.
- Loader treatment: Arrow-native candidate filtering plus `multi_compress_to_buffer` completed in 171.03s versus 174.34s for the current loader. Maximum store load fell from 9.46s to 7.34s, but the E2E gain was only 1.9% because loading is 5% of tuned wall time.
- Profile: a 50 Hz `py-spy --subprocesses` capture attributed 50.4% of shard-process CPU to case folding, splitting, and token 5-gram construction; 13.3% to cold import/introspection; 7.5% to reducer Python; 6.1% to verification intersections and gates; 3.4% to Parquet-to-Python conversion; 1.8% to Zstd decode; and 0.4% exclusive to cloudpickle. Inclusive cloudpickle stacks totaled 3.66 sampled seconds. Profiling inflated wall time from about 174s to 300s, so only the attribution is used.
- Framing treatment: replacing shard subprocesses with inline actor execution was slower at matched process concurrency: eight one-CPU workers took 207.01s inline versus 179.92s with subprocesses. Inline execution also raised aggregate CPU from 552.1s to 678.4s and peak RSS from 720MB to 990MB. Actor and task work need separate processes; a resident child-process pool remains plausible, but pickle removal alone is not a material opportunity.
- Caveat: 281,192 of the 298,158 local reducer groups are singletons because the selected Focus subset omits members in other production sources or shards. Loading density is representative, but verifier tuning needs a fixture restricted to groups with at least two locally selected members.
- Artifact: `http://127.0.0.1:7878/s/j3l0t8cq/artifacts/profile`
- Next action: build the multi-member fixture, optimize the collision-safe token n-gram representation, and rerun the E2E control/treatment on local Iris.

### 2026-08-13 20:45 - FD10X-004 collision-safe n-gram construction

- Hypothesis: replacing the per-position Python slice/tuple generator with C-level `zip` over shifted token lists reduces the dominant preparation cost without changing representation or decisions.
- Commit Hash: `6cf9d5c6ef3d97b98db74770b8c1ba7e5df359a9` plus the working-tree treatment.
- Fixture: 16 Focus shards filtered to the 17,669 clusters with at least two locally selected members; 303,636 documents and 426,629 direct comparisons. This avoids the singleton-heavy runtime artifact in FD10X-003.
- Microbenchmark: on 25,000 real candidate texts, shifted `zip` produced exactly equal `(token count, frozenset[tuple[str, ...]])` values and improved median preparation time from 8.436s to 6.126s, or 1.38x.
- E2E result: the current implementation completed in 137.00s and 229.84 aggregate worker CPU-seconds. Shifted `zip` completed in 119.25s and 198.18 CPU-seconds, reducing wall time 13.0% and CPU 13.8% (1.15x E2E speedup).
- Correctness: both arms reported 303,636 candidate and cluster members, 17,669 clusters, 426,629 direct comparisons, 853 accepted duplicates, and identical decision and histogram counters.
- Validation: `tests/processing/classification/deduplication/test_fuzzy_verification.py` passed all 12 tests.
- Next action: profile the optimized multi-member run and use the new distribution to rank native text preparation, resident child processes, and framing work.

### 2026-08-14 00:18 - FD10X-005 wheel-backed 100M gate

- Hypothesis: eight Zephyr workers with eight independently addressable store subprocesses each reproduce the direct-tree 100M load result when the native extension is installed from a published wheel.
- Native package: `marin-dupekit-native==0.1.7.dev175081582358367` from `https://github.com/marin-community/marin/releases/download/dev-wheels/index-5653bc28d1cceb38.html`.
- Iris job: `/power/fuzzy-dedup-100m-gate8-wheel-20260814-0009` on `cw-us-east-08a`; eight workers, eight stores per worker, 64 GB RAM per worker, eight execution slots per worker.
- Result: 99,900,825 source documents; 32,151,573 retained documents; 64 store subprocesses; 1,258.31 aggregate load CPU-seconds; 261.29 seconds load wall; 232.11 seconds slowest store; 104.61 GiB resident. The prior direct-tree attempt measured 1,262.09 CPU-seconds, 250.26 seconds wall, and 104.86 GiB resident.
- Validation: the coordinator and all eight workers installed the exact published wheel; all 64 child actors opened their own ports; Stage 0 started with 524 tasks. The first attempt retried after an incomplete object-store metadata read, confirming the outer job retry gate.
- Caveat: CoreWeave object storage remained impaired. Use this gate for package, topology, retained-memory, and aggregate-CPU validation. Do not use its elapsed scan/reduce result as the production bandwidth estimate.
- Local recovery gate: 133 focused tests passed, including a behavior test that terminates a store child during a direct compute request and verifies supervisor-driven reconstruction and retry.
- Next action: finish the current Stage 0 observation, push the cleanup checkpoint, then run a clean 1B gate when object-store throughput recovers.
