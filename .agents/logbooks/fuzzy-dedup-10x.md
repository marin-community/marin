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
- The earlier 30–60 minute planning range covered source scan and load but not the now-observed verification reducer cost. The current clean-run planning estimate is 4–5 hours on 200 workers when calibrated to the production retry; a sample-only one-store extrapolation is 13.9 hours and is retained as the conservative bound.
- The production retry used 8,192 map and reduce shards. Each reducer explicitly discovered every mapper chunk schema from Parquet footers; a sampled 281 MB chunk had a 5.62 MB footer, implying about 46 GB of footer metadata per reducer before payload processing. Main now caps chunks at 512 row groups, and the branch groups source shards to eight pipeline shards per worker.
- Mapper-published schemas remove that explicit reducer footer pass. On the exact 10M-document, 10-worker, 80-shard fixture, pipeline wall fell from 1,272.11s to 984.98s (1.29x) and worker CPU fell 7.7%. All semantic counters and all 52,816 output rows matched exactly.
- A naïve resident n-gram index is not viable: the current collision-safe `TokenNgrams` representation used about 109 KB/document more RSS than compressed storage on 20,000 real candidates. A packed signature plus cold exact-text fallback remains promising because 81.6% of sample comparisons fail the n-gram containment gate.

## Hypothesis Queue

### Active

- `FD10X-H1`: eight row-group load partitions per worker reduce maximum load elapsed by at least 4x. Next test: sweep 1, 2, 4, and 8 concurrent partitions per worker on the 10M sample.
- `FD10X-H2`: Arrow-native ID filtering removes most Python work from non-candidate rows and reduces load CPU by at least 2x. Next test: compare the current row iterator with `pyarrow.compute.is_in` and Polars prefiltered scans.
- `FD10X-H3`: `ZstdCompressor.multi_compress_to_buffer` reduces selected-text compression CPU by at least 2x while retaining one frame per document. Next test: compare frame bytes and decompressed payloads against the current compressor.
- `FD10X-H6`: a packed per-document n-gram signature can reject containment failures before text decompression while staying within 2x of the current store footprint. Next test: prototype packed signatures over 100k candidates and exact fallback for signature passes.

### Blocked

- Full verifier timing needs candidate and MinHash attributes aligned to the selected normalized shards. Resume after the local candidate fixture is materialized from the production candidate lineage.

### Falsified / Dead End

- Reusing compressed Parquet page bytes as document values: the local source is Snappy Parquet, pages hold encoded multi-row column data, and independent keyed lookup requires document boundaries.
- Storing the current `TokenNgrams` object for every candidate: measured resident overhead is about 109 KB/document, which extrapolates to hundreds of terabytes for the production candidate set.

### Promoted

- `FD10X-H4`: worker-supervised subprocess stores expose independent CPU parallelism without tying store shard count to Zephyr worker count.
- `FD10X-H5`: native token preparation and store-local intersection remove the dominant Python reducer path.
- `FD10X-H7`: mapper-written shuffle sidecars carry each chunk's schema so reducers can plan scans without opening every Parquet footer first.

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

### 2026-08-14 19:50 - FD10X-006 production shuffle diagnosis

- Hypothesis: the full retry is slow because the 8,192-way shuffle multiplies Parquet footer discovery across every mapper and reducer.
- Commit Hash: `e1584e625c4c82199312b3125146e9115b529873` rebased onto `fc498ed09e`.
- Iris job: `/rav/fuzzy-dedup-full-prod-balanced-task4-retry3-20260814-055900`; Zephyr execution `20260814-062119-305595af`; 64 workers and 256 concurrent task slots.
- Stage 0 result: 8,192 shards, 5,951,461,965 records, 1,618,783,351,400 payload bytes, 456,850.08 worker CPU-seconds, and 1,985.83 seconds wall.
- Stage 1 observation: 7,275 of 8,192 reducers had completed after about 12.9 hours. Completed tasks had 1,318.9s median and 1,469.6s p95 elapsed, about 1,072.6s median worker CPU, and only about 43 MB mean payload.
- Footer evidence: a sampled 281.2 MB chunk had 8,192 row groups and a 5,617,004-byte footer. Each old reducer opened schemas for about 8,230 chunks, or roughly 46 GB of footer metadata before reading its target payload. See [Echo wiki 150](https://echo.oa.dev/wiki/150).
- Failure mode: later reducer profiles blocked in `MemoryStore.get_many` recovery after worker restarts; the referenced store endpoint refused connections. No production cluster restart or mutation was performed.
- Interpretation: main's 512-row-group cap, branch pipeline-shard grouping, and mapper-published schemas address independent multipliers. The running retry contains neither the complete rebased treatment nor the schema-sidecar change.
- Next action: run a matched 10M-document local A/B and preserve exact output equality.

### 2026-08-14 21:18 - FD10X-007 mapper-published shuffle schemas

- Hypothesis: serializing each chunk schema into its mapper-written sidecar removes the reducer's explicit all-chunk footer discovery pass without changing shuffle output.
- Commit Hash: `e1584e625c4c82199312b3125146e9115b529873` plus the working-tree schema-sidecar treatment.
- Command: `UV_CACHE_DIR=/tmp/marin-fuzzy-uv-cache uv run /tmp/benchmark_focus_local_iris.py --mode e2e --max-shards 152 --zephyr-workers 10 --store-shards-per-worker 1 --load-concurrency 1 --pipeline-shards-per-worker 8 --loader current --stage-runner subprocess --tag <arm>`.
- Config: exactly 10,000,000 normalized Focus documents; 5,537,783 candidate members; 152 source shards grouped into 80 map/reduce shards; ten local Iris workers; identical candidate, MinHash, resource, lookup-batch, and output settings. The control already included main's 512-row-group cap and branch shard grouping.
- Result:

  | Arm | Pipeline wall | Total wall | Zephyr worker CPU | Stage 0 wall | Stage 1 worker CPU | Peak task RSS |
  |---|---:|---:|---:|---:|---:|---:|
  | Footer schema discovery | 1,272.11s | 1,355.96s | 4,717.51s | 79.05s | 4,259.67s | 1.66 GB |
  | Mapper-published schema | 984.98s | 1,067.40s | 4,354.65s | 74.28s | 3,932.76s | 1.68 GB |

- Delta: pipeline wall improved 22.6% (1.29x); worker CPU fell 7.7%; Stage 0 improved 6.0%. A serialized schema for this workload is 344 bytes per chunk.
- Correctness: zero differences across semantic counters; both arms reported 2,234,872 clusters, 4,960,139 direct comparisons, and 52,816 accepted duplicates. Collecting and sorting every output column produced exactly equal 52,816-row frames.
- Validation: 25 shuffle tests passed; the full Zephyr suite passed with 365 tests, three deselections, and one expected failure. The initial sandboxed full-suite attempt failed only where loopback sockets were prohibited; the same suite passed outside the socket-restricted sandbox.
- Interpretation: this is a real but sub-10x local improvement. The production benefit should be larger because the local topology has 80 mapper chunks rather than thousands and uses local files rather than object-store footer GETs.
- Next action: combine this measurement with the live task distribution for the 200-worker estimate; investigate a compact verification index for the remaining compute floor.

### 2026-08-14 21:42 - FD10X-008 n-gram store feasibility

- Hypothesis: replacing resident compressed documents with precomputed collision-safe `TokenNgrams` removes decompression and n-gram construction from store-side comparisons at acceptable memory cost.
- Commit Hash: `e1584e625c4c82199312b3125146e9115b529873` plus the working tree.
- Command: `UV_CACHE_DIR=/tmp/marin-fuzzy-uv-cache uv run /tmp/benchmark_ngram_store.py {compressed,ngrams} --limit 20000`.
- Config: the first 20,000 candidate documents in Focus shard `part-00274-of-00333.parquet`; 135,309,124 total characters; Zstd level 3; verification token 3-grams.
- Result: compressed payload was 41,538,612 bytes and took 0.67s to build. The current `TokenNgrams` objects contained 14,575,136 distinct n-grams, took 3.75s to build, and left 2.60 GB resident versus 0.43 GB for the compressed arm, about 109 KB/document incremental RSS.
- Interpretation: a resident index built from the current object is infeasible. A packed 64-bit signature would require about 5.8 KB/document before offsets and overhead, close to the current store's observed resident bytes per document. It can be used as a safe prefilter with exact cold-text fallback for signature passes. On the 10M fixture, 4,045,404 of 4,960,139 comparisons (81.6%) failed containment and are the target fast path.
- Negative result: a follow-up four-store-per-worker local run was invalidated when the local Iris autoscaler removed three busy slices at its ten-minute idle threshold. Before invalidation it loaded all 5,537,783 documents into 40 subprocess shards in 68.84s and put 40 reducers in flight; its timing is not used for extrapolation.
- Next action: prototype a packed signature with exact fallback and measure bytes/document, rejection parity, and comparison throughput on 100k real candidates before changing the persistent store.

### 2026-08-14 21:45 - FD10X-009 200-worker estimate

- Hypothesis: the matched local run and the observed production retry bound a useful clean-run estimate for 200 workers.
- Commit Hash: `e1584e625c4c82199312b3125146e9115b529873` plus the working tree.
- Scale factor: the production Stage 0 count is 5,951,461,965 records versus 5,537,783 local candidates, or 1,074.70x. The local pool had ten two-CPU workers and one store subprocess per worker. The target pool has about 200 four-CPU workers.
- Sample-only model: scale Stage 0 and Stage 2 by 40x task capacity but Stage 1 by only 20x because one store subprocess per worker owns the delegated comparison work. This projects 0.55 hours map, 13.13 hours verification reduce, and 0.23 hours final reduce, or 13.9 hours pipeline wall before load and startup.
- Live-calibrated model: the 64-worker retry completed 7,275 of 8,192 Stage 1 reducers in 12.9 hours, implying 14.56 hours without the later recovery stall. Scaling workers from 64 to 200 gives 4.66 hours; applying the measured 1.29x schema-sidecar delta gives 3.61 hours. Allowing for map, final reduce, load, and remote variance gives a 4–5 hour clean-run planning estimate.
- Interpretation: 4–5 hours is the operational estimate; 13.9 hours is the conservative Focus-sample bound. Neither establishes the requested 10x reduction. Main's row-group cap and reduced fanout should help more at production scale than the local A/B captures, but that benefit is not assigned an unsupported multiplier.
- Index target: 5.8 KB/document of packed 64-bit signatures extrapolates to about 34.7 TB total, or 174 GB per 200 workers, close to the current store's observed resident bytes per document. Building the current native n-grams for the full candidate count extrapolates to about 23 minutes at 800 cores. A replacement signature store with cold text fallback could therefore fit the target topology and must reduce the 3.6-hour Stage 1 projection to at most about 1.5 hours to make the overall retry 10x faster than the observed 64-worker run.
- Next action: land the independently validated mapper-schema change, then gate the packed-signature design on memory and exact-output parity before changing the production verifier.

### 2026-08-14 23:18 - FD10X-010 bounded n-gram signature gate

- Hypothesis: a fixed 512-byte, 4,096-bit token 3-gram bitmap can reject definite non-subsets before decompressing and preparing store-resident documents while preserving exact verifier decisions.
- Implementation: each distinct n-gram sets four positions derived from its XXH3 hash. A member is rejected only when one of its occupied bits is absent from the representative bitmap. Surviving comparisons use the collision-safe exact `TokenNgrams` implementation. Stored exact representative preparations are released one cluster tuple at a time.
- Config: 15 of the 152 Focus shards, ten local Iris workers, 80 pipeline partitions, 5,537,783-document full-store load, 128-request store batches, and the production verification thresholds.
- Result: 547,436 candidate members generated 397,942 direct comparisons. The bitmap rejected 124,529 comparisons, 31.3% of the comparison stream, in addition to 70,241 member-longer rejections. Exact verification accepted 759 duplicates.
- Signature-only negative result: accepting every bitmap survivor would retain about 202,413 exact-negative comparisons, or 51.0% of the 397,183 exact negatives. The 512-byte bitmap therefore works as a one-sided prefilter and does not meet the 10% signature-only error target. It has zero false rejections for strict set containment by construction.
- Memory: a first full run loaded the complete store but accumulated exact representative preparations across unrelated clusters and exhausted the local host during verification. Grouping each store batch by representative tuple bounds that cache to one cluster at a time. The grouped full smoke is recorded separately after completion.
- Next action: validate full-output parity and measure the grouped store peak; evaluate a 4 KiB capped exact-hash/Bloom hybrid for signature-only decisions.

## Background Research Brief - signature-only verification and batched Parquet scans

**Effort:** medium

**Question:** Can fuzzy verification use only a bounded per-document signature with less than 10% false positives or negatives, and can request bundles replace the resident document store with ordered Parquet scans?

### Prior internal work

- [Echo wiki 121](https://echo.oa.dev/wiki/121) records the fuzzy-dedup audit and the need for full-text verification after MinHash candidate generation. Its sorted-ID and source-range invariants support a merge-join implementation when both inputs are co-partitioned.
- [Marin PR 7866](https://github.com/marin-community/marin/pull/7866) introduced worker-local memory stores to avoid carrying document text through the cluster shuffle and to move comparisons away from reducer tail tasks.
- [Marin PR 4695](https://github.com/marin-community/marin/pull/4695) added external sorting for bounded-memory Zephyr reducers.
- `lib/zephyr/src/zephyr/dataset.py` provides `sorted_merge_join`; both sides must have matching shard ranges and sorted keys. Normalized Parquet row order is not an API guarantee, so source-local ID requests still need sorting or indexing.

### External sources

- The [Apache Parquet concepts guide](https://parquet.apache.org/docs/concepts/) identifies row groups as the unit of parallelization. Sequential bundle processing should therefore coalesce requests per source file and row group.
- [Graf and Lemire, Xor Filters](https://arxiv.org/abs/1912.08258) gives a smaller membership filter than Bloom filters, but an opaque xor filter cannot answer document-to-document set containment unless the member n-grams are still enumerable.
- [RAMBO](https://arxiv.org/abs/1910.02611) and [Bloofi](https://arxiv.org/abs/1501.01941) index membership across many Bloom filters. They help route queries to document sets; they do not replace the final containment decision between two opaque document signatures.

### Empirical and analytical bounds

- A 36,678-document candidate sample has 711 mean and 240/1,212/2,276/8,646 p50/p90/p95/p99 distinct token 3-grams. A fixed 4 KiB record with a 64-byte header can store about 1,344 packed 24-bit hashes, covering about 91% of sampled documents without truncation; 2,048 and 4,096 hashes cover 94.4% and 97.1%.
- A 24-bit sorted-hash list has an approximate one-missing collision probability of 0.006% against 1,000 representative n-grams, 0.030% against 5,000, and 0.060% against 10,000. Exact subset merge over the capped list has no false negatives for documents that fit the cap, apart from treating a hash collision as a false positive.
- A 4 KiB, four-hash Bloom fallback has approximate one-missing false-positive probabilities of 0.005% at 729 n-grams, 4.4% at 5,000, and 24.7% at 10,000. At the sampled p99 of 8,646 n-grams, the one-missing probability is 18.1% and the two-missing probability is 3.3%.
- Documents exceeding a roughly 1,340-hash cap are about 9% of the document sample. Their comparison-weighted fraction is not yet known and is the main uncertainty in the under-10% target.

### Recommended signature-only treatment

- Use a fixed 4 KiB record. Store character count, token count, distinct 3-gram count, line count, saturation and under-tokenization flags, and a 128-bit normalized token-sequence hash in a small header.
- Store sorted packed 24-bit hashes when the document has at most about 1,340 distinct n-grams. Use a 32K-bit Bloom representation for overflow documents, or mark them for exact fallback. The nominal 10-billion-document allocation is 40.96 TB, leaving about 9 TB of the stated 50 TB budget for indexes and worker overhead.
- Character count and distinct n-gram count provide sound early rejections for the current subset rule. Top-word frequencies can reject valid set-subsets and lose ordering information, so they should not control the decision. Token-sequence and line metadata are useful for the rare saturated, under-tokenized, and local-representative gates.
- The expected document-weighted error is below 10% if every capped-list document is treated exactly and all overflow mistakes are charged pessimistically, but that is not yet a comparison-weighted guarantee. Gate the design on one million actual comparison pairs, recording exact and signature decisions by n-gram-count, saturation, and representative kind.

### Batched Parquet scan assessment

- A complete request-bundle plan adds two global exchanges: cluster reducers emit requested IDs partitioned by source file, table workers scan files in ID order, and fetched values return to cluster owners. Dynamic local-representative selection means later comparison requests are unknown until earlier decisions complete.
- The 10M fixture retains 5,537,783 candidate documents and processes 33.16 billion candidate characters. At production scale, shuffling fetched compressed text or 4 KiB signatures would move tens of terabytes. Scanning a file separately for each reducer bundle repeats Parquet I/O; coalescing all bundles before scanning recreates a distributed sort/merge barrier.
- The design is most useful as the second tier of a signature-first verifier: resolve capped signatures beside a persistent source-partitioned index, then group only ambiguous or overflow IDs by source Parquet for one ordered fallback scan. This bounds resident raw-text memory and applies the extra exchange to the fallback fraction rather than every candidate.

### Ranked next experiments

1. Replay one million real comparison pairs through the 4 KiB capped-list/Bloom hybrid and report comparison-weighted false-positive and false-negative rates.
2. Measure a persistent source-partitioned 4 KiB sidecar with memory mapping or direct row-group reads, including per-worker RSS and lookup throughput.
3. Add a two-tier fallback prototype only if the ambiguous comparison fraction is small enough to offset its two extra exchanges.

### 2026-08-14 23:34 - FD10X-011 full bounded-signature smoke

- Hypothesis: the 512-byte bitmap and cluster-scoped representative cache complete the 10M-document run within local memory while preserving every exact output decision.
- Command: `UV_CACHE_DIR=/tmp/marin-fuzzy-uv-cache uv run --no-sync python /tmp/benchmark_focus_local_iris.py --mode e2e --max-shards 152 --zephyr-workers 10 --store-shards-per-worker 1 --load-concurrency 1 --lookup-batch-size 128 --pipeline-shards-per-worker 8 --loader current --stage-runner subprocess --worker-cpu 2 --worker-ram 7g --task-cpu 1 --task-ram 2g --tag bitmap-10m-10w-80shards-grouped`.
- Result: the 5,537,783-document store used 34.91 GB resident and a 35.25 GB peak upper bound. The pipeline completed in 795.45s and the full call in 939.79s. One store reconstructed after unrelated stale local shard processes created host pressure; its reload increased store load from the control's 60.80s and 436.97 CPU-seconds to 123.67s and 998.83 CPU-seconds.
- Fast paths: 1,858,148 comparisons, 37.5% of 4,960,139, were definitive bitmap non-subsets. Character length rejected another 854,417 before decompression. Exact n-gram preparation remained necessary for 2,247,574 comparisons.
- Delta: versus the mapper-schema control, pipeline wall fell from 984.98s to 795.45s, a 19.2% reduction or 1.24x speedup. Combined with the original footer-discovery control, pipeline wall fell from 1,272.11s to 795.45s, a 37.5% reduction or 1.60x speedup. Worker CPU was 4,484.95s versus 4,354.65s; the treatment exchanges CPU and 512 bytes/document for lower reducer critical-path latency.
- Correctness: all semantic decision counters matched the exact control, including 4,045,404 containment rejections, 854,417 member-longer rejections, and 52,816 accepted duplicates. Sorting all output columns by ID produced exactly equal 52,816-row frames.
- Production estimate: applying the measured 1.24x verifier delta to the live-calibrated mapper-schema projection reduces the 200-worker Stage 1 estimate from 3.61 hours to 2.92 hours. Allowing for map, final reduce, store load, and remote variance gives a 3.5-4 hour clean-run estimate. The combined treatment is material but does not establish a 10x end-to-end improvement.
- Memory extrapolation: the observed 6.30 KB/candidate store footprint projects to about 37.5 TB for the production retry's 5.95 billion candidates, or 188 GB per worker at 200 workers. A 4 KiB signature-only replacement would use 24.4 TB, or 122 GB per worker, before indexes and process overhead.
- Next action: publish the exact prefilter treatment as a draft PR and measure the 4 KiB signature-only error rate on the actual comparison stream before replacing exact fallback.
