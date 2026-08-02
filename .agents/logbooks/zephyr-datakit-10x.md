---
topic: zephyr-datakit-10x
issue: https://github.com/marin-community/marin/issues/7885
description: Benchmark and optimize representative Zephyr datakit pipelines toward a 10x throughput improvement.
author: Marin
---

# Zephyr Datakit 10x: Research Logbook

## Scope

- Goal: Reduce end-to-end processing time by 10x on representative datakit pipelines without changing output semantics.
- Primary metrics: per-stage active task time, CPU time, bytes and records per second, peak memory, task retries, and end-to-end wall time after excluding scheduler wait.
- Constraints: benchmark with real data locally before medium-scale Iris runs; use `cw-us-east-02a` or `cw-rno2a` at batch priority for remote runs; preserve safe local pytest marker defaults; do not move large data across regions.
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/7885
- Experiment prefix: `Z10X`
- Shared tags: `zephyr-10x`, `datakit`

## Current TL;DR

- Research and the first real-data local baseline use untouched commit `d1d081a33`.
- The current code already contains Arrow batch reads, PyArrow expression lowering, Polars-based shuffle internals, and separate map/reduce resource configurations. It still materializes Python rows at the default reader, shuffle adapter, reducer, and writer boundaries.
- Direct DataFrame shuffle ingestion has the best existing matched evidence: PR #7200 reduced tier-2 Map→Scatter CPU from 293.6s to 120.6s (2.43x). Fixed-resource worker packing in PR #6996 did not improve compute time.
- The proposed first milestone is a real-data local harness plus direct Arrow batch writing and an explicit `map_batches` API. The research artifact defines 10K/100K/1M-row local sizes and tier-shaped remote gates.
- The first implementation milestone now measures 1.76x lower wall time, 1.75x lower process CPU, and 16.8% lower RSS growth at 1M rows with exact logical-output parity.
- A 10x end-to-end gain is a north star, not the current estimate. Near-term evidence supports a 1.5–2.5x pipeline target; individual row-heavy kernels may improve by 10x.

## Current Baseline

- Date: 2026-08-02
- Code ref: `d1d081a33`
- Corpus: 10,000 rows from the first ten row groups of FineWeb-Edu conversion `92cece42bcce787ee4af4619ab449fe48d86230d`; local input 28,551,851 bytes.
- Row path, five fresh-process repetitions: median wall 0.744s, median process CPU 0.694s, median RSS growth 122,515,456 bytes, 13,446 input rows/s.
- Output: 4,320 rows, 12,691,805 bytes, semantic digest `640a9d64eeeff2e58c1517f8e70c1fea664239db3f8b167f40d87dd235b0d881`.

## Hypothesis Queue

### Active

- `Z10X-001`: Packing several Zephyr task subprocesses into each Iris-hosted actor may reduce Iris task count and amortize fixed worker costs, but controlled prior results show no compute win when total resources are fixed. Next test: compare 1, 4, 8, and 16 task slots per actor with a fixed 16-64-slot fleet, including startup and preemption cost.
- `Z10X-002`: Preserving `pyarrow.RecordBatch` or `polars.DataFrame` chunks across fused map/filter/write operations will remove Python row conversion and improve throughput on schema-stable stages. Next test: benchmark identity, projection, filter, and derived-column transforms with row and chunk APIs.
- `Z10X-003`: Lowering supported transforms into Polars expressions will outperform equivalent Python callbacks, especially for text-length, null, projection, and scalar predicates. Next test: measure matched semantics on a representative normalized-text shard.
- `Z10X-004`: Stage-specific chunk size and worker/process topology will outperform one global setting because normalize, shuffle, and tokenization have different CPU, memory, and I/O profiles. Next test: derive a small factorial sweep from tier canary resource definitions.
- `Z10X-005`: Reusing spawned subprocesses will amortize interpreter imports, JIT, and model loading on short shards without `fork` hazards. Next test: measure fixed per-shard startup before designing a runner.

### Blocked

- None.

### Falsified / Dead End

- Worker packing as a compute-speed optimization at fixed CPU/RAM: PR #6996 found parity or a slight regression. Packing remains useful for reducing Iris replicas and controller work.
- `joblib` as a general cross-process model cache: issue #7120 found that native tokenizer/Numba state is not picklable and imports/JIT remain per subprocess. A persistent process is the relevant alternative.

### Promoted

- None.

## Decision Log

- 2026-08-02: Use per-stage Finelog task statistics for remote A/B decisions; wall time remains a secondary signal because Iris queueing and worker provisioning add noise.
- 2026-08-02: Require output parity and retry/error checks before accepting throughput improvements.
- 2026-08-02: Track the research in GitHub issue #7885 and this append-only logbook; publish architecture tradeoffs separately as a Weaver artifact.
- 2026-08-02: Define 64-way execution as task slots, not Iris actor replicas. The topology sweep holds 64 CPU fixed and compares 64×1, 16×4, 8×8, and 4×16 actor/slot shapes.
- 2026-08-02: Do not run the canonical tier-3 Nemotron data on the allowed CoreWeave clusters; its source is pinned to `europe-west4`. Use tier-3 shard geometry with region-local data instead.
- 2026-08-02: Keep the first code milestone distinct from open PR #7200: implement and benchmark Arrow batch map/write first, then coordinate productionization or replacement of #7200 rather than duplicating it.
- 2026-08-02: Make msgpack-based routing hash, per-shard membership, and per-shard order parity hard gates for any native scatter path.
- 2026-08-02: Run the fixed-resource topology comparison at batch priority on non-preemptible workers; measure packed-actor retry blast radius in a separate preemptible condition.

## Negative Results Index

- `Z10X-N01`: Fixed-resource actor packing did not make stages faster in PR #6996.
- `Z10X-N02`: The tier-2 Polars shuffle port did not improve end-to-end wall time in PR #5963, showing that tier-1 gains do not generalize automatically to skewed inputs.
- `Z10X-N03`: `joblib` could not share the loaded Luxical model across subprocesses in issue #7120.
- `Z10X-N04`: Size-aware MinHash shard ordering improved the modeled 103-shard makespan by less than 1%; the production files are already nearly uniform.
- `Z10X-N05`: Fusing MinHash and LSH into one native transformation was 1.9% slower after the allocation-free n-gram change; the fused API was discarded.

## Entry Log

### 2026-08-02 00:26 UTC - Z10X-000 research prologue

- Hypothesis: The current post-Fray Zephyr implementation retains enough Python row materialization and fixed process topology to leave a multi-fold local improvement available, with additional scale gains from worker multiplexing.
- Commit Hash: `d1d081a33`
- Command: `weaver summary`; code and GitHub/Echo discovery commands will be recorded in the background-research entry.
- Config: local checkout at `origin/main`; no benchmark data selected yet.
- Result: Found prior Zephyr performance work in `.agents/projects/20260430-zephyr-performance/`, merged stage-specific resource support in PR #6996, merged Polars shuffle internals in PR #5963, and an open DataFrame-native scatter proof of concept in PR #7200.
- Interpretation: The 2026-04 proposal is useful prior art, but several recommendations are now implemented or obsolete. The new research must profile the current execution path before ranking changes.
- Next action: audit current code, canary definitions, GitHub history, and Finelog benchmark tooling; then choose local baseline stages and sizes.

### 2026-08-02 - Z10X-001 background research and current-state audit

- Hypothesis: Row materialization remains the largest tractable overhead after the merged Polars shuffle internals, while packing primarily improves orchestration density.
- Commit Hash: `d1d081a33`
- Sources: current Zephyr/Fray/datakit code; `.agents/projects/20260430-zephyr-performance/`; Echo repository, GitHub, and wiki indexes; PRs #5814, #5859, #5963, #6996, #7145, and #7200; issues #7120 and #7686; Apache Arrow, Polars, and DataFusion documentation.
- Result: `load_parquet(batch_mode=True)` already yields RecordBatches, but non-scatter stage output uses pickle chunks, shuffle wraps each item in a cloudpickle payload after Python key evaluation, and the writer rebuilds tables with `from_pylist`. Fray represents an actor group as one Iris job with N replicas. Tier 3 already packs 16 map tasks per actor for several stages.
- Prior measurements: Parquet row-group/batch exact-dedup example 16m→6m; tier-1 Polars shuffle 4671s→3442s (1.36x), tier-2 no gain; direct DataFrame scatter 293.6s→120.6s Map→Scatter CPU (2.43x); shared pools 1542s→880s only in a startup-bound 100M smoke; fixed-resource packing parity/slight regression.
- External evidence: Arrow RecordBatch is the bounded streaming unit; Polars recommends expressions and lazy/streaming execution; DataFusion documents Python object conversion as one of the slowest UDF paths. These sources support the representation choice, not a numeric uplift claim.
- Interpretation: Implement the shortest complete batch path first—RecordBatch read, batch callback/native expression, direct batch writer—then generalize the proven native scatter approach. Benchmark topology independently at fixed fleet resources.
- Next action: publish `.agents/projects/20260802-zephyr-datakit-10x/research.md` as a Weaver artifact and request Claude Opus review before implementation.

### 2026-08-02 - Z10X-002 Claude Opus research review

- Hypothesis: An independent review will catch incorrect architecture assumptions or benchmark gates before they become implementation work.
- Commit Hash: `d1d081a33`
- Review artifact: https://loom.oa.dev/s/pok3gck3/artifacts/opus-review
- Result: Opus verified every cited code path and all eight prior-work measurements. It required explicit coordination with open PR #7200, bit-for-bit stable routing and per-shard order gates, a schema/empty-batch contract, non-preemptible topology comparisons, and a clearer separation between the 10x final evidence threshold and the approximately 1.9x illustrative representation-only ceiling.
- Incorporation: Revision 2 limits the first implementation to the distinct batch map/write path; treats #7200 as an existing prototype to land/harden or explicitly replace; removes persistent-process work from the first required matrix; marks 3–6x speculative; verifies DupeKit RecordBatch input/output; and documents the repeated corpus as invalid for connected-components or dedup-ratio benchmarks.
- Interpretation: The first milestone is now bounded enough to implement without duplicating active work. A 10x end-to-end claim remains gated on repeated full-pipeline evidence and likely requires algorithmic/materialization changes beyond this first tranche.
- Next action: publish research revision 2 and its issue update, then build the real-data L1 harness and record the untouched row-path baseline before changing writer/API code.

### 2026-08-02 - Z10X-003 untouched 10K row-path baseline

- Hypothesis: A fused row-wise filter/map/write stage provides a stable real-data baseline for measuring the cost of Python row materialization at small scale.
- Commit Hash: `d1d081a33`
- Command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 10000 --modes row --repeat 5`
- Config: FineWeb-Edu conversion `92cece42bcce787ee4af4619ab449fe48d86230d`, first ten row groups, columns `id,text,score`; filter `score >= 3`; derive `text_chars`; one local Zephyr thread; each repetition in a fresh subprocess.
- Result: Wall seconds 0.744, 0.744, 0.663, 0.712, 0.777 (median 0.744); process CPU seconds median 0.694; RSS growth median 122,515,456 bytes; 4,320 output rows and 12,691,805 output bytes.
- Semantic gate: digest `640a9d64eeeff2e58c1517f8e70c1fea664239db3f8b167f40d87dd235b0d881`, computed over canonical 4,096-row Arrow batches so Parquet row-group layout does not affect parity.
- Interpretation: Startup and imports are material at 10K rows, so this size is a correctness and fixed-cost probe rather than the primary throughput gate.
- Next action: measure untouched 100K and 1M row paths, then add failing behavior tests for direct RecordBatch map/write.

### 2026-08-02 - Z10X-004 untouched 100K row-path baseline

- Hypothesis: Scaling the same corpus and transform to 100K rows will reduce fixed-cost dominance enough to expose Python row conversion and writer costs.
- Commit Hash: `d1d081a33`
- Command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 100000 --modes row --repeat 5`
- Config: Deterministic tenfold repetition of the pinned 10K corpus with suffixed IDs; otherwise identical to `Z10X-003`.
- Result: Wall seconds 10.366, 9.973, 25.577, 23.661, 6.828 (median 10.366); process CPU seconds median 8.900; RSS growth median 365,535,232 bytes; 43,200 output rows and 126,895,804 output bytes; median 9,647 input rows/s.
- Semantic gate: digest `a5d554e45924c994206d4c3f025932a5151d2b23c84d67d8a8a4b481c2dcf756` was identical in all repetitions.
- Interpretation: The result shows substantial host-noise variance, including two 2.3–3.4x slower samples. Use medians for the initial local comparison and require an interleaved row/batch A/B after implementation; do not interpret the extremes as code effects.
- Next action: measure the untouched 1M row path with five fresh-process repetitions before changing library code.

### 2026-08-02 - Z10X-005 untouched 1M row-path baseline

- Hypothesis: At 1M rows, fixed startup cost will be small enough for this stage to serve as the primary local throughput and memory gate.
- Commit Hash: `d1d081a33`
- Command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 1000000 --modes row --repeat 5`
- Config: Deterministic hundredfold repetition of the pinned 10K corpus with suffixed IDs; otherwise identical to `Z10X-003`.
- Result: Wall seconds 59.971, 56.024, 57.295, 59.236, 68.337 (median 59.236); process CPU seconds median 62.289; RSS growth median 419,147,776 bytes; 432,000 output rows and 1,268,648,680 output bytes; median 16,882 input rows/s.
- Semantic gate: digest `a572e6383444e3a88db93b311737f7f34743635dd3183af3affe6b0f7269f1b0` was identical in all repetitions.
- Interpretation: Unlike the 100K probe, four of five wall-time samples fall in a 7% band. Use the 1M median as the primary untouched baseline and preserve the 68.3s sample in the distribution rather than trimming it.
- Next action: add behavior tests for the public RecordBatch map/write contract, demonstrate that they fail on untouched code, then implement the direct path.

### 2026-08-02 - Z10X-006 batch map/write behavior and 10K A/B

- Hypothesis: Fusing Arrow batch read, filter/map, and direct batch writing will reduce CPU and memory while preserving logical output.
- Commit Hash: working tree based on `d1d081a33`
- Test command: `uv run --project lib/zephyr pytest lib/zephyr/tests/test_writers.py::test_write_parquet_file_accepts_record_batches lib/zephyr/tests/test_writers.py::test_write_parquet_file_preserves_typed_empty_record_batch lib/zephyr/tests/test_writers.py::test_write_parquet_file_rejects_record_batch_schema_drift lib/zephyr/tests/test_dataset.py::test_dataset_map_batches_writes_parquet -q`
- Behavior result: All four tests failed on untouched code at the row-only writer or missing `Dataset.map_batches`; all four pass after the direct RecordBatch implementation.
- Benchmark command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 10000 --modes row,batch --repeat 5`
- Result: Interleaved median wall 0.735s row versus 0.539s batch (1.36x); process CPU 0.689s versus 0.478s (1.44x); RSS growth 146,636,800 versus 111,845,376 bytes (23.7% lower).
- Semantic gate: Both arms produced 4,320 rows with schema `id,text,score,text_chars` and digest `640a9d64eeeff2e58c1517f8e70c1fea664239db3f8b167f40d87dd235b0d881` in every repetition.
- Interpretation: Even the startup-sensitive size shows a repeatable gain, and the treatment's direct Parquet layout is slightly smaller without changing logical output. Larger sizes determine whether conversion savings grow with data volume.
- Next action: run the same interleaved A/B at 100K and 1M rows.

### 2026-08-02 - Z10X-007 100K row versus batch A/B

- Hypothesis: The batch-path advantage will grow at 100K rows as per-row Python conversion becomes a larger fraction of stage work.
- Commit Hash: working tree based on `d1d081a33`
- Command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 100000 --modes row,batch --repeat 5`
- Result: Interleaved median wall 6.975s row versus 4.066s batch (1.72x); process CPU 6.808s versus 4.219s (1.61x); RSS growth 371,568,640 versus 311,525,376 bytes (16.2% lower).
- Semantic gate: Both arms produced 43,200 rows with digest `a5d554e45924c994206d4c3f025932a5151d2b23c84d67d8a8a4b481c2dcf756` in every repetition.
- Interpretation: The matched interleaved comparison is materially less noisy than the untouched 100K series and confirms that the gain comes from the batch representation rather than the row-writer refactor. The batch arm was faster in every pair.
- Next action: run the primary 1M interleaved A/B, then lint and broaden behavior regression coverage.

### 2026-08-02 - Z10X-008 1M row versus batch A/B

- Hypothesis: The direct Arrow path will retain at least the 100K throughput gain on the primary 1M local gate without increasing peak memory.
- Commit Hash: working tree based on `d1d081a33`
- Command: `uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 1000000 --modes row,batch --repeat 5`
- Result: Interleaved median wall 60.328s row versus 34.329s batch (1.76x); process CPU 63.912s versus 36.446s (1.75x); RSS growth 422,920,192 versus 351,944,704 bytes (16.8% lower). The batch arm sustained 29,129 median input rows/s versus 16,576 for rows.
- Semantic gate: Both arms produced 432,000 rows with schema `id,text,score,text_chars` and digest `a572e6383444e3a88db93b311737f7f34743635dd3183af3affe6b0f7269f1b0` in every repetition.
- Interpretation: The milestone delivers a stable 1.75–1.76x improvement for the representative map/filter/write stage, not 10x end-to-end. Batch wall times span only 2.8%, while one row outlier raises row variance; the median CPU result independently matches the wall-time uplift.
- Next action: finish regression coverage, run required checks, snapshot the milestone in a PR, then define the first medium-scale tier-shaped A/B from this API.

### 2026-08-02 - Z10X-009 first-milestone validation and review

- Commit Hash: `79ad144d7`
- Mechanical checks: `./infra/pre-commit.py --changed-files --fix` passed Ruff, Black, Pyrefly, license, AST, conflict, whitespace, and Markdown checks.
- Focused tests: 102 writer/Dataset tests passed before commit; after advisory-review cleanup, 24 writer and batch-map tests passed and a one-repeat 10K benchmark preserved the pinned digest in both arms.
- Full Zephyr suite: 347 passed, 4 deselected, 1 expected failure, and `test_sorted_merge_join_inner_basic_integration[iris]` timed out at the suite-wide 60-second limit. The exact Iris-backed test passed alone in 42.32s; the changed local writer/Dataset path is not involved in the join.
- Advisory review: Applied immutable benchmark configuration, a shared score threshold/result marker, a named corpus-digest result, concrete byte-count types, split parent/child entry points, and shorter internal docstrings. Kept row and RecordBatch accumulation separate because schema widening plus Python conversion and exact batch-schema enforcement have different contracts; sharing their small flush block would add callback indirection and couple those semantics.
- Next action: commit the targeted review cleanup, push the branch, and open the first milestone PR.

### 2026-08-02 - Z10X-010 federated medium-scale harness

- Hypothesis: A fixed 64-slot CoreWeave comparison can separate the representation gain from actor-job packing effects while preserving fast iteration.
- Commit Hash: working tree based on `e87bd2a2f`
- Federation probe: `uv run iris --config lib/iris/config/marin.yaml job run --target-cluster cw-us-east-02a --job-name zephyr-10x-federation-probe --cpu 0.1 --memory 1GB --disk 1GB --priority batch --no-preemptible --no-sync -- python -c ...`
- Probe result: `/loom/zephyr-10x-federation-probe` succeeded through the Marin controller. The peer job resolved `MARIN_PREFIX=s3://marin-us-east-02a/marin`, confirming that benchmark inputs and outputs can stay in the selected CoreWeave region.
- Harness config: stage one pinned 100K-row FineWeb-Edu Parquet object inside `cw-us-east-02a`, then present it as 64 Zephyr input shards. Run `row:4`, `batch:4`, and `batch:1` sequentially: 16 actor jobs with four subprocess slots for the matched representation A/B, plus 64 single-slot actor jobs for the packing comparison. Each arm uses 64 total CPU slots, 4 GiB RAM and 1 GiB disk per task slot, non-preemptible worker resources, and inherited batch priority.
- Gates: all 64 output footers contribute to file, row, byte, and schema checks; a fixed-4,096-row logical digest of a representative shard verifies values and order across arms. Zephyr `cpu_time_total` and memory statistics from Finelog are the primary performance signals; the harness wall time is context only.
- Caveat: Reusing the same pinned real-data object avoids a large staging copy and is valid for map/filter/write CPU measurement, but it does not represent source-key diversity and must not be used for deduplication-ratio or shuffle-skew conclusions.
- Next action: validate the distributed path with a two-shard federated smoke, then launch the 64-shard A/B through the same Marin federation route.

### 2026-08-02 - Z10X-011 CoreWeave 64-slot representation A/B

- Hypothesis: The Arrow batch map/write path will reduce aggregate stage CPU and peak shard memory at 6.4M rows, while 16 actors with four slots will reach the stage barrier faster than 64 single-slot actors at the same fleet size.
- Commit Hash: `624ee4a00`
- Job: `/loom/zephyr-10x-chunk-medium-v1`, submitted to the `marin` controller with `--target-cluster cw-us-east-02a --priority batch --no-preemptible`; root and all child coordinators/workers succeeded without retries or preemptions.
- Config: 64 logical shards, each reading the same pinned 100K-row FineWeb-Edu corpus from the region-local S3 store; 6.4M input rows and 2.7648M output rows per arm. Every arm requested 64 CPU slots, 4 GiB RAM and 1 GiB disk per task slot.
- Representation result: `row:4` execution `20260802-021748-8f6a0fb8` used 234.47 CPU-seconds, 717,950,976 peak bytes, and 467,111,808 average bytes. `batch:4` execution `20260802-021849-7cc00933` used 174.29 CPU-seconds, 663,384,064 peak bytes, and 444,455,488 average bytes. The batch arm reduced CPU by 25.67% (1.35x efficiency), peak memory by 7.60%, and average memory by 4.85%.
- Topology result: `batch:1` execution `20260802-021951-1d26f591` used 171.37 CPU-seconds, 670,048,256 peak bytes, and 465,542,912 average bytes versus 174.29, 663,384,064, and 444,455,488 for `batch:4`. Compute cost is within 1.7%, but the observed stage barrier was 16.94s for 64 one-slot actors versus 7.52s for 16 four-slot actors; logs show the larger pool ramping from 3 to 46 live workers while work was already completing.
- Semantic gate: Each arm wrote 64 files and 2,764,800 rows with the same schema and representative digest `a5d554e45924c994206d4c3f025932a5151d2b23c84d67d8a8a4b481c2dcf756`. `records_in` and `records_out` matched exactly. Finelog `bytes_processed` differs by output-path string length between row and batch arms and is not a data-byte counter for this fused stage.
- Interpretation: Chunk-native map/write remains a real efficiency win at medium scale, though the gain is smaller than the 1.75x local 1M-row result. Packing four task slots per actor preserves compute efficiency and substantially reduces readiness delay for this short stage. The wall comparison is a topology signal, not a code-efficiency verdict, and needs a broader packing sweep before choosing a default.
- Next action: run a 64-slot `batch:16,8,4,2,1` topology sweep on 10K-row shards in reverse order, then use CPU-neutral stage-barrier and worker-ramp evidence to choose the first recommended packing range.

### 2026-08-02 - Z10X-012 worker-pool ceiling and topology heuristic

- Hypothesis: Packing lightweight tasks into a modest number of Iris replicas controls fixed startup cost without changing steady-state CPU efficiency; excess shards should multiplex through the worker pull loop instead of creating an unbounded Kubernetes task set.
- Commit Hash: working tree based on `624ee4a00`
- Job: `/loom/zephyr-10x-topology-sweep-v1`, federated through `marin` to `cw-us-east-02a` at batch priority with non-preemptible worker resources. All five arms succeeded with no retries or preemptions.
- Config: 64 total CPU slots and 640K logical records per arm; packing order `16,8,4,2,1` subprocess slots per Iris replica, corresponding to `4,8,16,32,64` worker replicas. This intentionally small sweep characterizes fixed overhead only.
- Result: Finelog CPU totals were 34.16, 33.88, 33.35, 33.42, and 32.35 seconds, respectively; compute cost was effectively flat. Stage barriers were 4.11, 4.71, 4.06, 8.16, and 11.85 seconds. All arms produced matching counts, schema, and digest. Four to sixteen process slots avoided the slower large-pool ramp, with no meaningful distinction inside that range.
- User direction: Treat worker startup as an additive cost rather than the dominant long-run optimization. Use reasonable packing heuristics, keep Iris/Kubernetes task counts at roughly 1,000 or fewer, and multiplex additional Zephyr shards through the bounded pool.
- Implementation decision: Replace the advisory 1,024-worker default with a 1,000-replica distributed ceiling that also caps explicit requests. Local execution remains uncapped. Existing workers already pull successive shards, so this changes only control-plane fan-out and does not limit dataset size or shard count.
- Next action: validate and land the ceiling guardrail, then shift benchmark effort to large, region-local canary stages and per-record Arrow/native reductions.

### 2026-08-02 - Z10X-013 Arrow-native MinHash local correctness and memory control

- Hypothesis: Keeping each Parquet batch in Arrow through text truncation, DupeKit MinHash, null filtering, bucket casting, and writing will remove Python row materialization without changing MinHash output.
- Commit Hash: `a8cec3276`
- Config: 10,000 unique FineWeb-derived documents; unchanged MinHash parameters; row baseline from `624ee4a00` versus the Arrow batch implementation; fresh local processes.
- Result: The row baseline used 20.11 CPU-seconds, peaked at 804,679,680 bytes RSS, averaged 717,458,637 bytes, and wrote 4,896,672 bytes. The Arrow treatment used 19.94 CPU-seconds, peaked at 455,237,632 bytes RSS, averaged 446,997,299 bytes, and wrote 4,902,337 bytes. CPU was effectively flat while peak RSS fell 43.4% and average RSS fell 37.7%.
- Semantic gate: Both arms produced the same canonical digest, `3014592c2be427e1169845b19a75780c80c2b5bd010c08437e1d34e660c792c9`, over the same logical rows. The full fuzzy-dedup test module passed 71 tests; repository checks including Pyrefly passed.
- Interpretation: This small run is a correctness and memory control, not evidence of a scale throughput gain. MinHash's native hashing work dominates at 10,000 documents, so the production-scale A/B must decide whether avoiding Python conversion matters for sustained CPU or only memory.
- Next action: compare the old row path and Arrow path on the same 103 production shards containing 16,236,550 distinct documents and 41.10 GB of region-local compressed input.

### 2026-08-02 - Z10X-014 production-scale MinHash row versus Arrow A/B

- Hypothesis: Avoiding row dictionaries in the production MinHash attribute stage will reduce sustained CPU and memory, with a larger effect than the 10K control once conversion costs accumulate over millions of documents.
- Commit Hashes: row baseline `624ee4a00`; Arrow treatment `a8cec3276`.
- Jobs: `/loom/zephyr-10x-minhash-large-row-v2` and `/loom/zephyr-10x-minhash-large-batch-v1`, both submitted through the `marin` Iris federation to `cw-us-east-02a` at batch priority with non-preemptible workers. Both completed without retries, failures, or preemptions.
- Config: The same 103 production shards under `sample_100b_8ae7a94f/nemotron_cc_v2/medium_quality`, totaling 41,098,991,064 compressed input bytes and 16,236,550 distinct documents. Both arms used 16 Iris workers, each reserving 5 CPU, 32 GiB RAM, and 5 GiB disk, with one Zephyr task per worker.
- Result: Baseline execution `20260802-025956-765fbe35` used 17,346.21 CPU-seconds, 2,350,424,064 peak bytes, 2,048,113,527 average bytes, and 1,222.30 stage seconds. Arrow execution `20260802-032241-3c0973e9` used 16,914.10 CPU-seconds, 992,964,608 peak bytes, 851,460,279 average bytes, and 1,187.41 stage seconds. Arrow reduced CPU 2.49% (1.026x), peak memory 57.75%, average memory 58.43%, and stage elapsed 2.85%.
- Semantic gate: Both arms counted 16,236,550 documents, 422,150,300 buckets, and 867 truncated texts. A region-local validation job found 103 matching basenames, identical schemas, and 16,236,550 rows in each output; the first, middle, and last output tables were exactly equal. Output sizes were 7,707,007,971 row bytes and 7,710,290,996 Arrow bytes.
- Interpretation: Python row conversion is not a primary CPU cost in this compute-heavy stage. The Arrow path's value is the large memory reduction, which makes denser subprocess packing safe. Finelog reports about one busy CPU per nominal five-CPU worker in both arms, consistent with DupeKit's current single-threaded Rust loops.
- Next action: hold the Iris worker count at 16 and run four one-CPU, 4-GiB MinHash subprocess slots per worker to measure steady-state utilization and wall-time uplift independently of worker startup.

### 2026-08-02 - Z10X-015 production-scale MinHash subprocess packing

- Hypothesis: MinHash is single-core per shard, so multiplexing four subprocesses inside each Iris worker will recover otherwise idle reserved CPUs without materially increasing aggregate CPU or Kubernetes fan-out.
- Commit Hash: `a8cec3276`.
- Job: `/loom/zephyr-10x-minhash-large-packed4-v1`, submitted through the `marin` Iris federation to `cw-us-east-02a` at batch priority with non-preemptible workers. The job completed without retries, failures, or preemptions.
- Config: The same 103 shards, 41.10 GB compressed input, 16,236,550 documents, and Arrow MinHash implementation as `Z10X-014`. Both treatments used 16 Iris workers. The one-slot arm reserved 5 CPU and 32 GiB per worker; the packed arm reserved 4 CPU and 16 GiB per worker and admitted four one-CPU, 4-GiB Zephyr subprocesses per worker, for 64 concurrent shard tasks behind 16 Kubernetes workers.
- Result: Packed execution `20260802-034647-60a91c1b` used 16,955.22 CPU-seconds, 988,962,816 peak bytes per subprocess, 844,256,864 average bytes, and 342.88 stage seconds. The one-slot Arrow arm used 16,914.10 CPU-seconds and 1,187.41 stage seconds. Packing improved stage elapsed 3.46x while changing aggregate CPU by +0.24%; peak and average per-subprocess memory were slightly lower. End-to-end root-job duration improved from 21m06.65s to 6m57.43s (3.03x), with setup and teardown included.
- Semantic gate: Both arms counted 16,236,550 documents, 422,150,300 buckets, and 867 truncated texts. Region-local validation found 103 matching basenames, identical schema and row totals, byte-identical aggregate output size of 7,710,290,996 bytes, and exact equality for the first, middle, and last output tables.
- Interpretation: This is a steady-state utilization gain, not a startup-overhead result. The Iris/Kubernetes task count stays at 16 while Zephyr multiplexes 103 shards through 64 process slots. The flat CPU total shows that packing changes elapsed time and reserved-resource efficiency rather than algorithmic work.
- Next action: make single-CPU task packing the MinHash default heuristic, retaining explicit overrides, then validate the configuration behavior and rerun the relevant local tests and repository checks.

### 2026-08-02 - Z10X-016 production shard-ordering negative result

- Hypothesis: Scheduling the largest MinHash shards first will reduce the second-wave tail after packing raises concurrency from 16 to 64 process slots.
- Commit Hash: `a458d6087`.
- Job: `/loom/zephyr-10x-minhash-shard-skew-v1`, submitted through the `marin` Iris federation to `cw-us-east-02a` at batch priority. The analysis read only region-local object metadata and Parquet footer counts.
- Config: The same 103 production shards used in `Z10X-014` and `Z10X-015`. Compare lexical input order with longest-processing-time-first proxies based on compressed object bytes and row counts, at 16 and 64 process slots.
- Result: Nearly every shard is approximately 400 MB compressed, with one small final shard. Size- or row-aware ordering improved modeled makespan by only 0.02–0.21% at 16 slots and 0.28–0.93% at 64 slots.
- Interpretation: The observed tail is the unavoidable second wave of 103 similarly sized shards over 64 slots, not meaningful input skew. Listing metadata or reading 103 footers in the scheduler would add complexity and remote calls for less than 1% modeled benefit.
- Next action: Do not add size-aware ordering for this workload. Reduce per-example native CPU and intermediate allocation instead.

### 2026-08-02 - Z10X-017 allocation-light native MinHash kernel

- Hypothesis: Hashing UTF-8 slices instead of allocating one Rust `String` per character n-gram, and collapsing punctuation and whitespace without an intermediate string or regular-expression replacement, will reduce sustained MinHash CPU without changing signatures.
- Commit Hash: working tree based on `a458d6087`.
- Implementation: Keep whole-string lowercase semantics, remove the regex and punctuation-table pass, use byte windows for ASCII text, and use character-boundary byte slices for Unicode n-grams. The public CleanText, MinHash, and MinHashLSH transformations remain separate. A proposed fused MinHash+LSH transformation was 1.9% slower after the allocation improvements and was removed.
- Local result: On 10,000 unique FineWeb-derived documents with the production 286-permutation, 26-band configuration, the released kernel used a 19.1667-second median CPU time and the optimized kernel used 16.0367 seconds, a 16.33% reduction (1.195x). CleanText alone improved from 1.41376 to 0.460507 median CPU-seconds (3.07x).
- Compatibility gate: An initial per-character lowercase optimization passed narrow golden tests but changed Greek final sigma and therefore some production buckets. The first 16.24M-document treatment was discarded. Restoring whole-string lowercase produced byte-identical CleanText, MinHash, and LSH output on 100,000 deterministic mixed-Unicode documents, including the failing sigma context. The regression is covered explicitly in the unit tests.
- Job: `/loom/zephyr-10x-minhash-native-wheel-v3`, execution `20260802-054752-0b85bdf1`, submitted through the `marin` Iris federation to `cw-us-east-02a` at batch priority. It used the same 103 shards, 41.10 GB compressed input, 16,236,550 documents, 16 Iris workers, and four one-CPU Zephyr subprocess slots per worker as the packed baseline. The job completed with no failures, retries, or preemptions.
- Production result: The optimized stage used 14,928.82 CPU-seconds versus 16,955.22, a reduction of 11.95% (1.136x). Stage elapsed fell from 342.88 to 303.63 seconds (1.129x); end-to-end root duration fell from 6m57.43s to 6m14.40s (1.115x). Peak subprocess memory changed by +0.62% and average memory by -1.33%. Against the original one-slot row stage, the cumulative stage elapsed improvement is 4.03x and CPU is 13.94% lower.
- Semantic gate: Both arms counted 16,236,550 documents, 422,150,300 buckets, and 867 truncated texts. Region-local validation found 103 matching basenames, identical schemas and per-file row counts, byte-identical aggregate output size of 7,710,290,996 bytes, and exact equality for the first, middle, and last output tables.
- Interpretation: The sustained CPU result confirms that per-n-gram native allocation is a meaningful cost at production scale, while the 64-slot packing result remains the larger elapsed-time gain. The accepted implementation preserves context-sensitive Unicode lowercase behavior instead of trading semantics for another small speed increment.
- Next action: Commit the native kernel milestone and update PR #7888. Continue coordinating the larger native-scatter opportunity with PR #7200 rather than duplicating its implementation.

### 2026-08-02 - Z10X-018 100B S3 benchmark topology checkpoint

- Hypothesis: A bounded set of concurrent Zephyr pools can expose source-level parallelism on the full 100B sample without creating thousands of Iris/Kubernetes tasks, while stage CPU metrics identify the remaining serialization and native-code costs.
- Commit Hash: `18ab481c1`.
- Input: `sample_100b_8ae7a94f` in region-local S3 contains 115 sources, 768 normalized shards, and 103,716,988 records. Hugging Face corpus ingress and dependency setup are excluded from all controlled timings.
- Topology probe: One 243-worker pool reserved 3,888 vCPU and completed global exact dedup in 80.03 seconds using 47,028.72 CPU-seconds. One 60-worker pool completed the same stage in 92.20 seconds using 12,579.71 CPU-seconds. The smaller pool traded 15.2% wall time for 73.3% less CPU, leaving capacity for three concurrent source pipelines; four 60-worker pools reserve at most 3,840 vCPU and 240 Kubernetes tasks.
- Scope correction: The full reference DAG is not a clean Zephyr benchmark on this fleet because Luxical imports a CUDA-enabled Torch build on CPU-only pods and fails on missing `libcublasLt`. The S3-native benchmark entry point therefore reuses only global exact dedup, per-source tokenization, native MinHash, and global fuzzy dedup.
- User direction: Treat the 60-worker/four-stream layout as a reasonable heuristic and stop spending primary effort on packing. After the 100B checkpoint, focus on individual-stage serialization, Python allocation, Arrow retention, native lowering, and writer backpressure.
- Follow-up: Rebase the branch onto PR #7145 (`zephyr: share worker pools across pipelines`) and measure its single shared pool across concurrent task streams. That PR reports an 880-second shared-pool versus 1,542-second dedicated-pool 100M smoke, but it remains secondary to per-stage optimization.
- Active job: `/loom/zephyr-10x-datakit-100b-18ab-v5`, federated to `cw-us-east-02a` at batch priority with four concurrent pools of at most 60 workers × 16 vCPU, 160 GiB RAM, and 32 GiB disk.
- Next action: Wait for the controlled 100B stage report, pause for review, then rank serialization/native/writer hotspots by aggregate CPU and active time before implementing the next change.

### 2026-08-02 - Z10X-019 100B shared-pool scale result

- Hypothesis: PR #7145's shared Zephyr pool can expose all 115 source streams through fewer than 1,000 Iris tasks, while the branch's Arrow and native-stage changes reduce controlled CPU independently of topology.
- Commit Hash: pooled experiment `413912986`, based on PR #7145 plus PR #7888. Reproducible branch: `weaver/infra-zephyr-10x-pool7145-20260802`.
- Control-plane correction: A 116-pipeline saturation attempt exposed that Fray's Iris backend dropped `ActorConfig.max_concurrency`, leaving the actor server at 32 threads. Blocking `run_pipeline` calls then starved worker registration, heartbeats, and pulls. The experiment branch now propagates actor concurrency through the Iris actor server and includes a 40-blocking-call regression test that leaves capacity for a ping. The failed attempt is excluded from performance results.
- Job: `/loom/zephyr-10x-datakit-100b-pool-max-4139-v5`, federated through `marin` to `cw-us-east-02a` at batch priority. One shared pool used 243 Iris actors × 16 vCPU, with four 4-vCPU subprocesses per actor: 972 logical task slots, 3,888 worker vCPU, and 116 concurrent pipelines. All 243 actors registered; all 231 exact/tokenize/MinHash pipelines completed with no retries, preemptions, organic failures, or coordinator-limit rejects. The root succeeded in 7m43.99s; subsequent coordinator and worker kills were expected context teardown.
- Input and parity: 115 sources, 768 S3 shards, 103,716,988 documents, 110,202,288,980 output tokens, and 2,696,641,688 MinHash buckets. A region-local artifact gate found all 231 expected artifacts and exact record parity across exact dedup, tokenization, and MinHash.
- Exact dedup: The dedicated-pool baseline used 18,545.79 CPU-seconds, 388.87 summed stage seconds, and 4.225 GB peak memory. The concurrent shared-pool treatment used 6,154.92 CPU-seconds, 68.73 stage seconds, and 1.956 GB peak memory: 66.81% less CPU, 5.66× lower stage elapsed, and 53.7% lower peak memory.
- Tokenization: The baseline used 142,932.06 CPU-seconds, 16,723.25 summed stage seconds, and 2.275 GB peak memory. The pooled Arrow-native treatment used 168,062.24 CPU-seconds, 12,619.62 summed stage seconds, and 3.245 GB peak subprocess memory. Stage work consumed 17.58% more CPU despite the lower summed elapsed, so the current four-thread tokenizer shape is a throughput trade rather than an efficiency win. Treatment processor timing, which excludes initialization, was 117,471.43 aggregate seconds or 938,120 tokens per processor-second; the older baseline does not expose a directly comparable initialization/processing split.
- MinHash: The baseline used 103,051.74 CPU-seconds, 10,033.89 summed stage seconds, and 1.107 GB peak memory. The pooled native treatment used 87,066.79 CPU-seconds, 8,641.88 summed stage seconds, and 1.111 GB peak memory: 15.51% less CPU and 13.87% lower summed elapsed at equivalent peak memory.
- Total efficiency: Exact, tokenization, and MinHash CPU changed from 264,529.59 to 261,283.95 seconds, a 1.23% reduction. This is the primary cross-topology efficiency result. The raw first-controlled-stage to last-MinHash/tokenizer span changed from 10,589.94 to 368.74 seconds, or 28.72×, but this is an operational topology result rather than the A/B verdict because the span includes tokenizer model initialization and setup that cannot be separated from the older baseline.
- Utilization: The pooled run averaged 708.58 active CPU cores and peaked at 1,513.31, despite reserving 3,888 vCPU. At peak the coordinator reported 1,328 inflight tasks and no queue. The remaining throughput limit is work intensity, per-stage parallelism, and I/O, not Iris task startup; reserving more CPU alone will not produce another comparable gain.
- Output hygiene: The benchmark now routes every generated StepSpec through `marin_temp_bucket(ttl_days=7, ...)`, keyed by the required run tag. Region-local cleanup removed the 232 baseline stage prefixes, two earlier exact-dedup prefixes, and all three shared-pool experiment roots after validation. The source sample was excluded by explicit path assertions.
- Next action: Pause at the requested gate. Treat shared pooling as the scale topology, keep Iris replicas below 1,000, and focus the next implementation pass on tokenizer CPU, native multi-row operations, and I/O/backpressure rather than further worker packing.
