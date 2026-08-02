# Zephyr chunk-native performance research

## Recommendation

Make Zephyr chunk-native in three increments: accept and preserve Arrow batches through fused map/filter/write stages; route Arrow/Polars batches through shuffle without a cloudpickle payload; then lower a small, explicit Zephyr expression vocabulary into native Polars expressions. Treat Iris worker packing as a separate topology optimization. Packing reduces Iris replicas and controller pressure, but controlled prior measurements do not show a compute-speed benefit when total CPU and RAM are fixed.

The first milestone should target a 2x improvement in the local map/filter/write kernel, with output parity and no memory regression. A 10x speedup is plausible for individual Python-heavy transforms, but is not yet a credible end-to-end estimate for the current standard datakit pipeline. The current tier-1 run spends about 175 seconds downloading before any processing, and its post-Polars processing stages still include MinHash, connected components, joins, and tokenization whose native work is unaffected by removing Python serialization. The evidence supports a 1.5–2.5x near-term end-to-end target. A 3–6x result is speculative and would require several unproven changes to compound: native joins, batch-native bucket expansion, persistent worker processes, and fewer materialized artifacts. Reaching 10x end to end would require reducing or fusing the underlying work, not just changing representation.

Track the experiments and decisions in [issue #7885](https://github.com/marin-community/marin/issues/7885).

## Claude Opus review disposition

[Claude Opus reviewed revision 1](https://loom.oa.dev/s/pok3gck3/artifacts/opus-review) against the exact cited commit and verified every code citation and prior-work figure. This revision incorporates its required changes:

- The first implementation milestone is limited to the distinct batch map/write path. It will not duplicate the open scatter work in PR #7200.
- Stable routing is a hard compatibility gate: any native scatter path must reproduce `deterministic_hash(key) % num_output_shards` and preserve per-shard order.
- The 10x goal remains the final evidence threshold, but the representation tranche is explicitly expected to fall short. Any later 10x attempt must expand into measured algorithmic or materialization reduction.
- The topology comparison uses batch priority with non-preemptible workers; retry blast radius is a separate deliberately preemptible experiment.
- The batch API contract now defines schema drift, empty batches, and stateless callbacks. Process reuse is removed from the first required matrix.

## What exists today

Zephyr already has most of the substrate needed for this work:

- `load_parquet(batch_mode=True)` yields `pyarrow.RecordBatch` objects, and the physical planner can pass them to an ordinary `map` callback ([dataset.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/dataset.py#L620-L671), [plan.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/plan.py#L210-L245)).
- Parquet files can be split on row-group boundaries. Earlier exact-dedup work reported a representative 16-minute to 6-minute improvement after adding row-group splitting and batch hashing ([PR #5859](https://github.com/marin-community/marin/pull/5859)).
- Shuffle buffers, sorting, merging, and Parquet chunk files are implemented with Polars. This improved tier-1 wall time from 4,671 seconds to 3,442 seconds, or 1.36x, but did not improve tier 2 and produced noisy tier-3 results ([PR #5963](https://github.com/marin-community/marin/pull/5963)).
- `ScatterWriter.write(df)` already accepts a routing-column DataFrame, and external sort/merge is already native Polars. The remaining scatter gap is producing routing columns without first wrapping every user row in a cloudpickle payload.
- Worker and task resources are separate. A long-lived `ZephyrWorker` admits concurrent shard tasks against its CPU/RAM pool and creates a runner for each admitted task ([execution.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/execution.py#L1163-L1313)). Tier 3 already uses 16-CPU actors with 1/16-scale map tasks for MinHash, fuzzy dedup, consolidate, and tokenize ([datakit_nemotron_ferry.py](https://github.com/marin-community/marin/blob/d1d081a33/experiments/ferries/datakit_nemotron_ferry.py#L90-L173)).
- Finelog and the perf collector expose per-stage task wall time, CPU time, peak memory, retries, failures, and end-to-end wall time. Remote comparisons can use task CPU/active time rather than scheduler-sensitive wall time.

The remaining hot path repeatedly crosses the Python object boundary:

```text
Parquet row group
  → RecordBatch
  → list[dict] for the default reader
  → Python map/filter callbacks
  → msgpack(key) + deterministic_hash(key) + Python sort callback
  → cloudpickle(item) payload
  → Polars shuffle Parquet
  → cloudpickle.loads(item)
  → Python reducer
  → Table.from_pylist(...)
  → output Parquet
```

Three concrete adapters account for this behavior:

- Non-shuffle stage output is stored in pickle chunks ([stage_io.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/stage_io.py#L189-L211)).
- Shuffle calculates arbitrary Python keys and sort values per item and stores `cloudpickle.dumps(item)` in a binary `__payload__` column before handing the batch to Polars ([shuffle.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/shuffle.py#L185-L213)).
- The Parquet writer rebuilds Arrow tables from row dictionaries in micro-batches ([writers.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/zephyr/src/zephyr/writers.py#L105-L179)).

This is exactly where Arrow's `RecordBatch` abstraction is useful: it is a bounded, immutable horizontal slice of columnar arrays intended for streaming and parallel execution. PyArrow tables can retain multiple batches as chunked arrays without immediately copying them. Polars uses the Arrow memory model and recommends expressions and lazy/streaming execution so it can optimize a whole query and process batches without materializing the full dataset. Python UDFs remain an escape hatch, but converting arrays to Python values is documented as one of the slowest paths in Arrow-native engines.

## Worker topology: name the two kinds of concurrency

Fray's Iris backend already submits an actor group as one Iris job with `N` replicas, not `N` independent jobs ([iris_backend.py](https://github.com/marin-community/marin/blob/d1d081a33/lib/fray/src/fray/iris_backend.py#L720-L776)). Each replica is a Zephyr actor; each actor can host several concurrent shard subprocesses. The current `max_workers` cap is therefore a cap on actor replicas, not on total shard-task concurrency.

For a 64-CPU map fleet with 1 CPU and 4 GiB per shard task, the configurations to compare are:

| Iris actor replicas | Actor shape | Task slots per actor | Total task slots | Failure blast radius |
|---:|---:|---:|---:|---:|
| 64 | 1 CPU / 4 GiB | 1 | 64 | 1 shard |
| 16 | 4 CPU / 16 GiB | 4 | 64 | up to 4 shards |
| 8 | 8 CPU / 32 GiB | 8 | 64 | up to 8 shards |
| 4 | 16 CPU / 64 GiB | 16 | 64 | up to 16 shards |

The sweep should choose among these shapes; do not pre-declare a default when prior fixed-resource evidence found no compute gain. Four actors × 16 slots minimizes replicas, while 8×8 or 16×4 reduce the retry blast radius when task memory is heavy-tailed or preemptions are frequent. Losing a packed actor requeues every in-flight shard on it.

Prior evidence sets expectations:

- A fixed-resource comparison of 64 single-slot actors with 32 two-slot actors found a statistically significant 1.6% slowdown in one stage, parity in another, and no packing speedup. Packing is therefore not a compute optimization by itself ([PR #6996](https://github.com/marin-community/marin/pull/6996)).
- An in-process Iris controller benchmark at 5,000 tasks measured about 257 ms steady-state reconciliation for 79 workers × 64 tasks, 742 ms for 625 × 8, and 4.33 seconds for 5,000 × 1. This is an order-of-magnitude illustration of Iris replica/fanout pressure, not a measurement of Zephyr's in-actor task threads.
- A shared-pool experiment improved a startup-bound 100M reference smoke from a mean 1,542 seconds to 880 seconds, or 1.75x, by avoiding roughly 60 coordinator/worker-pool startups. Its authors and reviewers explicitly cautioned that production jobs run for hours and may not benefit similarly ([PR #7145](https://github.com/marin-community/marin/pull/7145#issuecomment-4954254097)).

Do not split a 4–16 replica actor group into several Iris jobs merely to avoid a large job record. The existing replica job is the intended representation. Revisit job splitting only after observing coordinator serialization, status-query, or failure-domain limits at hundreds of replicas.

## Proposed API and execution changes

### Milestone 1: Arrow batches through map/filter/write

Make the supported fast path explicit while retaining row callbacks as the general path:

```python
def enrich_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    text = batch.column("text")
    keep = pc.and_(pc.is_valid(text), pc.not_equal(pc.utf8_trim_whitespace(text), ""))
    filtered = batch.filter(keep)
    return filtered.append_column("text_chars", pc.utf8_length(filtered.column("text")))

pipeline = (
    Dataset.from_files(input_glob)
    .load_parquet(batch_mode=True)
    .map_batches(enrich_batch)
    .write_parquet(output_pattern)
)
```

`map_batches` should be a one-input-batch to one-output-batch operation with `RecordBatch` as the public interchange type. A later `flat_map_batches` can cover one-to-many output. Avoid an "Arrow row" abstraction: selecting a scalar row recreates the Python-object overhead this work is meant to remove.

The callback is stateless across batches. It may return a zero-row batch, which must retain its schema and continue through the fused stage. The writer skips zero-row batches for row counts but remembers their schema so an all-empty stream still produces a typed empty Parquet file. With an explicit output schema, extra or incompatible fields fail. Without one, match the current dictionary path's compatible widening behavior before the first Parquet flush; reject schema drift that cannot be represented after the writer schema is fixed. Tests compare field types and field/schema metadata, not only row values.

Implementation work:

1. Add explicit `MapBatchesOp` semantics and types, even if it initially reuses the fused map executor.
2. Teach Parquet and Vortex writers to accept a stream of `RecordBatch` objects directly, validate schemas at batch boundaries, and flush based on `nbytes` without `to_pylist`/`from_pylist`.
3. Add batch-native projection and filter operations. Use Arrow compute for simple projection/filter and Polars for expression sets that benefit from fusion.
4. Preserve batch metrics: rows, bytes, conversion time, transform time, and write time.

Expected uplift: 2–6x for identity/projection/filter/write kernels dominated by row conversion; 1.2–2x for a full stage with substantial compression, hashing, or remote I/O. This must be measured rather than assumed.

### Milestone 2: productionize batch-native scatter and expression routing

Do not reimplement the open draft [PR #7200](https://github.com/marin-community/marin/pull/7200). The land order is: complete the distinct batch map/write milestone; then either help land and harden #7200 or explicitly coordinate a replacement with its author. #7200 already prototypes expression keys, direct `RecordBatch`/Polars ingestion, routing-column construction, and a fuzzy-dedup caller. The remaining work is productionization: typing, behavior-focused tests, stable routing, combiner semantics, plan visibility, and conversion metrics. A Python reducer may still receive dictionaries after merge; a future batch reducer can remove that final conversion.

The proof of concept measured the tier-2 fuzzy-attribute Map→Scatter CPU sum at 293.6 seconds on main and 120.6 seconds with direct DataFrame ingestion, a 58.9% reduction or 2.43x speedup over 10.3 million rows. Reduce CPU was unchanged, as expected. This is the strongest direct evidence for the proposed architecture.

Proposed expression form:

```python
pipeline.group_by(
    key=col("file_idx"),
    sort_by=col("id"),
    reducer=write_partition,
)
```

Add a `to_polars_expr` lowering for the existing comparison, arithmetic, boolean, null, and nested-field nodes. Expand the vocabulary only when a real datakit transform uses it: aliases, casts, string length/replacement, list length/explode, struct construction, conditional expressions, and stable hashing. Unsupported Python callables keep the current row/payload path and should be visible in the plan and counters.

Native routing must not use Polars' implementation-defined hash. It must reproduce the current msgpack-based `deterministic_hash(key) % num_output_shards` bit-for-bit. Acceptance tests compare each shard's membership and order across row and batch modes, across actor counts, and against resumed scatter output. A globally resorted digest is insufficient. The physical plan must show native-expression versus Python-callable routing, and counters must record rows entering Python materialization.

Expected uplift: about 2–2.5x for Python-heavy Map→Scatter phases based on the existing A/B; smaller for end-to-end shuffle because sort, Parquet I/O, and reduce remain.

### Milestone 3: lower representative datakit stages

Prioritize conversions with both high row counts and simple columnar semantics:

1. Fuzzy-dedup final attribute routing: adopt the proven direct batch path.
2. Fuzzy bucket emission: replace scalar iteration over `buckets` with list `explode`, derive IDs as columns, and feed scatter directly.
3. MinHash attributes: DupeKit's local native extension was verified to accept and return `pyarrow.RecordBatch`. Construct the output list column and write it as Arrow instead of converting every scalar and row to Python.
4. Normalize on Parquet inputs: batch-native text validation, projection, renaming, whitespace handling, and direct expression-key scatter. Keep a row fallback for heterogeneous JSON/JSONL sources until an Arrow JSON reader is justified.
5. Consolidate: evaluate a streaming Polars co-partitioned join for the common `KEEP_DOC`/`REMOVE_DOC` cases. Keep `REMOVE_SPANS` on the Python path initially.
6. Tokenize: preserve batch input until the tokenizer boundary. Tokenization itself already batches up to 64 documents, so expect a smaller serialization win than normalize or fuzzy scatter.

### Deferred: process reuse and pool lifecycle

`SubprocessRunner` starts a fresh interpreter per shard. Prior investigation measured roughly 3.6 seconds for heavy imports, 1.3 seconds for Numba JIT, and 2.9 seconds to load an 882 MB model in one Luxical workload ([issue #7120](https://github.com/marin-community/marin/issues/7120#issuecomment-4949338894)). Profile fixed startup on representative stages, but do not make process reuse part of the first implementation matrix. If later evidence justifies it, evaluate a bounded spawn-based pool per actor that reloads a process after a crash. Do not begin with `fork`: PyArrow, gRPC, Torch, and existing actor threads make fork safety a separate infrastructure project.

Shared worker pools across separate `ZephyrContext.execute` calls should remain optional and independently measured. They help many-small-stage smoke workloads, but changing the production execution model solely for smoke tests would weaken fidelity.

## Benchmark plan

### Local corpus

Build one cached local corpus from the first ten 1,000-row Parquet row groups of FineWeb-Edu `sample/10BT`, file `0000.parquet`, pinned to Hugging Face conversion commit `92cece42bcce787ee4af4619ab449fe48d86230d`. The source file has 726 row groups of 1,000 rows; selecting ten groups transfers roughly 50 MiB rather than the 2.15 GiB file. Keep `text`, `id`, `score`, and representative metadata columns.

Generate 10K, 100K, and 1M-row variants locally by deterministic repetition with a repetition suffix in `id`. This preserves real document sizes and text while avoiding a large external transfer and preventing accidental duplicate-key skew. Record the source commit, selected row groups, schema, row count, uncompressed bytes, and content digest in the benchmark manifest. Do not commit the corpus.

The repeated corpus is valid for row-local map/write, scatter-routing, and per-document MinHash attribute kernels. It is invalid for deduplication or connected-components benchmarks: repeated text creates pathological duplicate clusters. Any future local CC benchmark needs a non-repeated sample with a controlled duplicate distribution.

### Local kernels

| ID | Stage shape | Baseline | Treatment | Sizes |
|---|---|---|---|---|
| L1 | Parquet → projection/filter/derived column → Parquet | row reader + Python callbacks + `from_pylist` writer | RecordBatch + native expressions + batch writer | 10K, 100K, 1M rows |
| L2 | records → scatter by `file_idx`, sort by `id` | Python keys + cloudpickle payload | Polars expression routing + direct columns | 100K, 1M, 10M rows |
| L3 | Parquet → DupeKit MinHash → Parquet | Arrow compute then scalar/dict conversion | Arrow batch output and writer | 10K, 100K rows |

Run one warm-up and at least five measured iterations per arm in alternating order. Capture median and spread for wall time, process CPU time, rows/s, input/output bytes/s, and peak RSS. Use a single Polars thread for the single-core kernel comparison, then repeat at the intended task CPU allocation to expose nested parallelism. Compare sorted output digests, schemas, row counts, null counts, and stage counters.

### Remote topology sweep

Use `cw-rno2a` or `cw-us-east-02a` at batch priority. Hold the fleet at 64 CPU and the same total RAM, then compare 64×1, 16×4, 8×8, and 4×16 actor/slot shapes over at least 256 balanced shards. Run the primary comparison on non-preemptible workers so actor losses do not contaminate the CPU/active-time signal. Run a separate deliberately preemptible condition to measure retry amplification and packed-actor blast radius. Run a CPU-bound batch transform and a shuffle stage separately. Report:

- Finelog CPU and active time per stage and per successful shard;
- stage wall time after workers are ready, plus worker-ready/startup time separately;
- actor replica count, admitted task slots, CPU utilization, and peak RSS;
- preemptions, requeued shards, bytes reprocessed, and failures;
- output parity.

The decision is not "fastest single trial." Prefer the most packed shape whose p95 task memory fits with at least 25% headroom and whose retry-amplification cost is acceptable. Expect 4×16 or 8×8 to win on orchestration footprint, not necessarily task CPU time.

### Medium-scale progression

1. Tier-1-shaped FineWeb-Edu run: full end-to-end A/B, because it completes quickly enough to detect integration regressions.
2. Tier-2 skew run: stage-targeted normalize and fuzzy-dedup A/Bs during iteration, then one end-to-end confirmation. Its 46 GiB and injected 128–256 MiB documents are the memory/skew gate.
3. Tier-3-shaped run: use its approximately 1,000 input / 1,380 output shard geometry and 16-slot actor shapes, but do not run the canonical Nemotron tier 3 on the allowed CoreWeave clusters. Its raw data is pinned in `europe-west4`, and the ferry deliberately fails on cross-region access ([submit_perf_run.py](https://github.com/marin-community/marin/blob/d1d081a33/scripts/ci/submit_perf_run.py#L105-L123), [datakit_nemotron_ferry.py](https://github.com/marin-community/marin/blob/d1d081a33/experiments/ferries/datakit_nemotron_ferry.py#L188-L204)). Use an existing `marin-us-east-02a` dataset or a generated tier-2-derived shard set instead; do not copy Nemotron across regions.

Remote A/B decisions use the `ab-test-zephyr` per-stage Finelog workflow. Alternate arms, run two trials per arm when the first result is within 20%, and reject a change if output parity fails, retries materially increase, or peak memory loses the safety margin.

## Uplift model and gates

The latest cited tier-1 Polars run in PR #5963 reported: download 174.9s, normalize 744.8s, MinHash 393.0s, fuzzy dedup 835.7s, consolidate 163.1s, and tokenize 615.2s. Even an illustrative optimistic model—3x normalize, 1.5x MinHash, 2.5x fuzzy dedup, 2x consolidate, 1.5x tokenize, unchanged download—reduces the summed stage time from about 2,927 seconds to about 1,511 seconds, or 1.9x. This arithmetic is not a forecast; it demonstrates the Amdahl limit and why a 10x end-to-end claim requires more than serialization work.

Use these milestone gates:

| Milestone | Required evidence | Claim allowed |
|---|---|---|
| Batch writer/API | L1 ≥2x at 1M rows, same schema/digest, no peak-RSS regression over 20% | chunk-native map/write is faster |
| Native scatter | L2 ≥2x and tier-2 Map→Scatter CPU reduction reproduced | native scatter removes Python overhead |
| Datakit lowering | tier 1 ≥1.5x processing-only and tier 2 ≥1.25x, no reliability loss | practical pipeline uplift |
| Topology | same task CPU, fewer replicas/startup or lower controller load, acceptable retry cost | denser Iris execution |
| 10x | repeated full end-to-end run at the declared scale, including all uncached required stages | Zephyr datakit is 10x faster |

## Risks and safeguards

- Schema drift: Arrow/Polars conversions may promote string/list types. Treat an explicit schema as a contract and compare field types and metadata in every parity test.
- Nested parallelism: one task may call Polars, PyArrow, DupeKit, or a tokenizer that uses native threads. Size task CPU from measured utilization; do not assume one Python callback equals one core.
- Chunk memory: row groups and Polars buffers can exceed their encoded Parquet size. Keep budget-aware flushes, expose batch bytes, and test tier-2 mega-docs before scaling.
- Determinism: stable routing and per-shard order are hard compatibility gates, not optional safeguards. Polars hashes must not replace Zephyr's msgpack-based deterministic hash.
- Fallback cliffs: a single arbitrary Python callback can force row materialization. Show the selected execution mode in the physical plan and emit conversion counters so users can see the cost.
- Packed-actor retries: an actor loss requeues every active slot. Account for bytes reprocessed, not only the number of preemptions.
- Benchmark contamination: separate worker provisioning from active stage time, alternate A/B order, and use Finelog CPU/active time as the primary remote signal.

## Ranked hypotheses

1. Direct batch writers, followed by productionizing PR #7200's scatter prototype: highest confidence, high impact, bounded implementation; existing 2.43x map-stage evidence.
2. Batch-native fuzzy bucket expansion and normalize expressions: high row counts and obvious Python loops; likely material but needs semantic work.
3. Dense actors at 8–16 slots: high-confidence controller/replica reduction, low-confidence compute improvement.
4. Persistent spawn runner: profile-only until fixed startup is shown to matter on a representative stage.
5. Shared pools and SQL/DataFusion: useful adjacent directions, but not first-milestone dependencies. DataFusion can stream Arrow through the C interface, yet a distributed SQL execution model is broader than the measured serialization bottleneck.

## External references

- [Apache Arrow: data types and in-memory model](https://arrow.apache.org/docs/python/data.html)
- [Apache Arrow: dataset scanning returns record batches](https://arrow.apache.org/docs/python/dataset.html)
- [Polars: lazy query execution and streaming](https://docs.pola.rs/user-guide/lazy/execution/)
- [Polars: expressions and Arrow memory model](https://docs.pola.rs/user-guide/migration/pandas/)
- [Polars: `Expr.map_batches`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_batches.html)
- [DataFusion: Arrow batch streaming and zero-copy interchange](https://datafusion.apache.org/python/user-guide/io/arrow.html)
- [DataFusion: Python conversion cost in UDFs](https://datafusion.apache.org/python/user-guide/common-operations/udf-and-udfa.html)
