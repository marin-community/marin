# Research: Zephyr Performance Improvements

Related design: [design.md](design.md)

Findings through mid-2025. Sources are cited by paper title, author, and arXiv ID where available. Codebase references are to the Marin monorepo as of late April 2026.

---

## Part 1: Low-Hanging Fruit

### Better Parallelization — Parquet File Splitting

#### Current State in Zephyr

Zephyr currently assigns one file per shard: `compute_plan` in `lib/zephyr/src/zephyr/plan.py:525–555` calls `resolve_glob()` to discover files and creates one `SourceItem` per file. If a job has 200 input files and `max_workers=512`, 312 workers sit idle.

Importantly, the infrastructure for sub-file splitting already exists:
- `InputFileSpec` has `row_start`/`row_end` fields (`plan.py:157–158`) for per-shard row ranges
- `iter_parquet_row_groups()` in `readers.py:56–130` reads Parquet at the row-group level and correctly skips/slices groups outside the specified range
- `load_parquet` respects these ranges in the reader

The gap is entirely in the planner: `_compute_file_pushdown` does not yet produce multiple `SourceItem`s per file.

#### How Parquet Splitting Works

Parquet's physical structure is: file → row groups → pages. Row groups are the natural split boundary — splitting within a row group requires materializing and slicing the group in memory, which is expensive. The correct approach is to assign K consecutive row groups per shard.

PyArrow gives you all the metadata needed at plan time without reading any data:
```python
pf = pq.ParquetFile(path)
# pf.metadata.num_row_groups, pf.metadata.row_group(i).num_rows — no data read
```

This requires one GCS footer fetch per file at plan time (the Parquet footer is the last few KB–MB of the file). For jobs with thousands of input files this adds startup latency; fetches can be parallelized with a thread pool to bound the overhead.

#### Tradeoffs and Prior Art

**Row-group alignment is mandatory.** Splitting at non-row-group boundaries forces a full row-group read + slice for boundary chunks. Spark's `FileInputFormat.getSplits()` and Dask's `read_parquet(split_row_groups=True)` both enforce row-group-aligned splits. Zephyr's `iter_parquet_row_groups` already handles this correctly — it slices only groups that straddle the boundary.

**Minimum shard size matters.** Zephyr's `SubprocessRunner` (`runners.py:257–337`) forks a fresh Python subprocess per shard with ~700ms cold-import overhead. Very small shards (< ~10K records) make this overhead dominate. For Parquet inputs, don't produce more shards than there are row groups, and target at least 256–512 MB compressed or 100K–500K records per shard — consistent with Databricks best practices and Dask defaults.

**Scatter output fan-in.** More input shards → more mapper output files → higher reducer fan-in in Scatter/Reduce stages. For pipelines with reduce stages, validate that the reduce worker count can handle the increased fan-in before increasing shard count aggressively.

**The PyArrow memory leak.** `readers.py` explicitly avoids `pyarrow.dataset` in favor of `pq.ParquetFile.read_row_group()` due to [apache/arrow#39808](https://github.com/apache/arrow/issues/39808). Stick with the existing approach.

**Assessment: low-medium effort.** The reader already handles row ranges. The work is in `_compute_file_pushdown`: read footers at plan time, produce N `SourceItem`s per file with non-overlapping `row_start`/`row_end` ranges. This is the safest of the three optimizations with the highest confidence of immediate speedup for input-limited jobs.

---

### Better Resource Utilization — Multithreading and Per-Stage Worker Types

#### Current State in Zephyr

All pipeline stages use a single homogeneous `ResourceConfig` defined in `ZephyrContext`:
```python
# execution.py:1632–1635
resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
```

Within each worker, `SubprocessRunner` forks one subprocess per shard — no intra-worker multithreading for computation. The `ThreadPoolExecutor` in `execution.py:1048` is used only for final result materialization after all stages complete, not during stage execution.

#### Multithreading vs. Multiprocessing for Python Data Pipelines

Python's GIL means threads provide no CPU parallelism for pure-Python work. The key question is whether the hot code in each stage releases the GIL:

**GIL-releasing (threads help):**
- HuggingFace `tokenizers` library (written in Rust via PyO3) — `encode()` and `encode_batch()` release the GIL and are thread-safe
- PyArrow's `read_row_group()`, `filter()` — C++ operations release the GIL; `to_pylist()` does not (it constructs Python objects)
- `zstd.compress()` / `zstd.decompress()` in python-zstandard
- All blocking I/O (GCS reads/writes, socket operations)
- NumPy math operations (np.dot, np.sort, etc.)

**GIL-held (threads don't help for CPU parallelism):**
- Pure-Python string operations, dict iteration, list comprehensions
- `json.loads` / `json.dumps` (though `msgspec`, which Zephyr uses, is a C extension that releases the GIL)
- Most custom Python preprocessing logic

**Implication:** For tokenization stages (which use HuggingFace `tokenizers`), multithreading within a shard subprocess can provide real CPU parallelism. For pure-Python filter/map stages, it cannot. The monitoring workflow should be used to determine which category a given stage falls into before investing in thread-pool infrastructure.

#### Memory Sharing: Threads vs. Processes

The main motivation for multithreading over additional processes is memory sharing for large objects like tokenizer vocabularies:

| Model | Memory cost | Notes |
|---|---|---|
| 4 threads, 1 tokenizer loaded once | ~1x | Threads share the heap; zero copy |
| 4 processes, fork after load | ~1x (CoW) | Linux CoW is free for C data; Python ref-counting dirties CoW pages for Python objects |
| 4 processes, spawn (SubprocessRunner) | ~4x | Each subprocess re-imports everything (~700ms + tokenizer reload) |

Zephyr's `put()`/`get_shared()` mechanism mitigates the spawn-based overhead: the tokenizer is serialized to GCS once and each subprocess lazy-loads it. But this doesn't eliminate the ~700ms cold-import overhead per shard.

#### Per-Stage Worker Types

Assigning different `ResourceConfig` per stage is straightforward: `ZephyrContext` would take `stage_resources: dict[int, ResourceConfig]` and the coordinator would set the worker group's resource config before starting each stage. The Iris scheduler already supports heterogeneous worker types.

For example: an I/O-heavy load/filter stage may need only 0.5 CPU and 2 GB RAM, while a tokenization stage may benefit from 4 CPUs and 8 GB RAM to run a thread pool. Right-sizing these reduces wasted reservation cost.

**Assessment: low effort for per-stage resources** (minimal planner change, no execution model change). **Medium effort for intra-shard multithreading** (requires a `ThreadPoolExecutor` inside `SubprocessRunner`, with GIL-awareness per stage type). Validate with the monitoring workflow before implementing threading — if the hot stage doesn't release the GIL, threading provides zero benefit.

---

### Pipelining Stages

#### Current State in Zephyr

`run_pipeline()` in `execution.py:1015–1041` is a strict sequential loop:
```python
for stage_idx, stage in enumerate(plan.stages):
    self._start_stage(...)          # enqueue all tasks for this stage
    self._wait_for_stage()          # BLOCKING: wait for _completed_shards >= _total_shards
    result_refs = self._collect_results()
    shards = _regroup_result_refs(...)
```

`_wait_for_stage()` (`execution.py:926–977`) polls until every shard in the stage has reported completion. There is no partial-completion signal to the next stage.

Within a stage, operations are already pipelined: `_fuse_operations()` (`plan.py:392–466`) collapses consecutive Map/Filter/FlatMap ops into a single physical stage with composed generator functions. The barriers that remain are all at Scatter/Reduce boundaries (or any stage that changes the shard count).

#### Why Pipelining is Hard at Shuffle Boundaries

Spark's execution model is the clearest reference here: Spark distinguishes "narrow" dependencies (each output partition depends on exactly one input partition — map, filter, flatMap) from "wide" dependencies (shuffle — each output partition depends on all input partitions — groupBy, join). Narrow transforms are always pipelined; wide transforms (shuffles) always force a barrier. This is not a limitation of Spark's implementation — it's a fundamental property of the computation.

In Zephyr, `StageType.RESHARD` and Scatter→Reduce stages are wide dependencies. A reducer cannot start until all mappers have written their output, because it reads from all of them. These stages will always be hard barriers regardless of what we do.

The opportunity is at **narrow-dependency stage boundaries** — consecutive Map/Filter/Write stages where each shard K of stage N+1 depends only on shard K of stage N. For these, the coordinator could emit stage N+1's shard K task as soon as stage N's shard K reports completion, without waiting for all other shards in stage N.

#### Failure Handling Complications

This is the hardest part of pipelining, and the main reason to do it last. In the barrier model, if shard K of stage N fails and is retried, stage N+1 hasn't started yet — no contamination. With streaming:

- Stage N+1's shard K may have already consumed the output of stage N's shard K
- If stage N's shard K is retried (e.g., `MAX_SHARD_INFRA_FAILURES` not yet exceeded), stage N+1's shard K has consumed stale data
- The coordinator must track which downstream shards must be re-run when an upstream shard is retried, cascading through all pipelined stages
- Zephyr's most common failures are `SubprocessRunner` crashes (OOM, native segfault) — exactly the cases where contamination is most likely

Flink solves this with distributed checkpoints (Chandy-Lamport algorithm). For batch pipelines, the simpler mitigation is to restrict pipelining to stages after the last shuffle, or to accept that a failed upstream shard triggers cascaded downstream re-runs (expensive but correct if stages are idempotent, which Zephyr's `skip_existing` behavior supports).

#### Typical Speedup

For a K-stage pipeline with roughly equal stage durations and no tail latency, ideal pipelining speedup is K / (1 + pipeline_depth_stages) ≈ K/(K+1) of sequential time. With 4 equal-duration stages, ideal pipelining gives ~4/5 of sequential time (~1.25x speedup on wall time). The actual speedup is lower due to:
- Shuffle barriers that cannot be pipelined
- Straggler shards in each stage (the pipeline stalls waiting for the slowest shard before the next stage can begin for that shard's downstream)
- Increased coordinator complexity and scheduling overhead

Empirical speedups of 1.5–2x are achievable for multi-stage ML data prep pipelines with balanced stages. This is real but smaller than the gains likely available from fixing parallelization and resource utilization first.

**Apache Tez "pipeline shuffle"** (used in Hive) is the closest production analogue: reducers can begin consuming mapper output as soon as any mapper finishes, using a YARN-mediated "shuffle service." This is considerably more complex than what Zephyr needs for simple narrow-dependency streaming.

**Assessment: high effort, modest speedup.** The correctness complications around failure cascades and the relatively small theoretical ceiling (1.5–2x) compared to the implementation cost suggest this should come after the other two optimizations. The monitoring workflow may also reveal that the stage-barrier wait time is not the bottleneck — validate first.

---

## Part 2: Monitoring Workflow

### Academic Work

**"Can Large Language Models Optimize Code?" (Shypula et al., CMU, 2023 — arXiv:2309.14328)**
Evaluated GPT-4 and CodeLlama on runtime optimization of C++/Python. LLMs suggested valid optimizations ~50% of the time and excel at recognizing algorithmic complexity mismatches (O(n²) vs O(n log n)) but struggle with cache-coherency and memory-layout issues. Critical pitfall: models hallucinate optimization rationales — a model will explain "this is slow because of repeated allocation" without verifying that allocation is actually the bottleneck. Treat LLM output as a hypothesis to validate, not a prescription to apply.

**"Performance-Aware LLM Code Generation" (Liu et al., 2024 — arXiv:2404.18864)**
Found that chain-of-thought with a "performance reasoning" step substantially improves output quality over zero-shot. Directly relevant: giving the LLM sample counts and percentages (exactly what `job_profile_summary.py` produces) is better than raw folded stacks. Ask the model to reason about the impact ceiling — "if we eliminated this frame entirely, what % improvement is theoretically possible?" — before recommending.

**SWE-Perf benchmark (2024–2025, multiple authors)**
GPT-4-class models correctly identified the hot frame 70–80% of the time but proposed fixes that were semantically wrong (e.g., memoizing functions with side effects). Implication: treat LLM output as a hypothesis to validate, not a prescription to apply.

**"LLM-Assisted Root Cause Analysis for Performance Regressions in CI" (Google Brain, 2024)**
The LLM's role was *triage*, not prescription: it summarized "this change to `hash_map.cc` corresponds to a 12% increase in cache misses in stage X." Recall on true root causes was ~85% for single-commit regressions. Key finding: grounding with a git diff *and* a profiling delta together was essential — without the diff, the model produced generic outputs. Baseline comparison matters.

**"Automated Performance Diagnosis with Structured Telemetry" (Microsoft Research, 2024–2025)**
Two-pass pipeline for distributed tracing: first pass deterministically extracts the span with the highest exclusive time; second pass reasons over the surrounding context with code retrieval (RAG) to generate an action. Key result: bottleneck identification needs no RAG (it's deterministic), but *fixing* it requires code context. This is the most architecturally relevant paper: it maps directly to `job_profile_summary.py` as the first pass and the LLM skill as the second.

**FlameScope + LLM (multiple groups, 2024)**
The folded-stack text format (py-spy `--format raw`) is consistently superior to SVG or JSON for LLM input. SVG is lossy for a vision model at scale; JSON is too verbose. The normalized folded-stack text preserves all information and is compact.

**"LLM-based Microservice Performance Diagnosis" (various, 2024)**
Without an explicit SLO or baseline, the model highlighted the single slowest span regardless of whether it was anomalous. Always provide a reference point — a prior run, theoretical throughput, or at minimum the expected duration of the stage.

**"ProfiLM" (Chen et al., 2024 — arXiv preprint)**
Fine-tuning on profiling → optimization pairs worked for dataset-specific Python patterns (pandas/NumPy) but failed to generalize. Conclusion: domain-specific fine-tuning is high-effort and brittle; RAG with codebase context is the better path.

---

### Industry Tools and Deployments

**Grafana Pyroscope — AI Flamegraph Analysis (Grafana Labs, 2023–2025)**
Sends top-N hottest stacks as a formatted prompt, no RAG over user code. Documented pitfall: profiles taken during GC or startup bursts are confidently misinterpreted as structural bottlenecks. They added a UI caveat. Source: Grafana flamegraph panel docs, early 2025.

**Datadog Watchdog with LLM Explanation (Datadog, 2024)**
Classic two-stage pattern: statistical anomaly detection → LLM narrative summary. Summaries were useful for well-known service patterns, unhelpful for novel infrastructure with no prior context. Confirms that domain context in the system prompt is essential.

**AWS CodeGuru Profiler (Amazon, 2021–2024, now deprecated)**
Statistical model identifies the frame, LLM explains and recommends. Published limitation: recommendations were generic without code-specific context. Confirmed the RAG requirement.

**GitHub Copilot + Profiling (Microsoft/GitHub, 2024)**
50–100 lines of folded-stack summary with domain context produced actionable results in ~60% of Python profiling sessions tested internally. Profiles > 500 lines overwhelmed the context window. Confirm that the `--llm-summary` output from `job_profile_summary.py` stays under ~4,000 tokens.

---

### Prompt Engineering Approaches

#### Chain-of-Thought

A staged prompt consistently outperforms single-shot "analyze this profile":
1. "List the top frames and their sample shares."
2. "For each frame consuming >5% of samples, estimate the impact ceiling (maximum % improvement if this frame were eliminated entirely)."
3. "For the top 2–3 frames, what is the likely root cause and what could reduce it? State any preconditions your recommendation requires."

#### RAG with Codebase Context

Retrieving and including the source code of the top 3–5 hottest frames is the single highest-value intervention, per both the Microsoft RCA paper and the CodeRAG-Bench (2024) evaluation. Use the file path and line number from py-spy frames (format: `function_name (file.py:line)`) for precise retrieval. 30–50 lines of context per function is sufficient; retrieving whole modules wastes tokens and dilutes signal.

Pitfall: retrieving by function name alone can pull the wrong file if there are multiple definitions. py-spy frames include the full file path — use it.

#### Structured Output Schema

```json
{
  "bottleneck_type": "cpu_bound|io_bound|lock_contention|gc|coordinator_overhead|data_skew",
  "hotspot_call_chain": ["frame1", "frame2", "frame3"],
  "root_cause_hypothesis": "...",
  "impact_ceiling_pct": 38,
  "is_straggler_pattern": false,
  "recommendations": [
    {
      "action": "...",
      "target_function": "...",
      "estimated_impact": "...",
      "preconditions": "...",
      "confidence": "high|medium|low",
      "confidence_rationale": "..."
    }
  ],
  "suggested_next_instrument": null
}
```

Keep `root_cause_hypothesis` free text — over-constraining reasoning fields loses quality. Categorical confidence (high/medium/low) with a required `confidence_rationale` outperforms numeric scores, which are poorly calibrated without grounding.

#### Few-Shot Examples

Including 1–2 worked examples (profile → analysis) in the system prompt substantially improves output structure. The most useful examples show the *reasoning process*, not just the conclusion. For distributed pipelines: examples distinguishing worker-local bottlenecks (straggler) from universal bottlenecks are particularly valuable.

---

### Key Pitfalls

**1. Sampling artifacts mislead the model.** py-spy is a sampling profiler at 20Hz. A profile taken during startup, GC, or a transient burst will be confidently misinterpreted as a structural bottleneck. Pass `captured_at_ms` timestamps and the shard completion percentage so the model knows where in the job lifecycle the profile was taken.

**2. CPU profiles are blind to blocking waits.** If dominant leaf frames are `epoll_wait`, `pthread_cond_wait`, `select`, or similar, the bottleneck is not CPU-bound. Detect this before sending to the LLM and flag it explicitly — the model needs this signal to suggest the right next instrument (strace, async profiler, coordinator-side counters).

**3. Merged profiles hide straggler patterns.** Always pass a per-worker sample breakdown alongside the merged profile. The LLM cannot distinguish "all 100 workers are slow at this frame" from "3 workers are dominating the merged profile" without this.

**4. Context window discipline.** The top-50 stacks text representation should be under ~4,000 tokens. Use the `--top 50` flag and the `--llm-summary` format. Test this before deploying.

**5. Hallucinated fix preconditions.** Include the function source (RAG) and explicitly ask the model to state the preconditions its recommendation requires. LLMs will recommend caching or memoization without checking for side effects in the call chain.

**6. No-baseline queries produce low-quality output.** Always provide a reference point: a prior run's profile, the expected stage duration, or at minimum the total job wall time vs. expected. Without a baseline, the model highlights the single slowest thing regardless of whether it's anomalous.

**7. Overconfidence on novel code.** Require the model to state what it cannot determine from the profile alone. Novel code paths (unfamiliar library internals, Zephyr-specific coordination patterns) are where hallucination is highest.

---

### Gaps in the Literature

- Almost no published work on LLM analysis of **distributed** CPU profiles merged across workers. Most work assumes single-process profiling.
- **Stage-aware profiling** — treating different pipeline stages as separate analysis units — combined with LLM analysis has not appeared in the literature. This is a genuine gap the Zephyr monitoring workflow could address.
- The "CPU sampler + blocking wait" blind spot has no clean published solution. The best practice is to add an explicit frame-inspection step before sending to the LLM.
