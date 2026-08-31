---
name: ab-test-zephyr
description: Run an explicitly requested Zephyr control/treatment benchmark on the same pre-normalized sample and compare Finelog stage metrics.
---

# A/B test Zephyr changes

## Signals

The coordinator writes one `zephyr.stage` row per completed stage and
`execution_id`. Use these fields:

| Field | Aggregation across executions | Interpretation |
|---|---|---|
| `cpu_time_total` | sum | Primary efficiency and compute-cost signal |
| `elapsed` | sum, labeled as summed stage elapsed | Secondary latency signal; sensitive to scheduling and stragglers |
| `items` | sum | Workload-equivalence check |
| `bytes_processed` | sum | Workload-equivalence check |
| `mem_peak_bytes_max` | max | Worst observed shard RSS and OOM guardrail |
| `mem_bytes_avg` | weighted interpretation only | Typical shard RSS context |
| `cpu_pct_avg` | weighted interpretation only | CPU saturation context |
| `item_rate`, `byte_rate` | do not aggregate | Derived from noisy elapsed time |

`cpu_time_total` sums process user and system CPU-seconds across completed
shards, so worker count and queue delay do not directly change it. Use it as the
primary efficiency signal and normalize per item or byte when accepted workload
sizes differ. `elapsed` measures the stage barrier and includes startup, I/O,
concurrency, queueing, and stragglers; repeat an elapsed-only result under
comparable scheduling conditions.

Keep CPU and elapsed time as separate outcomes:

- CPU flat or lower and elapsed lower: latency or topology win without added
  compute cost.
- CPU higher and elapsed lower: faster and more expensive.
- CPU lower and elapsed higher: cheaper and slower.
- Wall-only change from one comparison: inconclusive until repeated.
- Topology or batching change: report the latency/compute tradeoff; do not
  describe wall-time gains as equivalent per-core efficiency gains.

Calibrate thresholds from same-code repeats for the selected sample and pool
shape. A new OOM, application failure, or memory peak above the worker limit is
a regression regardless of CPU.

## Choose the comparison

### Existing runs

Start at [Collect execution IDs](#collect-execution-ids) when control and
treatment jobs already exist. Confirm that the control and every treatment used
the same immutable sample, stage range, sources, worker resources, concurrency,
parallelism, cluster, region, and priority.

A scheduled baseline is usable only when its report contains the same workload
fingerprint and its Finelog execution IDs remain queryable. Otherwise, launch a
matching control. Do not compare a standalone benchmark treatment with a
differently shaped ferry baseline.

### New runs

Default to one control at the branch/PR merge base and one treatment at the
branch/PR head. Add treatments only when the requester explicitly names each
additional commit or configuration. Record a stable name plus the exact SHA and
configuration difference for every extra arm; do not infer or invent arms.

Run on GCP in `europe-west4` with
`gs://marin-eu-west4/datakit/sample_100b_8ae7a94f` unless the requester
selects another sample or backend. The us-central1 GCS sample is available for
us-central1 runs. CoreWeave remains available for S3-local runs; select it
explicitly with the matching S3 sample and target cluster.

For a PR, read the diff and select the smallest stage range that exercises the
changed behavior:

| Change | Minimum coverage |
|---|---|
| Stage-local map, serialization, or tokenization path | The affected stage on enough shards to amortize startup |
| Shuffle, partitioning, spill, merge, or buffer behavior | Exact or MinHash through fuzzy dedup on skewed or production-shaped data |
| Shared-pool lifecycle, scheduling, or pipeline concurrency | All affected stages with representative concurrent sources |
| Documentation, tests, types, or log text only | Skip the remote benchmark with reviewer agreement |

Confirm the sample size, pool shape, stage range, cluster, and expected cost
before launching an expensive or production-scale comparison. Run local Zephyr
and Datakit tests before paying for remote workers.

## Prepare worktrees

For a PR, use the merge base as the control and the PR head as the first
treatment:

```bash
git fetch origin main
BASELINE_SHA=$(git merge-base origin/main HEAD)
TREATMENT_SHA=$(git rev-parse HEAD)
WORKTREE_ROOT=$(mktemp -d /tmp/zephyr-ab.XXXXXX)
git worktree add --detach "$WORKTREE_ROOT/control" "$BASELINE_SHA"
git worktree add --detach "$WORKTREE_ROOT/treatment" "$TREATMENT_SHA"
```

Record both SHAs. Preserve configuration-only arms in separate worktrees or
commits. Add arms only when explicitly requested.

## Launch the download-free benchmark

`experiments.datakit.zephyr_benchmark` accepts an existing normalized sample
and routes outputs to a seven-day temporary prefix. Use an immutable,
region-local sample. Its default input is the GCS 100B sample in `europe-west4`.
All arguments except fresh output run tags must match across the control and
treatments.

### Choose a benchmark scale preset

| Preset | Input | Workers | Map task | Reduce task | Pipelines | Dedup shards | CC rounds |
|---|---|---|---|---|---:|---:|---:|
| Full | `--sources all`: 256 GB, 768 parquet shards, 115 sources | 48 × 2 CPU / 16 GiB RAM / 16 GiB disk | Whole worker | Whole worker | 128 | 1000 | 3 |
| Light | `--source-fraction 0.1`: 26.8 GB, 165 parquet shards, 77 sources | 48 × 2 CPU / 16 GiB RAM / 16 GiB disk | Whole worker | Whole worker | 80 | 500 | 3 |

Use the full preset for a change that could regress the shared pool at
production scale (shuffle, partitioning, spill, buffer, or scheduling
changes). Use the light preset for a quicker, cheaper signal on a
stage-local change.

The full sample is the closest pre-built workload to the nemotron ferry's
measured ~350 GB shape. There is no pre-built sample near 10% of that size, so
`zephyr_benchmark.py` builds the light preset's data at launch time:
`--source-fraction 0.1` lists every source's parquet shards, greedily selects
whole sources ordered by ascending average shard size (favoring shard-dense
sources) until their combined size reaches ~10% of the full sample, and logs
the resulting source count, bytes, and shard count.

`zephyr_benchmark.py` fails with a clear error if `--pool-workers` exceeds the
parquet shard count of the selected sources (whether from `--sources` or
`--source-fraction`), rather than silently leaving workers idle.

### Worker and task sizing

The worker is a scheduling unit; the task is the subprocess capacity reserved
inside it. The presets use the reference pipeline's default worker with 2 CPU,
16 GiB RAM, and 16 GiB disk, and let both map and reduce tasks inherit that
whole worker. Global exact dedup, tokenization, MinHash, and fuzzy dedup
therefore all use the same one-task-per-worker shape as the reference pipeline.

`zephyr_benchmark.py` uses the worker and task resource shapes in the preset
table by default. Pass the resource flags only when the comparison intentionally
changes those shapes, and record every override in the workload fingerprint.

Both presets use the same 48-worker pool, exposing up to 48 concurrent map or
reduce tasks. This keeps the previous presets' aggregate 96 CPU and 768 GiB RAM
while matching the reference pipeline's worker shape. A 256-worker full run
completed in 23m41s, ramped to 512 concurrent reduce tasks, and triggered GCS
`429 SlowDown` responses.

The current 48-worker light preset completed in 27m55s and used 15.32 CPU-hours,
with no shard retries, GCS 429s, Iris failures, or preemptions. The earlier
12-worker packed light preset completed in 27m25s and used 16.91 CPU-hours.

The clean full-preset measurement still comes from the earlier packed shape. It
processed 256.4 GB in 2h54m28s and used 182.83 CPU-hours; Finelog recorded
18,970 completed shards and a 6.61 GiB maximum stage memory peak. A 48-worker
full attempt encountered 27 Iris worker-pool preemptions and is not valid
performance evidence, even if the pipeline eventually completes. Run a clean
48-worker full preset before citing a current full-preset duration.

The 2 CPU / 16 GiB worker is the reference pipeline default. Forty-eight of
them fit within the same aggregate CPU and RAM budget as the earlier packed
shape, while scheduling and task admission remain representative of the
pipeline being benchmarked.

### Pipeline and connected-components limits

The previous light benchmark took 2h20m with four concurrent pipelines, one
task admitted per worker, and the default connected-components budget. Finelog
(`zephyr.stage`) and Iris job logs identified both bottlenecks:

- **`--max-concurrent`** limits StepRunner fan-out and concurrent pipelines in
  the shared pool; worker count and per-task resources separately bound shard
  concurrency. During the tokenize/MinHash phase, each source contributes
  two independent pipelines, so low values strand most of the pool. Zephyr
  sizes the coordinator actor's call budget to
  `max(100, 2 * workers + max_concurrent_pipelines)`: long-lived pipeline waits
  cannot occupy the slots workers need for polling and heartbeats. Keep the two
  concurrency limits equal; the coordinator rejects excess pipelines instead
  of queueing them. Fray must pass this actor setting through to Iris's
  `ActorServer`; otherwise Iris's 32-call default can starve worker heartbeats
  even though Zephyr requested a larger budget.
- **`--dedup-cc-max-iterations`** bounds fuzzy dedup's connected-components
  rounds. Each round is a full scatter/reduce pass over the whole bucket graph
  regardless of how little remains to resolve; the earlier run showed 11
  sequential rounds at ~350s each — about 65 of the 140 minutes after
  tokenize/minhash finished — because `zephyr_datakit_steps` left
  `cc_max_iterations` at the library default of 10 instead of the 3 both
  datakit ferries use. Set it to 3 to match ferry behavior and cut this phase
  by roughly 3x; dedup completeness in the benchmark's output does not matter
  for a performance comparison.

### Select map or shuffle work

`--target all` runs exact dedup, tokenization, MinHash, and fuzzy dedup. For a
map-only comparison, use `--target map`; it runs tokenization and MinHash with a
fresh run tag.

A shuffle comparison reads the permanent MinHash inputs stored under the
normalized sample's `_benchmark_inputs/` subtree. Use `--target shuffle` with a
fresh run tag; no preparatory map benchmark is required. The benchmark checks
every selected source's MinHash artifact before starting the pool and directs
the operator to `materialize_zephyr_benchmark_sample --mode minhash` if the
sample needs a backfill. `shuffle` runs global exact and fuzzy dedup; `exact`,
`tokenize`, `minhash`, and `fuzzy` are also available for a single-stage run.
Map-only outputs remain temporary benchmark results and are not inputs to a
later shuffle run.

Set exactly one data-locality argument before launching:

```bash
# Default: GCS input and GCP compute in europe-west4.
SAMPLE_PREFIX=gs://marin-eu-west4/datakit/sample_100b_8ae7a94f
DATA_LOCALITY_ARGS=(--region europe-west4)

# GCP opt-in: use the existing us-central1 sample with us-central1 compute.
# SAMPLE_PREFIX=gs://marin-us-central1/datakit/sample_100b_8ae7a94f
# DATA_LOCALITY_ARGS=(--region us-central1)

# CoreWeave opt-in: S3 input and CoreWeave compute in cw-us-east-02a.
# SAMPLE_PREFIX=s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f
# DATA_LOCALITY_ARGS=(--target-cluster cw-us-east-02a)
```

Set the cluster or region from the actual sample prefix. If the mapping is
unknown, stop before launching. The benchmark passes `source_prefix` to
`marin_temp_bucket`, which keeps temporary outputs with the sample. Do not
override the output location or launch compute in a different region.

Launch each arm from its worktree:

```bash
cd <CONTROL_OR_TREATMENT_WORKTREE>
uv run iris --config=lib/iris/config/marin.yaml job run --no-wait \
  --job-name zephyr-ab-<RUN_TAG>-<ARM> \
  "${DATA_LOCALITY_ARGS[@]}" --memory=2G --disk=5G --cpu=1 --extra=cpu \
  --priority batch \
  -- python -m experiments.datakit.zephyr_benchmark \
    --sample-prefix "$SAMPLE_PREFIX" \
    --sources <COMMA_SEPARATED_SOURCES_OR_ALL> \
    `# or --source-fraction <FRACTION> for the light preset, in place of --sources` \
    --run-tag <FRESH_RUN_TAG>-<ARM> \
    --pool-workers <WORKERS> \
    --target <all|map|shuffle|exact|tokenize|minhash|fuzzy> \
    --max-concurrent <PIPELINES> \
    --dedup-max-parallelism <SHARDS> \
    --dedup-cc-max-iterations <ROUNDS>
```

Record this workload fingerprint for every arm:

- commit SHA and Iris job ID
- sample prefix and source selection
- benchmark target
- pool worker and per-task CPU, RAM, and disk
- maximum concurrent pipelines, dedup parallelism, and dedup CC max iterations
- Iris controller, data-local target cluster or region, priority, and preemptibility
- run tag

Use fresh run tags so no arm cache-hits. One matching control can be reused for
explicitly requested treatments launched in the same scheduling window. If the
decision depends on elapsed time, interleave additional control trials among
the treatments to measure scheduling noise.

If the request includes continuous monitoring, use `babysit-zephyr`. A failed
or preempted arm measures infrastructure reliability and carries no performance
result. Use `debug` only for a stated repeated fault.

## Collect execution IDs

A benchmark job can run many Zephyr pipelines on one shared pool. Collect every
`YYYYMMDD-HHMMSS-<hex>` execution ID from the control and each treatment's root
job and descendant logs:

```bash
uv run iris --cluster marin job logs <IRIS_JOB_ID> \
  --max-lines 200000 --no-tail --level info | \
  rg -o '[0-9]{8}-[0-9]{6}-[0-9a-f]{8}' | sort -u
```

See `lib/zephyr/OPS.md` for child-job naming when a missing execution needs a
specific coordinator log. Preserve the control and treatment ID lists with the
workload fingerprint.

## Query Finelog

Authenticate with `uv run iris --cluster marin login` when needed. Query the
`zephyr.stage` namespace through the cluster's Finelog deployment:

```bash
uv run finelog query marin --format table '
  SELECT execution_id, stage_name, status, cpu_time_total, elapsed,
         items, bytes_processed, mem_peak_bytes_max, mem_bytes_avg, cpu_pct_avg
  FROM "zephyr.stage"
  WHERE execution_id IN (<CONTROL_AND_TREATMENT_IDS>)
  ORDER BY execution_id, stage_name'
```

Every expected row must have `status = 'END'`. A `FAILED` row invalidates that
arm. Missing rows usually mean an execution ID was omitted or Finelog emission
failed; resolve the gap before reporting a pass.

Aggregate all executions in the control and one treatment, then compare by
`stage_name`:

```sql
WITH tagged AS (
  SELECT CASE
           WHEN execution_id IN (<CONTROL_IDS>) THEN 'control'
           WHEN execution_id IN (<TREATMENT_IDS>) THEN 'treatment'
         END AS arm,
         stage_name, cpu_time_total, elapsed, items, bytes_processed,
         mem_peak_bytes_max
  FROM "zephyr.stage"
  WHERE status = 'END'
    AND execution_id IN (<CONTROL_AND_TREATMENT_IDS>)
), aggregated AS (
  SELECT arm, stage_name,
         SUM(cpu_time_total) AS cpu_time_total,
         SUM(elapsed) AS elapsed,
         SUM(items) AS items,
         SUM(bytes_processed) AS bytes_processed,
         MAX(mem_peak_bytes_max) AS mem_peak_bytes_max
  FROM tagged
  GROUP BY arm, stage_name
)
SELECT b.stage_name,
       b.cpu_time_total AS control_cpu,
       t.cpu_time_total AS treatment_cpu,
       (t.cpu_time_total - b.cpu_time_total) / NULLIF(b.cpu_time_total, 0) AS cpu_delta,
       b.elapsed AS control_elapsed,
       t.elapsed AS treatment_elapsed,
       (t.elapsed - b.elapsed) / NULLIF(b.elapsed, 0) AS elapsed_delta,
       t.items - b.items AS items_delta,
       t.bytes_processed - b.bytes_processed AS bytes_delta,
       b.mem_peak_bytes_max AS control_mem_peak,
       t.mem_peak_bytes_max AS treatment_mem_peak
FROM aggregated b
JOIN aggregated t USING (stage_name)
WHERE b.arm = 'control' AND t.arm = 'treatment'
ORDER BY cpu_delta DESC;
```

Use this SQL once per treatment, reusing the same control IDs. Replace each ID
placeholder with comma-separated, single-quoted execution IDs. Keep each raw
query output with the report. Keep repeated trials separate; do not merge
different variants or unequal trial counts into one ID set.

## Validate comparability

Before interpreting deltas:

1. Confirm the control and treatment workload fingerprints match except for
   SHA, arm, and run tag.
2. Confirm each stage has matching `items` and `bytes_processed`, within a
   fraction of a percent. Explain and normalize any accepted mismatch.
3. Confirm the control and treatment completed the same execution and stage
   set.
4. Inspect `iris job describe <IRIS_JOB_ID>` for OOMs and peak task memory.
5. Check job logs for retries, preemptions, hardware faults, and stragglers.
6. Run the change's semantic validation separately. Matching item counts do not
   prove output equivalence.

Different work, a failed stage, or material infrastructure churn makes the
comparison inconclusive. Re-run before assigning a performance verdict.

## Report

For a PR, update one sentinel-marked comment so reruns do not accumulate stale
verdicts:

```markdown
<!-- zephyr-ab-test -->
🤖 ## Zephyr A/B test

Verdict: pass | regression | tradeoff | inconclusive

Workload: <sample, stage range, sources, pool shape, concurrency, cluster>
Control: <sha>, <job>, <execution count>
Treatments: <name, sha, job, and execution count for each>

| Treatment | Stage | CPU control | CPU treatment | CPU change | Elapsed control | Elapsed treatment | Elapsed change | Peak memory change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Data check: <items and bytes comparison>
Infrastructure: <preemptions, retries, failures, stragglers, or none>
Interpretation: <efficiency result, latency result, and any tradeoff>
```

Lead with CPU change, then elapsed time and memory. State whether elapsed came
from one comparison or repeated interleaved trials and label summed stage
elapsed. Launcher duration and task wall time do not replace stage metrics.

## Clean up

Remove temporary worktrees after preserving the SHAs, job IDs, execution IDs,
workload fingerprints, and Finelog output. Repeat the treatment command for
each additional worktree:

```bash
git worktree remove "$WORKTREE_ROOT/control"
git worktree remove "$WORKTREE_ROOT/treatment"
```

Benchmark outputs expire under their seven-day temporary prefix.

## Related guidance

- `babysit-zephyr` monitors every control and treatment job through terminal
  state.
- `debug` investigates repeated failures or unexplained infrastructure churn.
- `lib/zephyr/OPS.md` documents coordinator queries and straggler diagnosis.
- `lib/iris/OPS.md` documents job summaries, task attempts, and Finelog access.
