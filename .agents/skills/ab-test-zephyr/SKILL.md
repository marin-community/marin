---
name: ab-test-zephyr
description: Run paired Zephyr benchmarks on pre-normalized data and compare per-stage Finelog CPU, elapsed-time, and memory stats. Use for ad hoc comparisons and PR performance gates.
---

# A/B Test Zephyr Changes

Use one workflow for ad hoc comparisons and PR performance gates:

1. Run control and treatment with
   `experiments.datakit.zephyr_benchmark` on the same pre-normalized sample.
2. Collect every execution's `zephyr.stage` rows from Finelog.
3. Compare CPU, elapsed time, and memory per stage.
4. Publish the workload fingerprint, data-equivalence checks, infrastructure
   noise, and result in one report.

The benchmark starts after corpus download. Do not include Hugging Face download
time in a Zephyr performance result.

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

`cpu_time_total` is the sum of process user and system CPU-seconds across
completed shards. It is the default signal for code efficiency because worker
count and queue delay do not directly change it. Normalize it as CPU-seconds
per item or byte when the two runs processed slightly different amounts of
data.

`elapsed` measures a stage barrier. It captures startup, I/O, concurrency, and
straggler behavior that CPU time misses. It also moves with worker availability,
preemption, retries, autoscaling, and data skew. Report it, but repeat a result
when elapsed time is the only signal that changed.

Keep CPU and elapsed time as separate outcomes:

- CPU flat or lower and elapsed lower: latency or topology win without added
  compute cost.
- CPU higher and elapsed lower: faster and more expensive.
- CPU lower and elapsed higher: cheaper and slower.
- Wall-only change from one pair: inconclusive until repeated under comparable
  scheduling conditions.

[PR #7888](https://github.com/marin-community/marin/pull/7888) illustrates the
distinction. Shared pooling reduced the through-MinHash span 28.72x while total
CPU fell 1.23%; bounded token batching reduced summed stage elapsed 27.26% while
CPU rose 5.59%. The [Echo record](https://echo.oa.dev/wiki/77) preserves the
workloads and validation details.

Do not apply fixed wall-time thresholds to every benchmark. Calibrate CPU/item
and elapsed thresholds from same-code repeats for the selected sample and pool
shape. A new OOM, application failure, or memory peak above the worker limit is
a regression regardless of CPU time.

## Choose the comparison

### Existing runs

Start at [Collect execution IDs](#collect-execution-ids) when control and
treatment jobs already exist. Confirm that both runs used the same immutable
sample, stage range, sources, worker resources, concurrency, parallelism,
cluster, region, and priority.

A scheduled baseline is usable only when its report contains the same workload
fingerprint and its Finelog execution IDs remain queryable. Otherwise, launch a
paired control. Do not compare a standalone benchmark treatment with a
differently shaped ferry baseline.

### New runs

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

## Prepare paired worktrees

For a PR, use the merge base as the control and the PR head as the treatment:

```bash
git fetch origin main
BASELINE_SHA=$(git merge-base origin/main HEAD)
TREATMENT_SHA=$(git rev-parse HEAD)
WORKTREE_ROOT=$(mktemp -d /tmp/zephyr-ab.XXXXXX)
git worktree add --detach "$WORKTREE_ROOT/control" "$BASELINE_SHA"
git worktree add --detach "$WORKTREE_ROOT/treatment" "$TREATMENT_SHA"
```

Record both SHAs. If the experiment changes configuration without changing
code, use two worktrees or commits that preserve the exact control and
treatment configurations.

## Launch the download-free benchmark

`experiments.datakit.zephyr_benchmark` accepts an existing normalized sample
and routes outputs to a seven-day temporary prefix. Use an immutable,
region-local sample. All arguments except `--run-tag` must match between arms.
Use `--target-cluster=<CLUSTER>` for a federated CoreWeave sample or
`--region=<REGION>` for a GCP sample.

Launch each arm from its worktree:

```bash
cd <CONTROL_OR_TREATMENT_WORKTREE>
uv run iris --config=lib/iris/config/marin.yaml job run --no-wait \
  --job-name zephyr-ab-<RUN_TAG>-<ARM> \
  <PLACEMENT_FLAG> --memory=2G --disk=5G --cpu=1 --extra=cpu \
  --priority batch \
  -- python -m experiments.datakit.zephyr_benchmark \
    --sample-prefix <NORMALIZED_SAMPLE_PREFIX> \
    --sources <COMMA_SEPARATED_SOURCES_OR_ALL> \
    --run-tag <FRESH_RUN_TAG>-<ARM> \
    --pool-workers <WORKERS> \
    --pool-cpu <CPU_PER_WORKER> \
    --pool-ram <RAM_PER_WORKER> \
    --pool-disk <DISK_PER_WORKER> \
    --first-stage <exact|tokenize|minhash|fuzzy> \
    --last-stage <exact|tokenize|minhash|fuzzy> \
    --max-concurrent <PIPELINES> \
    --dedup-max-parallelism <SHARDS>
```

Record this workload fingerprint for both arms:

- commit SHA and Iris job ID
- sample prefix and source selection
- first and last stage
- pool workers, CPU, RAM, and disk
- maximum concurrent pipelines and dedup parallelism
- Iris cluster, target region, priority, and preemptibility
- run tag

Use fresh run tags so neither arm cache-hits. Launch the arms close together. If
the decision depends on elapsed time, alternate at least two trials per arm
when cost permits: control, treatment, control, treatment.

Delegate monitoring to `babysit-zephyr`. A failed or preempted arm is evidence
about infrastructure reliability, not a performance verdict. Diagnose repeated
failures with `debug`.

## Collect execution IDs

A benchmark job can run many Zephyr pipelines on one shared pool. Collect every
`YYYYMMDD-HHMMSS-<hex>` execution ID from the root job and descendant logs:

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

Aggregate all executions in each arm, then compare by `stage_name`:

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

Use this SQL as the Finelog query body after replacing each ID placeholder with
comma-separated, single-quoted execution IDs. Keep the raw query output with
the report.

## Validate comparability

Before interpreting deltas:

1. Confirm the workload fingerprints match except for SHA, arm, and run tag.
2. Confirm each stage has matching `items` and `bytes_processed`, within a
   fraction of a percent. Explain and normalize any accepted mismatch.
3. Confirm both arms completed the same execution and stage set.
4. Inspect `iris job summary <IRIS_JOB_ID>` for OOMs and peak task memory.
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
Treatment: <sha>, <job>, <execution count>

| Stage | CPU control | CPU treatment | CPU change | Elapsed control | Elapsed treatment | Elapsed change | Peak memory change |
|---|---:|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... | ... |

Data check: <items and bytes comparison>
Infrastructure: <preemptions, retries, failures, stragglers, or none>
Interpretation: <efficiency result, latency result, and any tradeoff>
```

Lead with CPU change, then elapsed time and memory. State whether elapsed time
came from one pair or repeated alternating trials. Label summed stage elapsed
as such. Iris launcher duration and summed task wall time may help diagnose
queueing or topology, but they do not replace the Finelog stage metrics.

## Clean up

Remove temporary worktrees after preserving the SHAs, job IDs, execution IDs,
workload fingerprint, and Finelog output:

```bash
git worktree remove "$WORKTREE_ROOT/control"
git worktree remove "$WORKTREE_ROOT/treatment"
```

Benchmark outputs expire under their seven-day temporary prefix.

## Related guidance

- `babysit-zephyr` monitors the paired jobs through terminal state.
- `debug` investigates repeated failures or unexplained infrastructure churn.
- `lib/zephyr/OPS.md` documents coordinator queries and straggler diagnosis.
- `lib/iris/OPS.md` documents job summaries, task attempts, and Finelog access.
