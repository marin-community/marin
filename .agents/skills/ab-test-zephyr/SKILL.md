---
name: ab-test-zephyr
description: Run a Zephyr control and treatment on pre-normalized data and compare per-stage Finelog CPU, elapsed-time, and memory stats. Use for ad hoc comparisons and PR performance gates; add named treatments only when requested.
---

# A/B Test Zephyr Changes

Run one control and one treatment of `experiments.datakit.zephyr_benchmark` on
the same immutable, pre-normalized sample; collect every `zephyr.stage` row;
compare CPU, elapsed, and memory per stage; and publish the workload,
comparability checks, infrastructure noise, and verdict together. Add arms only
when the requester names them.

## Interpretation

Aggregate across executions by stage as follows:

| Metric | Aggregation | Use |
|---|---|---|
| `cpu_time_total` | sum | primary efficiency/cost; normalize per item/byte if work differs |
| `elapsed` | sum, labeled summed stage elapsed | barrier latency; sensitive to scheduling/stragglers |
| `items`, `bytes_processed` | sum | workload equivalence |
| `mem_peak_bytes_max` | max | worst shard RSS/OOM guardrail |
| `mem_bytes_avg`, `cpu_pct_avg` | weighted context only | do not sum as ordinary metrics |
| `item_rate`, `byte_rate` | do not aggregate | derived from noisy elapsed |

Keep CPU and elapsed conclusions separate. CPU flat/lower plus elapsed lower is
a latency/topology win without added compute; CPU higher plus elapsed lower is
faster but more expensive; CPU lower plus elapsed higher is cheaper but slower.
A wall-only change from one comparison is inconclusive until repeated under
comparable scheduling. Topology/batching changes must report the tradeoff.
Do not apply universal wall thresholds: calibrate against same-code repeats.
Any new OOM, application failure, or worker-limit memory peak is a regression.

## Select and prepare runs

For existing jobs, verify the same immutable sample, stage range, sources,
resources, concurrency, parallelism, cluster, region, and priority. A scheduled
baseline is usable only if its workload fingerprint matches and its execution
IDs remain queryable. Otherwise launch a matching control.

For new runs, default to merge-base control and current head treatment. For PRs,
choose the smallest stage range exercising the diff; skip remote work for docs,
tests, types, or log-only changes with reviewer agreement. Run local Zephyr and
Datakit tests first. Confirm sample, pool shape, stage range, target, and cost
before expensive or production-scale work.

Use GCP `europe-west4` and this sample by default; use the matching region/sample
or explicitly selected CoreWeave target when requested:

```sh
SAMPLE_PREFIX=gs://marin-eu-west4/datakit/sample_100b_8ae7a94f
DATA_LOCALITY_ARGS=(--region europe-west4)
# us-central1: gs://marin-us-central1/datakit/sample_100b_8ae7a94f / --region us-central1
# CoreWeave: s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f / --target-cluster cw-us-east-02a
```

If the sample-to-region mapping is unknown, stop. For a PR use detached
worktrees at the merge base and treatment SHA, record both SHAs, and use fresh
run tags:

```sh
git fetch origin main
BASELINE_SHA=$(git merge-base origin/main HEAD)
TREATMENT_SHA=$(git rev-parse HEAD)
WORKTREE_ROOT=$(mktemp -d /tmp/zephyr-ab.XXXXXX)
git worktree add --detach "$WORKTREE_ROOT/control" "$BASELINE_SHA"
git worktree add --detach "$WORKTREE_ROOT/treatment" "$TREATMENT_SHA"
```

All arguments except `--run-tag` must match across arms. Tags must be fresh so
no arm cache-hits. Launch from each worktree:

```sh
cd <CONTROL_OR_TREATMENT_WORKTREE>
uv run iris --config=lib/iris/config/marin.yaml job run --no-wait \
  --job-name zephyr-ab-<RUN_TAG>-<ARM> "${DATA_LOCALITY_ARGS[@]}" \
  --memory=2G --disk=5G --cpu=1 --extra=cpu --priority batch -- \
  python -m experiments.datakit.zephyr_benchmark \
  --sample-prefix "$SAMPLE_PREFIX" --sources <SOURCES_OR_ALL> \
  --run-tag <FRESH_RUN_TAG>-<ARM> --pool-workers <WORKERS> \
  --pool-cpu <CPU_PER_WORKER> --pool-ram <RAM_PER_WORKER> \
  --pool-disk <DISK_PER_WORKER> --first-stage <STAGE> --last-stage <STAGE> \
  --max-concurrent <PIPELINES> --dedup-max-parallelism <SHARDS>
```

Record SHA, Iris job ID, sample/source, stage range, pool resources, concurrency,
cluster/region, priority/preemptibility, and run tag for every arm. Use
`babysit-zephyr` for monitoring; failed/preempted runs are infrastructure
evidence, not performance verdicts. Use `debug` for repeated failures.
When elapsed time determines the verdict, interleave additional control trials
among treatments to measure scheduling noise.

## Collect and query

Collect every execution ID from root and descendant logs:

```sh
uv run iris --cluster marin job logs <IRIS_JOB_ID> --max-lines 200000 --no-tail --level info | \
  rg -o '[0-9]{8}-[0-9]{6}-[0-9a-f]{8}' | sort -u
```

Read `lib/zephyr/OPS.md` for child-job naming. Authenticate with `uv run iris
--cluster marin login` when needed, then query all IDs:

```sql
SELECT execution_id, stage_name, status, cpu_time_total, elapsed,
       items, bytes_processed, mem_peak_bytes_max, mem_bytes_avg, cpu_pct_avg
FROM "zephyr.stage"
WHERE execution_id IN (<CONTROL_AND_TREATMENT_IDS>)
ORDER BY execution_id, stage_name
```

Every expected row must be `status = 'END'`; a `FAILED` row or missing execution
invalidates that arm until resolved. Aggregate all executions per arm with
`SUM(cpu_time_total)`, `SUM(elapsed)`, `SUM(items)`, `SUM(bytes_processed)`,
and `MAX(mem_peak_bytes_max)`, grouped by `stage_name`; compare each treatment
with the same control IDs. Keep raw query output and repeated trials separate.

Before a verdict, verify fingerprints, matching items/bytes (or explain and
normalize a small mismatch), matching execution/stage sets, OOMs via
`iris job describe`, and logs for retries, preemptions, faults, and stragglers.
Run semantic output validation separately. Different work, failures, or
material infrastructure churn means inconclusive; rerun.

## Report and clean up

Update one sentinel-marked PR comment so reruns replace stale results:

```markdown
<!-- zephyr-ab-test -->
🤖 ## Zephyr A/B test
Verdict: pass | regression | tradeoff | inconclusive
Workload: <sample, stages, sources, pool, concurrency, target>
Control: <sha>, <job>, <execution count>
Treatments: <name, sha, job, execution count>
Data check: <items/bytes>
Infrastructure: <retries, preemptions, failures, stragglers, or none>
Interpretation: <CPU efficiency, summed stage elapsed, memory, tradeoff>
```

Include per-stage CPU, summed elapsed, and peak-memory deltas; identify whether
elapsed uses one comparison or repeated interleaved trials. Do not substitute
launcher duration for Finelog stage metrics.

Preserve SHAs, IDs, fingerprints, and query output, then remove temporary
worktrees. Benchmark outputs expire under the seven-day temporary prefix:

```sh
git worktree remove "$WORKTREE_ROOT/control"
git worktree remove "$WORKTREE_ROOT/treatment"
```

See `babysit-zephyr`, `debug`, `lib/zephyr/OPS.md`, and `lib/iris/OPS.md` for
monitoring, failures, coordinator queries, and Finelog access.
