---
name: ab-test-zephyr
description: Run an explicitly requested Zephyr control/treatment benchmark on the same pre-normalized sample and compare Finelog stage metrics.
---

# A/B test Zephyr changes

Use this workflow only when the requester explicitly asks for a Zephyr A/B
benchmark. Compare one control at the branch merge base with one treatment at
the branch head unless the requester names additional commits or
configurations.

## Define the comparison

Before launching, record the exact SHAs and choose the smallest target that
exercises the change:

| Change | Minimum coverage |
|---|---|
| Map, serialization, or tokenization | Affected stage on enough shards to amortize startup |
| Shuffle, partitioning, spill, merge, or buffers | Exact or MinHash through fuzzy dedup on skewed or production-shaped data |
| Shared-pool lifecycle, scheduling, or pipeline concurrency | All affected stages with representative concurrent sources |
| Documentation, tests, types, or log text only | Skip the remote run with reviewer agreement |

Existing jobs are reusable only when the control and every treatment have the
same immutable sample, sources, target, worker shape, concurrency, logical
parallelism, cluster, region, priority, and preemptibility. A scheduled ferry
is not a control for a differently shaped standalone benchmark.

Confirm the sample, preset, target, expected cost, and launch authority before
an expensive run. Run local Zephyr and Datakit tests first.

For a PR, prepare detached worktrees:

```bash
git fetch origin main
BASELINE_SHA=$(git merge-base origin/main HEAD)
TREATMENT_SHA=$(git rev-parse HEAD)
WORKTREE_ROOT=$(mktemp -d /tmp/zephyr-ab.XXXXXX)
git worktree add --detach "$WORKTREE_ROOT/control" "$BASELINE_SHA"
git worktree add --detach "$WORKTREE_ROOT/treatment" "$TREATMENT_SHA"
```

Keep configuration-only arms in separate commits or worktrees. Do not infer
unnamed variants.

## Launch

`experiments.datakit.zephyr_benchmark` reads an existing normalized sample and
writes to a seven-day temporary prefix. Use the same immutable, region-local
sample for every arm.

### Presets

| Preset | Selection | Expected workload | Workers | Pipelines | Dedup shards | CC rounds |
|---|---|---|---|---:|---:|---:|
| Full | `--sources all` | 115 sources, 256 GB, 768 shards | 48 | 128 | 1000 | 3 |
| Light | `--source-fraction 0.1` | 77 sources, 26.8 GB, 165 shards | 48 | 80 | 500 | 3 |

Both presets use the reference worker default: 2 CPU, 16 GiB RAM, and 16 GiB
disk. Datakit stages use their normal task-resource defaults; the benchmark
does not configure map or reduce task shapes. Override worker resources only
when worker shape is the treatment, and record the override.

The light selector greedily adds whole sources by ascending average shard size
until it reaches the byte target. The launcher logs the resolved source, byte,
and shard counts and rejects a pool larger than the selected parquet shard
count. Treat the table counts as an expected fingerprint and investigate a
material mismatch.

Use the full preset for shared-pool, shuffle, spill, buffer, or scheduling
changes. Use light for a quicker stage-local signal.

### Targets

- `all`: exact dedup, tokenization, MinHash, and fuzzy dedup.
- `map`: tokenization and MinHash.
- `shuffle`: exact and fuzzy dedup using permanent sample-owned MinHash inputs.
- `exact`, `tokenize`, `minhash`, or `fuzzy`: one stage family.

Before a `shuffle` or `fuzzy` run, the launcher verifies every selected
source's MinHash artifact. If one is missing, follow
`experiments/datakit/README.md` to run the sample materializer in `minhash`
mode. Map benchmark outputs are temporary and cannot feed a later shuffle run.

### Data locality

Set one matching sample and compute location:

```bash
# Default GCP run.
SAMPLE_PREFIX=gs://marin-eu-west4/datakit/sample_100b_8ae7a94f
DATA_LOCALITY_ARGS=(--region europe-west4)

# GCP us-central1 alternative.
# SAMPLE_PREFIX=gs://marin-us-central1/datakit/sample_100b_8ae7a94f
# DATA_LOCALITY_ARGS=(--region us-central1)

# CoreWeave alternative.
# SAMPLE_PREFIX=s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f
# DATA_LOCALITY_ARGS=(--target-cluster cw-us-east-02a)
```

Stop if the sample's storage location is unknown. Do not override the output
location or run compute in another region.

### Command

Run the same command from each worktree, changing only the arm, SHA, and fresh
run tag:

```bash
cd <CONTROL_OR_TREATMENT_WORKTREE>
uv run iris --config=lib/iris/config/marin.yaml job run --no-wait \
  --job-name zephyr-ab-<RUN_TAG>-<ARM> \
  "${DATA_LOCALITY_ARGS[@]}" --memory=2G --disk=5G --cpu=1 --extra=cpu \
  --priority batch \
  -- python -m experiments.datakit.zephyr_benchmark \
    --sample-prefix "$SAMPLE_PREFIX" \
    --sources <COMMA_SEPARATED_SOURCES_OR_ALL> \
    `# use --source-fraction <FRACTION> instead for a fractional preset` \
    --run-tag <FRESH_RUN_TAG>-<ARM> \
    --pool-workers <WORKERS> \
    --target <all|map|shuffle|exact|tokenize|minhash|fuzzy> \
    --max-concurrent <PIPELINES> \
    --dedup-max-parallelism <SHARDS> \
    --dedup-cc-max-iterations <ROUNDS>
```

Record this workload fingerprint for every arm:

- commit SHA, Iris job ID, and run tag
- sample prefix, source selection, and benchmark target
- worker count, CPU, RAM, and disk
- maximum pipelines, dedup parallelism, and connected-components rounds
- Iris controller, region or target cluster, priority, and preemptibility

Fresh run tags prevent cache hits. If elapsed time drives the decision, repeat
and interleave control trials to measure scheduling noise.

Use `babysit-zephyr` when continuous monitoring is requested. A failed or
preempted arm has no performance result. Use `debug` only for a repeated fault.

## Collect and validate results

Collect every execution ID from the root and descendant logs:

```bash
uv run iris --cluster marin job logs <IRIS_JOB_ID> \
  --max-lines 200000 --no-tail --level info | \
  rg -o '[0-9]{8}-[0-9]{6}-[0-9a-f]{8}' | sort -u
```

See `lib/zephyr/OPS.md` for child-job naming. Preserve each arm's ID list with
its workload fingerprint.

Query Finelog's `zephyr.stage` rows after authenticating with
`uv run iris --cluster marin login` when necessary:

```bash
uv run finelog query marin --format table '
  SELECT execution_id, stage_name, status, cpu_time_total, elapsed,
         items, bytes_processed, mem_peak_bytes_max, mem_bytes_avg, cpu_pct_avg
  FROM "zephyr.stage"
  WHERE execution_id IN (<CONTROL_AND_TREATMENT_IDS>)
  ORDER BY execution_id, stage_name'
```

Every expected row must have `status = 'END'`. Resolve missing rows and exclude
arms with failed stages before interpreting deltas.

Aggregate each arm by stage:

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
), totals AS (
  SELECT arm, stage_name,
         SUM(cpu_time_total) AS cpu,
         SUM(elapsed) AS elapsed,
         SUM(items) AS items,
         SUM(bytes_processed) AS bytes,
         MAX(mem_peak_bytes_max) AS mem_peak
  FROM tagged
  GROUP BY arm, stage_name
)
SELECT c.stage_name,
       c.cpu AS control_cpu, t.cpu AS treatment_cpu,
       (t.cpu - c.cpu) / NULLIF(c.cpu, 0) AS cpu_delta,
       c.elapsed AS control_elapsed, t.elapsed AS treatment_elapsed,
       (t.elapsed - c.elapsed) / NULLIF(c.elapsed, 0) AS elapsed_delta,
       t.items - c.items AS items_delta,
       t.bytes - c.bytes AS bytes_delta,
       c.mem_peak AS control_mem_peak, t.mem_peak AS treatment_mem_peak
FROM totals c JOIN totals t USING (stage_name)
WHERE c.arm = 'control' AND t.arm = 'treatment'
ORDER BY cpu_delta DESC;
```

Run the aggregate once per treatment and retain the raw output. Keep repeated
trials separate.

Interpret the signals as follows:

| Signal | Use |
|---|---|
| Sum of `cpu_time_total` | Primary efficiency and compute-cost result |
| Sum of `elapsed` | Secondary stage latency; includes startup, I/O, queueing, and stragglers |
| Sum of `items` and `bytes_processed` | Workload-equivalence check |
| Max of `mem_peak_bytes_max` | Worst shard RSS and OOM guardrail |
| `mem_bytes_avg`, `cpu_pct_avg` | Weighted diagnostic context only |
| `item_rate`, `byte_rate` | Do not aggregate |

CPU and elapsed are separate outcomes. Lower elapsed with higher CPU is a
latency/cost tradeoff, not an efficiency win. Treat a wall-only result from one
comparison as inconclusive. Normalize CPU per item or byte for an accepted
workload mismatch. A new OOM, application failure, or memory peak above the
worker limit is a regression.

Before assigning a verdict:

1. Match fingerprints except for SHA, arm, and run tag.
2. Match execution and stage sets.
3. Match items and bytes within a fraction of a percent, or explain and
   normalize the difference.
4. Inspect `iris job describe <IRIS_JOB_ID>` for failures, preemptions, and
   memory peaks.
5. Check logs for retries, hardware faults, and stragglers.
6. Validate semantics separately; equal counts do not prove equal output.

Different work or material infrastructure churn makes the comparison
inconclusive and requires a clean rerun.

## Report and clean up

Update one sentinel-marked PR comment:

```markdown
<!-- zephyr-ab-test -->
🤖 ## Zephyr A/B test

Verdict: pass | regression | tradeoff | inconclusive
Workload: <sample, target, sources, pool, concurrency, cluster>
Control: <sha, job, execution count>
Treatments: <name, sha, job, execution count>

| Treatment | Stage | CPU change | Summed elapsed change | Peak memory change |
|---|---|---:|---:|---:|
| ... | ... | ... | ... | ... |

Data check: <items and bytes>
Infrastructure: <preemptions, retries, failures, stragglers, or none>
Interpretation: <CPU result, latency result, and tradeoff>
```

Lead with CPU, then elapsed and memory. State whether elapsed comes from one
comparison or repeated trials. Launcher duration does not replace stage
metrics.

After preserving SHAs, job IDs, execution IDs, fingerprints, and query output:

```bash
git worktree remove "$WORKTREE_ROOT/control"
git worktree remove "$WORKTREE_ROOT/treatment"
```

Benchmark outputs expire with their seven-day temporary prefix. Use
`lib/zephyr/OPS.md` for coordinator diagnostics and `lib/iris/OPS.md` for job,
task-attempt, and Finelog operations.
