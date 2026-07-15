---
name: ab-test-zephyr
description: A-B test a Zephyr pipeline change using per-stage finelog stats instead of wall time. Use when comparing two Zephyr runs (baseline vs. treatment) to judge whether a code or config change made a stage faster, cheaper, or lighter.
---

# Skill: A-B Test Zephyr Changes

Compare two Zephyr runs — a baseline (control) and a treatment — to decide
whether a change helped, hurt, or did nothing. The signal lives in the
per-stage finelog stats emitted by the coordinator, **not** in stage wall time.

## Do not compare stage wall time

The single most common mistake is comparing how long a stage *took*. Stage wall
time (`elapsed`) and everything derived from it (`item_rate`, `byte_rate`) is
confounded by factors that have nothing to do with your change:

- **Worker count / autoscaling** — the same work over 200 vs. 400 workers halves
  wall time with zero code change.
- **Preemption and retries** — a preempted shard restarts from scratch; its stage
  wall time balloons.
- **Stragglers and data skew** — one slow shard sets the stage barrier; the other
  199 finished long ago (see `lib/zephyr/OPS.md` → Straggler Detection).
- **Queue / scheduling waits** — capacity contention delays shard starts.

Two back-to-back runs of *identical* code routinely differ 10–30% in stage wall
time from these effects alone. Treat `elapsed`, `item_rate`, and `byte_rate` as
context, never as the verdict. If someone reports "my change made stage X 15%
faster" citing wall time, that number is almost certainly noise until confirmed
by CPU time.

## Use CPU time and resource footprint instead

The coordinator writes one `zephyr.stage` row per stage at completion, keyed by
`execution_id`. These fields are aggregated from per-shard samples with fixed
semantics (`lib/zephyr/src/zephyr/runners.py:197-202`) and are the ones that
actually reflect a code change:

| Field | Aggregation | What it measures | Use for A-B |
|---|---|---|---|
| `cpu_time_total` | **sum** of per-shard CPU-seconds | total compute work done | **primary signal** — scheduling-independent |
| `items` | sum | total items processed | sanity: must match across A/B |
| `bytes_processed` | sum | total bytes processed | sanity: must match across A/B |
| `mem_peak_bytes_max` | max across shards | worst-case shard RSS | memory regressions / OOM risk |
| `mem_bytes_avg` | avg across shards | typical shard RSS | memory footprint |
| `cpu_pct_avg` | avg across shards | utilization | context (are workers CPU-bound?) |
| `elapsed`, `item_rate`, `byte_rate` | wall-time derived | scheduling noise | **avoid — see above** |

`cpu_time_total` is total CPU-seconds summed over every shard, so it does not
care how many workers ran or how they were scheduled. If your change makes a
stage cheaper, `cpu_time_total` drops; if it regresses, it rises. That is the
number to report.

**Validate the comparison first.** `cpu_time_total` is only comparable when the
two runs did the same work — confirm `items` and `bytes_processed` match (within
a fraction of a percent) between A and B before trusting any delta. When the
inputs differ slightly (e.g. a sampled dataset), normalize:
`cpu_time_total / items` (CPU-seconds per item) or `cpu_time_total /
bytes_processed` is the input-independent cost. A change that lowers CPU-time
per item is a real efficiency win even if total CPU time rose because it
processed more data.

For a change that trades CPU for memory (or vice versa), read both
`cpu_time_total` and `mem_peak_bytes_max` — a stage that is 5% cheaper on CPU
but pushes peak RSS past the worker limit is a regression, not a win.

## Workflow

### 1. Find each run's `execution_id`

`zephyr.stage` rows are keyed by `execution_id` (a `YYYYMMDD-HHMMSS-<hex>`
string), not by Iris job id. The coordinator logs it on startup:

```bash
uv run iris job logs <COORD_JOB_ID> --max-lines 2000 --no-tail | grep -i "execution_id="
# -> Coordinator job starting: name=..., execution_id=20260715-050040-1a2b3c4d, ...
```

See `lib/zephyr/OPS.md` → "Child job naming" for locating the `-coord` child
job. Record the baseline and treatment `execution_id`s.

### 2. Pull the per-stage stats

Query the `zephyr.stage` namespace through the finelog CLI (it tunnels to the
cluster's finelog deployment; see `lib/finelog/OPS.md`). Authenticate once with
`uv run iris --cluster marin login` if you have not already.

```bash
uv run finelog query marin --format table '
  SELECT execution_id, stage_name, cpu_time_total, items, bytes_processed,
         mem_peak_bytes_max, elapsed
  FROM "zephyr.stage"
  WHERE status = '"'"'END'"'"'
    AND execution_id IN ('"'"'<BASELINE_ID>'"'"', '"'"'<TREATMENT_ID>'"'"')
  ORDER BY stage_name, execution_id'
```

A stage that failed writes a `status = 'FAILED'` row; filter to `END` for clean
comparisons, or drop the filter to see where a run died.

### 3. Compare per stage

Join the two runs on `stage_name` and compute the deltas that matter:

```bash
uv run finelog query marin --format table '
  SELECT b.stage_name,
         t.cpu_time_total  - b.cpu_time_total                       AS cpu_delta,
         (t.cpu_time_total - b.cpu_time_total) / b.cpu_time_total   AS cpu_pct,
         t.items - b.items                                          AS items_delta,
         t.mem_peak_bytes_max - b.mem_peak_bytes_max                AS mem_peak_delta
  FROM "zephyr.stage" b
  JOIN "zephyr.stage" t USING (stage_name)
  WHERE b.execution_id = '"'"'<BASELINE_ID>'"'"'
    AND t.execution_id = '"'"'<TREATMENT_ID>'"'"'
    AND b.status = '"'"'END'"'"' AND t.status = '"'"'END'"'"'
  ORDER BY cpu_pct DESC'
```

Read it as:

- `items_delta` / `bytes` near zero → the runs are comparable; trust the CPU
  numbers. Non-trivial `items_delta` → normalize to CPU-seconds per item first.
- `cpu_pct` is the headline: negative means the treatment did less compute work.
- `mem_peak_delta` guards against a CPU win that blows the memory budget.

### 4. Report

Lead with the per-stage `cpu_time_total` delta (absolute and %), note memory,
and state the `items`/`bytes` match that makes the comparison valid. Mention
wall time only as context, explicitly flagged as noisy. One noisy run is not a
result — if a stage shows preemptions or a large `items` mismatch, re-run before
concluding.

## When per-run stats are not enough

A single A-B pair with visible preemption or straggler churn is inconclusive.
For a gated, repeatable comparison on a PR that touches `lib/zephyr` internals,
use **evaluate-zephyr-perf** — it submits a treatment ferry against a scheduled
baseline and applies a threshold table. This skill is for ad-hoc comparisons of
two runs you already have.

## Composes with

- `babysit-zephyr` — launch and monitor the two runs being compared.
- `evaluate-zephyr-perf` — formal PR perf gate against a scheduled baseline.
- `lib/zephyr/OPS.md` — coordinator queries, straggler/skew diagnosis, child-job
  naming for locating the `-coord` job.
