---
name: query-inference-metrics
description: Investigate a vLLM serve's performance (throughput, TTFT/TPOT latency, queue depth, KV-cache usage) from durable finelog metrics with SQL. Use when asked how an inference or eval serve performed, or why it was slow. For kernel-level JAX profiling, use profile-training instead.
---

# Skill: Query Inference Metrics

A vLLM serve mirrors its `/metrics` into `rigging.telltale`, which forwards to
finelog every 15s (#7349), so serving metrics stay queryable after the job ends.
Training runs land in the same table under `source = 'levanter'`.

This is a triage map, not an inventory. Two queries give the bearing; the three
traps below let you write the rest.

## Running queries

From a Marin checkout:

```sh
uv run finelog query marin "<SQL>" --format table
```

- `marin` is the federated hub — almost always the right target. Per-cluster
  deployments (`cw-us-east-02a`, …) hold no `telltale` table.
- Double-quote the SQL, single-quote inside it, leave `telltale` unquoted.
- `IapLoginRequired` means log in (`lib/iris/OPS.md`). `--format jsonl|csv` to
  pipe; `--max-rows` raises the 100k cap.

## The data

One table, `telltale`, one row per sample. Typed, filterable columns: `name,
value, kind, ts, source, run, job_id, task_index, attempt, worker, region,
process_index`. Anything else — a histogram's `le`, `model_name`,
`finished_reason` — lives in a `labels` map read with `json_get(labels, 'key')`
(extract a key before you `GROUP BY`/`ORDER BY`; the map column itself errors).

Always filter on `name` first: finelog keys the table on it, so `WHERE name = '…'`
prunes by parquet stats; unfiltered scans are slow and can hit `Offset overflow`.
Match `name LIKE 'vllm:%'` for serving metrics — `source = 'vllm'` marks the
forwarding process, so it also tags ~10 stdlib `process_*`/`python_*` collectors.
Rows exist only for jobs run after the sink landed mid-July 2026 (#7336).

## Three ways to get a wrong number

1. **Series identity is `(job_id, worker, attempt)` plus the metric's own
   labels** (`engine`, `model_name`, `le`), not `job_id` alone. `run` is NULL on
   vLLM rows, and a serve spans replicas. Averaging a counter across replicas is
   meaningless.
2. **Counters are cumulative, per replica** — this is the rate pattern.
   `MAX(value)` per replica per time bin, `LAG` within the replica, then `SUM`
   across replicas. `GREATEST(delta, 0)` clamps a restart's negative jump but
   doesn't recover the tokens after it, so treat restarted series as approximate.
3. **Histograms are three families** — this is the quantile pattern. `_bucket`
   (cumulative, bound in `labels['le']`), `_count`, `_sum`; `CAST(le AS DOUBLE)`
   maps `+Inf` to `inf`. Find the bucket where cumulative `_count` crosses
   `q × total`; divide `_sum` by `_count` for a mean.

Stale-data trap: rows written before #7498 collapse
`vllm:prompt_tokens_by_source_total` into three indistinguishable series
(#7497); use `vllm:prompt_tokens_total` for the total on older jobs.

## Triage a serve

Examples use `/held/qwen3-evals-otbl-full-r2`, a Qwen3 eval serve. First see what
it recorded (`AND name LIKE '%cache%'` to narrow; read one row's `labels` for a
metric's dimensions):

```sql
SELECT name, kind, COUNT(*) AS samples
FROM telltale WHERE job_id = '/held/qwen3-evals-otbl-full-r2'
GROUP BY 1, 2 ORDER BY 1
```

**Lifetime totals** (the counter pattern, summing each counter's per-series max):

```sql
WITH peaks AS (
  SELECT name, worker, attempt, MAX(value) AS v
  FROM telltale
  WHERE job_id = '/held/qwen3-evals-otbl-full-r2'
    AND name IN ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
                 'vllm:e2e_request_latency_seconds_count',
                 'vllm:e2e_request_latency_seconds_sum', 'vllm:num_preemptions_total')
  GROUP BY 1, 2, 3
)
SELECT name, CAST(SUM(v) AS BIGINT) AS total FROM peaks GROUP BY 1 ORDER BY 1
```

```
vllm:e2e_request_latency_seconds_count | 57609
vllm:e2e_request_latency_seconds_sum   | 151792
vllm:generation_tokens_total           | 41163346
vllm:num_preemptions_total             | 0
vllm:prompt_tokens_total               | 25471239
```

Mean e2e latency = sum/count = 2.63s. `MIN(ts)`, `MAX(ts)`, and
`COUNT(DISTINCT worker)` on `vllm:generation_tokens_total` give the span and
replica count — here 2831 tok/s over 242 min on 2 replicas (every replica that
ran, not the steady-state count). The same pattern over
`vllm:request_{queue,prefill,decode}_time_seconds_sum` shows decode held 99% of
request time: generation-bound.

**Saturation** — did requests queue or KV fill? Peaks matter, not averages:

```sql
SELECT REPLACE(name, 'vllm:', '') AS gauge,
       ROUND(MAX(value), 3) AS peak, ROUND(AVG(value), 3) AS avg
FROM telltale
WHERE job_id = '/held/qwen3-evals-otbl-full-r2'
  AND name IN ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc')
GROUP BY 1 ORDER BY 1
```

```
kv_cache_usage_perc  | 0.151 | 0.06
num_requests_running | 16.0  | 9.421
num_requests_waiting | 7.0   | 0.019
```

Waiting peaked at 7 but averaged ~0 and KV never passed 15%, so the serve was
neither admission- nor memory-limited — just busy decoding long generations.
Gauges don't prove spare decode capacity, though; test a concurrency theory by
rerunning higher. `date_bin(INTERVAL '5 minutes', ts)` in the GROUP BY makes it a
timeline.

## Going deeper

The three traps are the whole toolkit. For a rate over time, delta a counter
(trap 2) across `date_bin` buckets. For a latency tail, apply the histogram
pattern (trap 3) to `vllm:time_to_first_token_seconds_bucket` (TTFT),
`vllm:inter_token_latency_seconds_bucket` (TPOT), or the e2e bucket. To split a
counter, `json_get` its label (e.g. `finished_reason` on
`vllm:request_success_total`) before the per-series `MAX` and `SUM`. Cross-check
independent numbers — a request total that disagrees with the histogram `_count`
usually means a counter got averaged across replicas instead of summed.
