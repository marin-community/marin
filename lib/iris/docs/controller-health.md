# Iris Controller Health Check

Step-by-step entry point for oncall when the Iris controller is suspected to be slow, stuck, or unreachable. The flow is:

1. **Check vitals.** If they pass, the controller is fine. Stop.
2. **If vitals fail, capture evidence.**
3. **Evaluate the evidence** — pattern-match first, escalate to LLM if no match.
4. **Act on the hypothesis.**

For the full operations reference (SQL schemas, restart procedure, known bugs) see [`OPS.md`](../OPS.md).

## Phase 1 — Check vitals

Run all four. If everything passes, the controller is healthy. Skip Phases 2-4.

### Status and worker count

```bash
uv run iris --cluster marin cluster status
```

- Healthy: `Running: True`, `Healthy: True`, expected worker count (e.g. `566/566 healthy`).
- Worrying: `Running: False`, low or zero workers, command itself errors.
- Caveat: this RPC is cheap. A busy controller still answers it; don't stop at this one alone.

### RPC stats dashboard

Open <https://iris.oa.dev/#/status>.

- Healthy: per-RPC latency steady, low error rate, no growing queues.
- Worrying: rising p99 latency on `launch_job` / `StartTasks` / `fetch_logs`, error rate climbing, in-flight requests piling up.

### Latency probe

```bash
time uv run iris --cluster marin query "SELECT 1"
```

The single most diagnostic command for "alive but slow." Exercises a DB-backed RPC that the cheap status RPC skips.

- Healthy: returns in <2s.
- Worrying: >10s or `DEADLINE_EXCEEDED`.

### Dummy job

```bash
uv run iris --cluster marin job run --cpu 1 --memory 2g -- echo hello
```

Tests the whole submission → scheduling → start path end-to-end.

- Healthy: lands and prints `hello` within seconds.
- Worrying: submission times out, task stuck `assigned`, or `StartTasks RPC failed: Request timed out`.

## Phase 2 — Capture evidence

You enter this phase only after vitals fail. Gather these artifacts in order. Each adds independent signal.

### Process state

```bash
uv run iris --cluster marin process status
```

CPU cores, RSS, thread count, open FDs, uptime.

- Worrying: ~1 full core sustained (Python GIL bound), threads climbing past ~200, RSS growing unbounded, FDs approaching ulimit.

```bash
uv run iris --cluster marin process logs --tail 500 --level WARNING
```

Recent warnings and errors. Scan for repeated retries, stack traces, autoscaler errors, RPC deadlines.

### Thread dump

```bash
uv run iris --cluster marin process profile threads
```

Single highest-value artifact. Read for stuck patterns. The documented one is `subprocess.run` → `terminate` on the reaper thread ([`OPS.md:199`](../OPS.md), bug #3678).

### CPU profile (if no stuck thread visible)

```bash
uv run iris --cluster marin process profile cpu -d 10
```

Writes a `.speedscope.json`. Open in speedscope.app or local viewer.

### Memory profile (if RSS is growing)

```bash
uv run iris --cluster marin process profile mem
```

Writes an HTML flamegraph.

### Subsystem state

```bash
uv run iris --cluster marin rpc controller get-scheduler-state
```

Pending queue, resource constraints.

```bash
uv run iris --cluster marin rpc controller get-autoscaler-status
```

Per-scale-group demand, `slice_state_counts`, `consecutive_failures`, `backoff_until`. Big and noisy; usually want to filter through `python3` / `jq` to the groups in active use.

```bash
uv run iris --cluster marin rpc controller get-provider-status
```

Provider-side scheduling events. Use when the autoscaler is failing and you need to know whether GCP/CW is rejecting requests.

### Worker fleet

```bash
uv run iris --cluster marin cluster vm status
```

Every slice and worker, with state. Scan for `VM_STATE_FAILED`, slices stuck `BOOTING` / `INITIALIZING`, or missing workers under a READY slice.

### Per-job evidence

If a specific job is stuck or failing, gather its history.

```bash
uv run iris --cluster marin job summary /user/job-name
```

```bash
uv run iris --cluster marin job bug-report /user/job-name
```

`bug-report` is strictly richer (attempt-by-attempt history with worker IDs and recent logs).

### VM-level state (when controller is unreachable, not just slow)

```bash
gcloud compute instances describe iris-controller-marin --zone=us-central1-a --format='value(status,statusMessage)'
```

## Phase 3 — Evaluate the evidence

### Pattern-match against known causes

| Symptom set | Likely cause | Conclusive check |
|---|---|---|
| Vitals: status fine, but query / dummy job times out | Controller alive but heavy handlers GIL-bound | Thread dump for any thread stuck in blocking I/O |
| Tasks stuck `ASSIGNED`, `last_heartbeat_ms` stale | Reaper thread stalled on synchronous `gcloud compute tpus tpu-vm delete` ([`OPS.md:199`](../OPS.md), bug #3678) | Thread dump for `subprocess.run` → `terminate` |
| `cluster status` itself unreachable | Controller process down or SSH tunnel broken | `gcloud compute instances describe` |
| RSS growing without bound | Memory leak in a long-running thread | Memory flamegraph |
| Autoscaler scale groups with high `consecutive_failures` and zero `ready` | GCP/CW reclaim or quota pressure on those zones | Provider status; not necessarily a controller bug |

Cross-reference [`OPS.md` § Known Bugs](../OPS.md) before concluding it's a new issue.

### If no pattern matches: escalate to an LLM

Dump the captured evidence to an LLM with enough context to form a hypothesis:

- Profiles, log excerpts, RPC dumps from Phase 2
- The contents of `lib/iris/OPS.md`
- Source code at the addresses identified in the thread dump

The LLM doesn't need to be right — it needs to give you a plausible hypothesis to test. Iterate.

## Phase 4 — Act on the hypothesis

| Diagnosis | Action | Reference |
|---|---|---|
| Alive but heavy RPCs slow; cause unidentified | Controller restart (seconds of downtime, workers and jobs survive) | [`OPS.md:21-23`](../OPS.md), [`restart-iris-controller` skill](../../../.claude/skills/restart-iris-controller/SKILL.md) |
| Reaper thread stalled on `gcloud` subprocess | Kill the stuck `gcloud` PID on the controller host (no restart needed) | [`OPS.md:199`](../OPS.md) |
| Controller VM down or unreachable | `iris cluster start` or escalate | [`OPS.md:9-11`](../OPS.md) |
| Workers down or missing under READY slices | Investigate the affected scale group via `cluster vm status` | [`OPS.md:91`](../OPS.md) |
| GCP-side preemption pressure on one zone | Move workload to a calmer zone or to reserved capacity; not a controller bug | — |
| Genuine controller backlog (deep queue, healthy threads) | Wait. The work is real. Restart would not help. | — |

## References

- [`lib/iris/OPS.md`](../OPS.md) — full operations reference, including SQL schemas and known bugs
- [`lib/iris/OPS.md:15-23`](../OPS.md) — controller restart procedure (dry-run → baseline → restart → verify)
- [`lib/iris/OPS.md:199`](../OPS.md) — reaper-thread stall on synchronous gcloud subprocess (#3678)
- <https://iris.oa.dev/#/status> — live RPC stats dashboard
- [`restart-iris-controller` skill](../../../.claude/skills/restart-iris-controller/SKILL.md) — canonical restart procedure
