---
date: 2026-07-26
system: coreweave
severity: outage
resolution: investigating
pr: none
issue: none
---

# GB200 log-shipping outage + NCCL clique-init failure wave (cw-us-east-08a)

## TL;DR

- Two concurrent GB200 incidents on cw-us-east-08a, both first visible ~20:40-21:00 UTC
  2026-07-25 and ongoing at 06:30 UTC 2026-07-26:
  1. Task stdout from GB200 (grug-train child) pods stopped reaching the log server.
     `iris job logs <child-task>` returns 0 lines; finelog SQL `select ... from log`
     shows 0 rows for the same keys. Parent (CPU-node, executor) logs flow fine.
     ALL evening jobs affected (multiple users), jobs from before ~20:40 fully readable.
  2. Since ~02:00 UTC, every new 16-task GB200 gang fails or hangs at NCCL/JAX clique
     init (0% GPU indefinitely; `133 (SIGTRAP)`, "tried to connect with a different
     incarnation"). Four independent agents' jobs affected; the last job known to train
     started 02:15. Job-internal gang retries did NOT escape; a fresh allocation did not
     either.
- No user-side fix found; treated as provider-side fabric/agent incident. Training
  measurements completed before the outage are intact but unreadable until log shipping
  recovers; new rack work is blocked until clique formation recovers.
- Lingering caveat: whether worker-side log buffers survive a long outage (vs. drop on
  overflow) is unknown — if they drop, the affected runs' per-step metrics are lost.

## Original problem report

During EP25 round-6 probes, `iris job logs` on a freshly completed grug-train child task
returned 0 lines while the same command served a 6-hour-old job fine:

```
iris --cluster=marin job logs /mwittmann/<job>/grug-train-<job>/0 --max-lines 50   # -> 0 lines, exit 0
```

Later, rack jobs stopped booting entirely: 7+ consecutive attempt failures of the form
`133 (SIGTRAP) ... tried to connect while it is already in error` and gangs sitting at
0% GPU for 45+ minutes.

## Investigation path

1. Verified the job itself succeeded (`iris job summary` -> `State: succeeded`), so the
   empty logs were a serving problem, not a run failure. Checked provenance via
   `iris query ... job_config` -> `environment_json` -> `MARIN_PROVENANCE` (base_commit
   + env) to confirm the intended code/env actually ran — a usable substitute for the
   "sentinel line" check when logs are dark.
2. Established scope: parent executor tasks (CPU nodes) served logs; GB200 child tasks
   did not — for EVERY user's evening jobs (d4, rav, mine). Pre-20:40 jobs served both.
   => cluster-side ingestion gap for one node family, not a per-job problem.
3. Tried the finelog SQL path (bypasses the batch fetch path):
   `IrisClient.remote(...)._cluster_client._log_client.query("select ... from log where key like ...")`.
   Worked for old jobs, 0 rows for affected ones => the data never left the workers;
   two serving paths share one broken source. (Valid fields: log.key, log.source,
   log.data, log.epoch_ms, log.level, log.cluster, log.seq — `timestamp_ms` does not
   exist; use `epoch_ms`.)
4. Dead end: in-container stdout is a pipe (`/proc/1/fd/1 -> pipe:[...]`), no local file
   copy to read via `iris task exec`.
5. Dead end: per-task telltale endpoints (`iris endpoints list <job>`) 403 through
   proxy-minted capability URLs ("endpoint-scoped token cannot access this endpoint" on
   `/`, `/metrics`, `/health`). BYPASS THAT WORKS: `iris task exec <task> -- curl
   http://<pod-ip:port>/metrics` from inside the pod — serves `levanter_step`,
   `levanter_train_loss`, `levanter_moe_drop_fraction`,
   `levanter_throughput_tokens_per_second` gauges. Only while the task is alive; latest
   values only, so poll on a cadence shorter than the step time you need to resolve.
6. NCCL wave: v2 rerun cycled 7 attempts (`failed 133 (SIGTRAP)` then repeated
   `cosched_failed ... bounced for atomic re-scheduling`) over 2.5h, never escaping;
   stopped it and resubmitted fresh (fa4lse boot-hang recipe: new
   `JAX_COMPILATION_CACHE_DIR`, new allocation) — v3 attempt 0 failed identically, so the
   "cursed allocation" theory died: the fabric itself was failing clique formation for
   everyone by then (d1/d4/rav jobs all 0% GPU or churning; rav's "running" job read 0%
   GPU via a read-only `task exec ... nvidia-smi`).
7. Watched retry budgets: parent `max_retries_preemption=1000` keeps a zombie gang
   cycling at 0% GPU roughly forever (cheap but useless); the child job's failure budget
   (11) eventually trips terminal `failed` on incarnation-mismatch errors
   ("3 unexpectedly tried to connect with a different incarnation").

## Root cause

Unknown (provider side). Two GB200 node-agent/fabric functions degraded within the same
evening on the same cluster: log shipping (~20:40) and distributed-clique formation
(~02:00). The correlation suggests a common agent/nodepool roll or fabric event, but no
user-visible evidence pins it. All user jobs and configs were exonerated by
before/after comparisons (identical config booted 20:55, failed 01:50+).

## Fix

None available user-side. Mitigations used: poll-based measurement capture via in-pod
telltale (`experiments/grug/moe/telltale_poll.sh` in the ep25-d3 worktree) for running
tasks; provenance-based validation for completed ones; submissions paused pending
recovery.

## How OPS.md could have shortened this

- `lib/iris/OPS.md` "Job Management" section: add a "logs dark for a live/completed job"
  diagnostic ladder — (1) `iris job summary` to separate run failure from log failure;
  (2) compare `job logs` on the parent vs the child task and on an old job, to tell
  per-job vs cluster-wide ingestion gaps; (3) finelog SQL cross-check via
  `IrisClient.remote(url, credentials=...)._cluster_client._log_client.query(...)` with
  `epoch_ms` (not `timestamp_ms`) as the time column; (4) for RUNNING tasks, in-pod
  telltale: `iris endpoints list <job>` then `iris task exec <task> -- curl
  http://<addr>/metrics`.
- `lib/iris/OPS.md` job-run gotchas: note that proxy-minted telltale capability URLs
  currently 403 on every app path; use the in-pod route instead.
- `lib/iris/OPS.md` boot-failure guidance: for repeated gang `cosched_failed` /
  `133 (SIGTRAP)` at NCCL init, distinguish "cursed allocation" (fresh submission with a
  new compile-cache dir escapes) from "fabric incident" (fresh submission fails
  identically) — the second attempt's outcome discriminates them in one cycle, and the
  check costs one `iris job list --state running` plus a 0%-GPU read on a peer's gang.

## Artifacts

- ep25-d3 worktree: `experiments/grug/moe/telltale_poll.sh` (self-healing telltale
  poller; re-resolves the endpoint each cycle after a reschedule changed the address),
  `experiments/grug/moe/harvest_ep25.py` (finelog-SQL metrics harvester; reproduces
  published reference numbers exactly).
- AGENT_LOG.md on branch agent/ep25-d3-qbprobes: full incident timeline 2026-07-25
  20:40 UTC -> 2026-07-26 06:30 UTC with per-job evidence.
