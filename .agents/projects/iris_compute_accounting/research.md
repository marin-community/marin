# Research: weekly preemptible-compute accounting from Iris

In-repo findings that shaped the design. Source: GitHub issue
[#7353](https://github.com/marin-community/marin/issues/7353) and a four-way
source survey (finelog data model, prometheus/MFU telemetry, the weekly-summary
surface, user/issue attribution). Line numbers are from `main` at
`1f2b88682bf47c4ed848d917208fa32107084c5d`.

## What the issue asks for

Weekly, per-user (and ideally per-GitHub-issue) rollup of preemptible TPU
chip-hours, plus a preemption-overhead metric ("what are we actually paying for
preemptible"), dropped into the weekly team summary ("MWS") the team already
reads. Maintainer (`@rjpower`) added two hypotheses: (a) the missing primitive
is "an iris event `<task started/preempted/resumed/finished> <user> <time>
<resources>` which we could roll up"; (b) we already capture prometheus GPU/TPU
MFU utilization at the process level and can fold in downtime / MFU-per-job /
per-user breakdowns; the shape is "a script that preloads into DuckDB then
builds a dashboard."

## 1. Lifecycle / accounting data model

**The authoritative per-attempt lifecycle is the controller DB, not finelog.**
`task_attempts_table` — `lib/iris/src/iris/cluster/controller/schema.py:423`:
`task_id, attempt_id, worker_id, state (int TaskState), created_at_ms,
started_at_ms, finished_at_ms, exit_code, error, attempt_uid, backend_id`.
"Chip-seconds between start and stop" = `finished_at_ms - started_at_ms` per
attempt. FK cascade `task_attempts → tasks → jobs`, all `ondelete=CASCADE`
(`schema.py:426, 331, 256`).

**Preemption is derived from attempts** (the standalone counter was removed;
`job.proto:237` reserves it). `lib/iris/src/iris/cluster/controller/attempt_counts.py`:
`PREEMPTION_ATTEMPT_STATES = {WORKER_FAILED, KILLED, PREEMPTED}`
(`attempt_counts.py:46`). Preemption predicate = attempt state in that set
**and** `started_at_ms IS NOT NULL` (an ASSIGNED-phase kill is retried without
charging); failure = `state == FAILED` (`counts_from_attempts`,
`attempt_counts.py:76`; production SQL `attempt_counts.py:89`). Wasted-preempt
time = `Σ (finished_at_ms - started_at_ms)` over attempts matching the
predicate. `JobStatus.preemption_count`/`failure_count` are populated as
aggregates (`job.proto:342`).

**Retention wall.** `job_retention` defaults to **7 days** —
`lib/iris/src/iris/cluster/controller/controller.py:226`
(`Duration.from_seconds(7 * 86400)`). `_prune_terminal_jobs`
(`controller/pruner.py`) deletes terminal jobs older than that, cascading to
tasks + attempts. A "weekly" rollup that reads the controller DB directly sits
exactly on the prune boundary.

**Query surface.** `ListJobs` is the history paginator
(`controller.proto:853`; `JobQuery` `controller.proto:205`, `limit` max 500,
`total_count`/`has_more`). `ExecuteRawQuery` (`controller.proto:898`, handler
`service.py:2855`) is an **admin-only, SELECT-only** passthrough to the
controller DB — guarded at `service.py:2868` (`first token != "SELECT"` →
`INVALID_ARGUMENT`). Device shape per job is `job_config.res_device_json`
(`schema.py:293`); `JobStatus.resources` (`ResourceSpecProto`, `job.proto:405`)
carries `TpuDevice.count` / `GpuDevice.count`.

**`iris.task_event` is NOT the lifecycle stream rjpower hypothesized.**
`TaskEventRow` — `lib/iris/src/iris/cluster/worker/stats.py:169`:
`task_id, attempt_id, ts, type (severity), reason, message, source, count`.
Producer `TaskEventLog.observe` — `lib/iris/src/iris/cluster/backends/k8s/tasks.py:1905`
— writes only on verdict *change*, **k8s backend only**. It records k8s
scheduling/admission diagnostics (Kueue denial, image-pull, unschedulable), not
RUNNING/PREEMPTED/RESUMED transitions, carries no user/resources/node, and is
retained ~1 hour (`StoragePolicy max_age_seconds=3600`, `stats.py:51`). Unusable
for a weekly rollup and absent on GCE/TPU.

**`iris.provisioning`** — one durable row per slice provisioning outcome.
`IrisProvisioning` — `lib/iris/src/iris/cluster/controller/autoscaler/provisioning.py:39`:
`ts, resource_type, scale_group (key), zone, accelerator_variant, outcome
(ready/stockout/error/preempted), error_message, worker_count,
provision_latency_ms`. Written by `Autoscaler._record_provisioning_outcome`
(`autoscaler/runtime.py:307`). No age cap (durable). **No user/job/task field**;
preemptible-vs-reserved is encoded only as a substring in `scale_group`
(`...-preemptible_...`); region derivable only from `zone`. This is the only
durable preemption+provision-latency record, but it is slice/pool-level and
cannot be joined to a job/user without a slice→task mapping that isn't recorded.

## 2. Utilization / MFU telemetry

**MFU per job already lands durably in finelog and is offline-queryable.**
Levanter computes MFU (`compute_instant_throughput`,
`lib/levanter/src/levanter/callbacks/_metrics.py:50`; peak from
`fray.device_flops.DEVICE_FLOPS`, covers TPU v3/v4/v5litepod/v5p/v6e + NVIDIA)
and logs `throughput/mfu` plus `p10/p50/p90/mean/stddev_mfu`
(`_metrics.py:148`). The Levanter `Trainer` unconditionally composes a
`TelltaleConfig` (`lib/levanter/src/levanter/trainer.py:799`), so every scalar
is mirrored as a prometheus gauge and forwarded. The finelog sink is durable:
`TELLTALE_NAMESPACE = "telltale"`, `FinelogMetricSink` calls `get_table` with
**no** `storage_policy` (`lib/finelog/src/finelog/telltale.py:21,34`), inheriting
the server default retention. `TelltaleMetric` — `lib/rigging/src/rigging/telltale.py`:
`name, value, kind, ts, source, run, job_id, task_index, attempt, worker,
region, process_index, labels`. So MFU is queryable as
`SELECT ... FROM "telltale" WHERE name='levanter_throughput_mfu' GROUP BY job_id`.
Grafana already reads this namespace (`infra/grafana/dashboards/training.json`).

**Gaps for "MFU per job":** (1) no user/owner column on `telltale` — the job
namespace path encodes the user but it isn't split out; (2) forwarded from every
`process_index`, so a rollup must pick process 0 or average (else double-count);
(3) no chip count on the row (device-weighting needs the Iris job resource
spec); (4) coverage is **Levanter-training only** — vLLM inference, Zephyr, raw
JAX emit no `levanter_throughput_mfu`.

**Hardware duty cycle.** GPU duty cycle exists via DCGM
(`DCGM_FI_DEV_GPU_UTIL`) scraped into `iris.worker`
(`lib/iris/src/iris/cluster/backends/k8s/node_metrics.py:265,468`), but it is
**node-level, k8s/CoreWeave-only**. **TPU has no duty-cycle telemetry anywhere.**
`iris.task` (`IrisTaskStat`, `stats.py:129`, 5s GCE / 15s k8s) declares
`accelerator_util_pct` / `accelerator_mem_bytes` but **neither producer ever
populates them** (`task_attempt.py:948`, `k8s/tasks.py:1749`) — memory/CPU/disk
only. So for TPUs, MFU (Levanter) is the *only* "are the chips busy" signal;
allocation chip-hours can always be computed, utilization-weighting cannot be
universal.

**Downtime.** No durable per-attempt start/stop stream in finelog;
`started_at`/`finished_at` live in controller memory + the served proto
(`task_attempt.py:270,553`). Idle-between-preempt-and-resume is only
approximable today from gaps in `iris.task`/`telltale` `ts` streams at poll
granularity — which is exactly why a durable accounting row is the fix.

## 3. The weekly-summary surface ("MWS")

"MWS" is not a literal repo string; it maps to the **weekly team summary posted
to Discord `#internal-discuss`**. The live precedent is the weekly GCS storage
report: `scripts/ops/storage/generate_report.py` publishes a markdown report as
a secret gist and posts a compact week-over-week table to Discord, cron'd in
`.github/workflows/ops-storage-report.yaml` (`0 14 * * 1`, Mondays 14:00 UTC).
The reusable publish pattern (markdown → gist → Discord) is clearest in
`scripts/ops/egress_report.py` (`_create_gist`, `_post_to_discord`,
`_compose_message`, `--dry-run`; daily `ops-egress-report.yaml`). Discord helper:
`scripts/ops/discord.py` — `post(channel, message)`, channels
`internal-discuss` / `code-review`, webhook from `DISCORD_WEBHOOK_<CHANNEL>` env
or a gcloud secret. Cron always lives in GitHub Actions (`ops-*.yaml`); the
runner tunnels into Iris/finelog with the CI service account (no cluster-native
scheduler).

**Direct precedent for compute accounting:** the removed
`infra/github_wandb_metrics.py` (deleted in PR #6383) logged weekly
`Total GFLOPS` / `Total Petaflops Across Runs` to W&B `marin-monitoring`. Its
cron is gone — nothing to reuse, but it confirms "petaflops this week" was
previously a tracked number.

**Dashboard home:** `infra/grafana` is a Cloud Run Grafana + a bridge that runs
finelog `Query` SQL (`GET /finelog/{cluster}/query?sql=&from=&to=`), reviewed as
`dashboards/*.json`. No usage/compute dashboard exists yet; a
`dashboards/compute.json` over a compute namespace + `telltale` is the natural
"nice dashboard." A daily provider-billing pipeline already writes a
`cost.events` finelog namespace (`scripts/cost_manager/`,
`ops-cost-report.yaml`), but it has no run/user/issue dimension — disjoint from
per-user Iris compute.

## 4. User and issue attribution

**Two user identifiers per job, and they diverge on the marin cluster.**
- `jobs.user_id = JobName.user` (`schema.py:249`; `types.py:199`) — the friendly
  owner, set **client-side** by `resolve_job_user()`
  (`lib/iris/src/iris/cluster/client/job_info.py:148`): `--user` → `IRIS_USER`
  env → enclosing job's user → `getpass.getuser()` → `"root"`. `IRIS_USER` is
  never set anywhere; Fray/Marin never pass `user=`. In practice this is the OS
  login user of the submitting machine — **noisy** (`root`, `ubuntu`, laptop
  names). The controller *can* overwrite it with the authenticated email
  (`service.py:1399`) but only when `identity.role != "admin"`, and marin sets
  `unprovisioned_role: admin` (`lib/iris/config/marin.yaml:75`), so the override
  never fires. `ListUsers` (`controller.proto:874`, `service.py:905`) groups on
  this noisy field.
- `jobs.submitting_user` (`schema.py:255`; `job.proto:385`) — the **authenticated
  principal**, set server-side by `submitting_user_for_root` (`service.py:117`,
  called `service.py:1416`) for any authenticated identity regardless of role,
  else `local_admin` for anonymous/CIDR/loopback callers. Under marin's IAP auth
  it is the verified email; children inherit the root's value (`ops/job.py:176`),
  so a whole experiment tree stays attributed to the launching principal. **This
  is the clean key** — stable, human-meaningful — with one blind spot: it
  collapses to `local_admin` for jobs whose root was submitted from inside the
  VPC / loopback / an in-cluster orchestrator.
- `owner_principal` (`controller.proto:135`) is **federation-only**
  (`federated_jobs` table, `schema.py:593`; set `service.py:1323`) — not present
  for ordinary local jobs.

**No arbitrary labels on Iris jobs.** `LaunchJobRequest` (`controller.proto:35`)
and `submit_job` (`client/protocol.py:22`) carry resources/entrypoint/env/
constraints/ports/retries/priority/image — no `labels`/`tags`/`metadata` map.
The only free-form, persisted, queryable per-job key/value is
`EnvironmentConfig.env_vars` (`job.proto:436`) → `job_config.environment_json`
(`schema.py:300`), reachable via `ExecuteRawQuery`. It is secret-redacted
(`redaction.py:40`) and child-inherited (`client.py:733`) — the inheritance is
desirable for issue tagging (set once on the root, propagates to the tree).

**Launch path.** marin `ExecutorStep` → `StepRunner._submit_iris_job`
(`step_runner.py:478`; job name = `_sanitize_job_name(f"{step.name_with_hash}-{uuid4[:8]}")`)
→ `FrayIrisClient.submit` (`lib/fray/src/fray/iris_backend.py:581`) →
`IrisClient.submit` (`client.py:637`) → `JobName.root(resolve_job_user(user), name)`.
The Iris job name is `stepname_<confighash>-<uuid>` — it does **not** encode an
issue and is distinct from the wandb run id. The only free-form context threaded
down is `env_vars`.

**No machine-level run→issue link exists for compute.** datakit has
`IngestionSourceManifest.issue_numbers` (`marin/datakit/ingestion_manifest.py:109`)
but it is dataset-scoped. Conventions exist (`exp<N>_` experiment filenames
`docs/explanations/guidelines.md:109`; a W&B-tag convention in the
`run-research` skill that no code sets/reads; URL scraping from experiment-issue
bodies in `scripts/pm/itemize_experiment_issues.py`) but nothing programmatic
links an Iris job to an issue. Options for a job→issue link: (1) stamp
`MARIN_ISSUE=<N>` into `env_vars` at `step_runner._submit_iris_job` — smallest,
persisted, child-inherited, no proto change; (2) a first-class `labels` map on
`LaunchJobRequest` — cleaner but touches proto + DB + client; (3) reuse a
wandb→issue bridge — no Iris change but brittle and training-only; (4) a
maintained lookup table.

## What surprised us

1. The "iris event we could roll up over" rjpower described **does not exist
   durably**. The data (per-attempt start/stop/state) is real but lives only in
   the controller DB behind a 7-day prune; `iris.task_event` is a same-named red
   herring (k8s scheduling verdicts, 1h retention). Building a durable
   per-attempt accounting row is therefore the load-bearing new piece.
2. **MFU per job is basically already done** — durable in `telltale`, Grafana
   already reads it — but only for Levanter training, with no user column and no
   chip weighting.
3. **TPU has zero hardware duty-cycle telemetry.** For TPUs, "utilization" can
   only mean Levanter MFU.
4. The clean per-user key is `submitting_user` (IAP email), **not** what
   `ListUsers` returns today (`user_id`, the OS username) — a subtlety the rollup
   must get right.
5. No `analyze_job_history.py` and no chip-hour / preemption-overhead script
   exists anywhere; the issue's reference to `scripts/iris/analyze_job_history.py`
   is aspirational. The nearest existing per-user resource math is
   `budget.py::compute_user_spend` — a live scheduling gate, not accounting.

## Peer-review findings (codex + fable + architect)

A three-way review of the first draft surfaced two load-bearing defects,
verified against source:

1. **Placement is destroyed at preemption, in the terminalization transaction.**
   Zone/region/preemptible live on the `workers` row (`worker_attributes`,
   `schema.py:484`, `ondelete=CASCADE`), reached via `task_attempts.worker_id`
   (`ondelete=SET NULL`, `schema.py:428`). `remove_worker` (`writes.py:722`)
   nulls that FK for all the worker's attempts, then deletes the worker
   (`writes.py:734`), and this runs in the same transaction that commits the
   terminal attempt state for worker-failure/preempt (`ops/worker.py`). So a
   post-hoc read sees `worker_id = NULL` for the entire preemption cohort at
   t=0 — a *pull* populator cannot recover placement for exactly the rows the
   design is about. Even for surviving attempts, `worker_retention` = **24h**
   (`controller.py:242`), far short of `job_retention` = 7d. → placement must be
   captured synchronously at terminalization (controller-side emit), before
   `remove_worker`.
2. **"Wasted chip-hours" as full preempted-attempt runtime is dishonest.**
   Levanter checkpoints and resumes, so most of a preempted attempt's runtime is
   retained — true waste ≈ (time since last checkpoint) + restart + recovery
   idle, not computable without a checkpoint-cadence signal we don't emit.
   Separately, user cancels land as `KILLED` rows with `started_at` stamped
   (`reconcile/task.py`, `reconcile/batches.py`), so the `attempt_counts`
   preemption predicate (built for retry-budgeting) counts a cancelled 10h ×
   256-chip run as preemption churn; and the proposed "validate against
   `JobStatus.preemption_count`" uses the *same* predicate → circular. → carry
   `terminal_cause` + `capacity_type` + the enclosing job's terminal state, report
   a labeled upper bound plus a separately-computed recovery-idle number.

Other confirmed corrections folded into the design/spec: `ExecuteRawQuery` guard
is a leading-token check that rejects `WITH` CTEs and returns the full result
set unbounded (`service.py:2868`); `TpuDevice.variant` is `"v5litepod-16"`/`"v4-8"`
not `"v5p"` (normalize via `device_counts_from_json`, `reads.py:36`; chip count =
`TpuDevice.count` per worker summed across the slice); `telltale.job_id` is the
job root (matches the accounting `job_id`, nullable); attempts must be week-clipped
by `[started, finished)` overlap; recovery latency is a cross-attempt self-join,
not `iris.provisioning` (which has no user/job key); nullable timestamps stored
NULL not 0; `TaskState`→`TerminalCause` mapping must be total and fail-fast;
per-attempt preemption counts a 16-host slice preempt as 16.

## Open threads carried into the design

Resolved by the peer review above: population is **controller-side emit at
terminalization** (pull cannot recover placement); "wasted" becomes a labeled
upper bound (`exposed_to_preemption`) plus a self-joined recovery-idle number.
Still open (see `design.md` → Open Questions): the emit's delivery guarantee
(best-effort + reconciliation vs. transactional outbox); the `capacity_type`
source's coverage across backends; the `MARIN_ISSUE` number's provenance; and
whether the Phase-0 attribution-coverage measurement justifies a `MARIN_USER`
stamp before the per-user table is worth shipping.
