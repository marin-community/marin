# Hero training stall alert

`TrainingProgressStalled` is a critical Grafana alert for active Iris hero roots. Permitted names are `<run-id>-coord` and `<run-id>-coord-<retry>`. It posts one Slack message per run and opens a Loom triage session on that thread. The alert does not kick, restart, or profile a job. Capture `iris process profile threads -t <task>` for the applicable tasks before an intervention.

This alert watches whether a run is stepping. [`TrainingLossSpike`](training-loss-spike-alert.md) watches what those steps produce, over the same enrollment.

The rule evaluates once a minute and waits five minutes before notification. An eligible root has a current `iris.task_state` row. The row is at most 90 seconds old and reports one or more running tasks. The root name matches `%/hero-%-coord` or `%/hero-%-coord-%`. The namespace before the run name is unrestricted.

Hero alert enrollment is a launch naming contract. The last root component is `<run-id>-coord` or `<run-id>-coord-<retry>`. The `<run-id>` value begins with `hero-`. The Levanter trainer `id` is the same `<run-id>`. A retry suffix changes the Iris job identity only. It does not change the logical run identity.

An eligible job alerts when either condition holds:

- Training has started and `progress_time_seconds` is at least 15 minutes old. An explicit training phase or a positive `step` identifies training.
- The root job has remained running for at least 45 minutes without entering training. Missing telemetry has the same initialization timeout.

The first case is labeled `training_stalled`; the second is `initializing_stale`. Missing and stale optimizer progress share one label so a delayed sample cannot replace one firing alert instance with another. Finished and progressing jobs emit zero-valued rows, which removes the firing series and resolves its original fingerprint. An idle fleet also emits an explicit zero.

The bridge derives `hero-20260819` from the root job, then queries the structured `telemetry_v1.run_id` column with exact equality. With concurrent hero runs it emits one `run_id IN (...)` predicate rather than a broad pattern scan. The newest `phase` row in the trailing hour selects one `execution_uid`; `step` and `progress_time_seconds` from older task attempts cannot keep the run healthy. Iris derives `execution_uid` from the controller-minted `attempt_uid`, which stays unique if a new controller uses the same numeric task attempt. Progress spans 30 minutes so a sample remains observable after crossing the 15-minute stall threshold.

Initialization and missing-progress grace start at the later of the current contiguous Iris running interval and the selected telemetry execution. A coordinator restart or trainer retry therefore receives a new initialization window. A hard hang retains its training execution anchor for one hour. All states remain instances of `TrainingProgressStalled`, and notification grouping excludes phase and reason, so later reclassification cannot open a second Slack group.

`iris.task_state.root_job_id` names the coordinator root, while `telemetry_v1.job_id` names the descendant trainer. For example, the root `/rav/hero-20260819-coord` owns telemetry from `/rav/hero-20260819-coord/grug-train-hero-20260819`. The bridge joins on origin cluster, exact `run_id`, and this descendant relationship. Alert labels use the coordinator root. No task-to-node mapping or GPU-utilization condition is required.

The required `service=levanter` telemetry records are `step`, `progress_time_seconds`, and numeric `phase` (`initializing=0`, `training=1`, `finished=2`). `TelemetryTracker` initializes phase and progress, records wall time after its completed-step `train/loss` callback, and marks a finished run. The hero launchers already pass their trainer ID into telemetry as `run_id`; W&B retains its separate `hero` tag.

Verify enrollment after launch with a bounded Finelog query:

```sql
SELECT
  "cluster",
  run_id,
  job_id,
  execution_uid,
  name,
  value,
  to_timestamp_millis(timestamp_ms) AS observed_at
FROM "telemetry_v1"
WHERE service = 'levanter'
  AND run_id = '<hero-run-id>'
  AND name IN ('phase', 'step', 'progress_time_seconds')
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '30 minutes') * 1000 AS BIGINT)
ORDER BY timestamp_ms DESC, seq DESC
LIMIT 20;
```

For an unsuffixed root, remove `-coord` to get `<hero-run-id>`. For a suffixed root, also remove the retry suffix. An empty result identifies missing Levanter telemetry. The root becomes `initializing_stale` after 45 minutes.

Levanter republishes phase every minute. A current phase row binds progress to one execution. Finelog and telemetry health alerts cover loss of the durable telemetry path. Launcher-specific process watchdogs provide additional coverage where configured.

Grafana routes `notification=hero-run` through `ops-critical`. It groups notifications by alert name and logical run. Thus, a new retry root stays in the same alert group. The bridge keys the Slack thread on this group and retains it for six hours after a resolution. A replacement retry that finishes its pending period joins the existing thread. A four-hour repeat notification also keeps the thread active. An idle fleet returns an explicit zero-valued `fleet/idle/healthy` row. The `noDataState: Alerting` state identifies an invalid or unavailable response.

## NCCL RAS snapshots

GPU Levanter runs poll NCCL RAS from JAX process 0 every ten minutes. NCCL returns a job-global communicator view, so one collector avoids repeating the same query from every rank. Steady-state polling uses `STATUS`; watchdog-triggered collection uses `VERBOSE STATUS`. Both run through NCCL's documented text protocol in a bounded Python subprocess with an eight-second outer deadline. The collector does not depend on the separately packaged `ncclras` executable. `NCCL_RAS_ENABLE=0` disables collection.

The client accepts up to 32 MiB of raw JSON in its own process, validates it with Pydantic models, and writes the complete reduced result to stdout. NCCL opts into a 32 MiB subprocess-output limit while other hardware probes retain the shared 256 KiB default. A malformed communicator increments the reduction's invalid count without discarding valid peers. The report records input and omitted communicator counts, while publication records enqueued and telemetry-lost rows.

Healthy periodic samples produce communicator and rank-count summaries but no per-collective or per-rank rows. A periodic mismatch adds minimum and maximum operation counts for the affected collective. A watchdog-triggered sample includes all collective progress plus every missing rank, rank with an NCCL error, and unique progress outlier. Lifecycle transitions such as finalize and destroy are reported as state, not classified as errors. A tied count split is a mismatch but does not assign an outlier rank.

`moe_hero_fsdp` configures a 20-second watchdog diagnostic budget. When the watchdog fires, only process 0 stops periodic collection, requests a fresh verbose sample, publishes the reduced evidence, flushes telemetry, and exits with code 124. The diagnostic thread cannot delay exit past that budget. Other ranks do not issue duplicate RAS queries; Iris propagates the failed coscheduled task to the gang.

Iris's Kubernetes sidecar ships task logs and does not poll RAS. The Iris node-agent NVIDIA probe records host hardware separately. Leave both enabled; neither duplicates these communicator snapshots.

Every row carries the collector's Iris `task_id`, attempt/execution identity, node, `process_index=0`, and `run_id`. Finelog promotes `run_id` into its structured column while retaining the resource attribute. Rank rows add NCCL's `rank_host`, `process_id`, `cuda_device`, and `nvml_device`; those fields identify the process represented by a global RAS rank instead of treating the collector task as the rank owner.

`ras_available=1` records a parsed response. `ras_available=0` includes an `outcome` such as `unavailable`, `client_timeout`, `deadline_exceeded`, `runner_output_limit`, `client_response_limit`, `invalid_payload`, or `invalid_client_output`. The `trigger` attribute is `periodic` or `stall`. `ras_poll_failures` and `ras_poll_timeouts` are counter deltas; sum them over the query window. `ras_poll_duration_seconds` is the client-side latency, while `ras_collection_duration_seconds` and `ras_collection_timeouts` come from NCCL's successful response.

Query a root run with a bounded timestamp range:

```sql
SELECT
  cluster,
  to_timestamp_millis(timestamp_ms) AS ts,
  run_id,
  json_get(resource_attributes_json, 'task_id') AS collector_task_id,
  name,
  kind,
  value,
  json_get(attributes_json, 'outcome') AS outcome,
  json_get(attributes_json, 'trigger') AS trigger,
  json_get(attributes_json, 'communicator_hash') AS communicator_hash,
  json_get(attributes_json, 'rank') AS rank,
  json_get(attributes_json, 'rank_host') AS rank_host,
  json_get(attributes_json, 'collective') AS collective,
  json_get(attributes_json, 'rank_statistic') AS rank_statistic,
  json_get(attributes_json, 'record_state') AS record_state
FROM "telemetry_v1"
WHERE service = 'levanter'
  AND run_id = '<run-id>'
  AND name IN (
    'ras_available',
    'ras_poll_duration_seconds',
    'ras_poll_failures',
    'ras_poll_timeouts',
    'communicators',
    'communicator_state',
    'communicator_ranks',
    'communicator_rank_status',
    'collective_operations',
    'ras_reduction_records',
    'ras_metric_records',
    'ras_collection_duration_seconds',
    'ras_collection_timeouts'
  )
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '30 minutes') * 1000 AS BIGINT)
ORDER BY timestamp_ms, name;
```

`collective_operations` appears for periodic mismatches and detailed stall captures. A one-sample count mismatch is not a hang verdict because ranks may be briefly offset while collectives are active. Check `ras_reduction_records` and `ras_metric_records` before interpreting absent rank detail. No retained RAS rows means collection or delivery is unverified; `ras_available=0` proves the collector ran and could not obtain a valid response.

The NVIDIA probe follows the same sparse convention. Every poll emits `nvidia_health_available` and `gpu_devices` counts grouped by `total`, `healthy`, `abnormal`, or `unknown`; per-device ECC, retired-page, and row-remap rows are emitted only when nonzero. Inventory, total memory, and power-limit rows are refreshed hourly rather than on every health poll.

GPU utilization, power, and power-limit metrics remain diagnostic evidence available in finelog and the capture bundle; they do not gate or classify this alert. This avoids hiding a stalled job when node attribution is unavailable.

The rule currently covers CoreWeave controllers whose active root-job state is forwarded into the `marin` finelog hub. GCE controllers do not emit `iris.task_state`, so their jobs are not evaluated by this rule.
