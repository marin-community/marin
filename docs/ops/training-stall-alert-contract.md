# Training stall warning

`TrainingProgressStalled` is a passive, warning-only Grafana rule. It does not kick, restart, or profile a job. Capture `iris process profile threads -t <task>` for the affected tasks before deciding whether to intervene.

The rule evaluates once a minute and waits five minutes before notifying. A root job is eligible only while its latest `iris.task_state` row is at most 90 seconds old, reports at least one running task, and has a retained `phase` row from `service=levanter`. This enrollment gate prevents long-running Zephyr, inference, and generic Iris jobs from being classified as training without expiring a still-running Levanter job after 24 hours. An eligible job warns when either condition holds:

- Training has started and `progress_time_seconds` is at least 15 minutes old. An explicit training phase or a positive `step` identifies training.
- The job has remained running for at least 45 minutes without entering training. Missing progress after Levanter enrollment counts as absent progress rather than suppressing the warning.

The first case is labeled `optimizer_progress_stale` or `optimizer_progress_missing`; the second is `initializing_stale`. Finished and progressing jobs emit zero-valued rows so Grafana can resolve their warning state. An idle fleet also emits an explicit zero.

A job that emits `step` but no `phase` runs a producer older than both phase and progress, and reports a zero-valued `producer_missing` row instead of a warning. Enrollment keys on phase because `TelemetryTracker` publishes it as it is constructed, which marks the producer generation exactly.

The bridge joins the durable streams by `(cluster, root job ID)`: `iris.task_state.root_job_id` equals `json_get(telemetry_v1.resource_attributes_json, 'job_id')`, and Finelog stamps both with their origin `cluster`. The task-state query scans the trailing hour. Progress rows scan the trailing 24 hours, while the sparse phase enrollment branch selects the latest retained phase without a lower time bound. An older running job still has an inferred running age of at least one hour, which is sufficient for both thresholds. No task-to-node mapping or GPU-utilization condition is required.

The required `service=levanter` telemetry records are `step`, `progress_time_seconds`, and numeric `phase` (`initializing=0`, `training=1`, `finished=2`). `TelemetryTracker` initializes phase and progress, records wall time after its completed-step `train/loss` callback, and marks a finished run.

## NCCL RAS snapshots

GPU Levanter runs poll NCCL RAS from JAX process 0 every ten minutes. NCCL returns a job-global communicator view, so one collector avoids repeating the same query from every rank. Steady-state polling uses `STATUS`; watchdog-triggered collection uses `VERBOSE STATUS`. Both run through NCCL's documented text protocol in a bounded Python subprocess with an eight-second outer deadline. The collector does not depend on the separately packaged `ncclras` executable. `NCCL_RAS_ENABLE=0` disables collection.

The client accepts up to 32 MiB of raw JSON in its own process, validates it with bounded Pydantic models, and writes a reduced result of at most 192 KiB to stdout. The shared command runner retains its 256 KiB ceiling. A malformed communicator increments the reduction's invalid count without discarding valid peers. The report also records input and omitted counts for communicators, progress summaries, and rank observations; publication records its own admitted and dropped counts.

Healthy communicators produce communicator and rank-count summaries, plus minimum and maximum operation counts per collective. `communicator_rank_status` is sparse: it contains missing ranks, ranks with NCCL errors, and, for a watchdog-triggered sample, unique collective-progress outliers. It does not contain one row for every healthy rank. Lifecycle transitions such as finalize and destroy are reported as state, not classified as errors. A tied count split is a mismatch but does not assign an outlier rank.

`moe_hero_fsdp` configures a 20-second watchdog diagnostic budget. When the watchdog fires, only process 0 stops periodic collection, requests a fresh verbose sample, publishes the reduced evidence, flushes telemetry, and exits with code 124. The diagnostic thread cannot delay exit past that budget. Other ranks do not issue duplicate RAS queries; Iris propagates the failed coscheduled task to the gang.

Iris's Kubernetes sidecar ships task logs and does not poll RAS. The Iris node-agent NVIDIA probe records host hardware separately. Leave both enabled; neither duplicates these communicator snapshots.

Every row carries the collector's Iris `task_id`, attempt/execution identity, node, `process_index=0`, and `root_run_uid`. Rank rows add NCCL's `rank_host`, `process_id`, `cuda_device`, and `nvml_device`; those fields identify the process represented by a global RAS rank instead of treating the collector task as the rank owner.

`ras_available=1` records a parsed response. `ras_available=0` includes an `outcome` such as `unavailable`, `client_timeout`, `deadline_exceeded`, `runner_output_limit`, `client_response_limit`, `reduced_output_limit`, `invalid_payload`, or `invalid_client_output`. The `trigger` attribute is `periodic` or `stall`. `ras_poll_failures` and `ras_poll_timeouts` are counter deltas; sum them over the query window. `ras_poll_duration_seconds` is the client-side latency, while `ras_collection_duration_seconds` and `ras_collection_timeouts` come from NCCL's successful response.

Query a root run with a bounded timestamp range:

```sql
SELECT
  cluster,
  to_timestamp_millis(timestamp_ms) AS ts,
  json_get(resource_attributes_json, 'root_run_uid') AS root_run_uid,
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
  AND json_get(resource_attributes_json, 'root_run_uid') = '<run-id>'
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

Repeated, unchanged minimum and maximum `collective_operations` values during stale optimizer progress are evidence of a stationary communicator. A one-sample count mismatch is not a hang verdict because ranks may be briefly offset while collectives are active. Check `ras_reduction_records` and `ras_metric_records` before interpreting absent rank detail. No retained RAS rows means collection or delivery is unverified; `ras_available=0` proves the collector ran and could not obtain a valid response.

The NVIDIA probe follows the same sparse convention. Every poll emits `nvidia_health_available` and `gpu_devices` counts grouped by `total`, `healthy`, `abnormal`, or `unknown`; per-device ECC, retired-page, and row-remap rows are emitted only when nonzero. Inventory, total memory, and power-limit rows are refreshed hourly rather than on every health poll.

GPU utilization, power, and power-limit metrics remain diagnostic evidence available in finelog and the capture bundle; they do not gate or classify this warning. This avoids hiding a stalled job when node attribution is unavailable.

The rule currently covers CoreWeave controllers whose active root-job state is forwarded into the `marin` finelog hub. GCE controllers do not emit `iris.task_state`, so their jobs are not evaluated by this rule.
