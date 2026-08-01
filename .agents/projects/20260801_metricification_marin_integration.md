# Marin metricification integration plan

Status: active

## Scope

This branch owns the single Marin producer/dashboard review unit after the
simple telemetry foundation. It starts at `729cc2d6a`, keeps the benchmark
decision record, and does not modify the vLLM repository. The corrected vLLM
dashboard/query feeder is integrated as three preserved commits. The Rigging
hardware probe feeder stays deferred until a corrected tip is provided.

The authoritative v1 transport remains the bounded in-process
`rigging.telemetry` exporter posting directly to Finelog `telemetry_v1`. This
work does not add an agent, WAL, OTLP receiver, durable outbox, generic rollup
engine, or physical repartitioning.

## Frozen identity and signal contract

- `root_run_uid` identifies the stable top-level effort.
- `execution_uid` identifies one launch, resume, or retry.
- `service` identifies the process family and `role` is a bounded enum.
- `job_id`, `task_id`, `attempt`, `worker`, and `process_index` identify Iris
  execution.
- `actor_uid`, `engine_uid`, `rank`, and `worker_index` identify distributed
  runtime entities.
- `node_uid`, `gpu_uuid`, and `pci_bus_id` identify hardware.
- `model_revision`, `checkpoint_uid`, `policy_step`, and `weight_version`
  identify code/model state.
- `result_uid`, `result_uri`, `wandb_run`, and `evaldash_run` are terminal
  evidence joins.
- No new telemetry field is named `run_id`.

Common numeric families are `work_completed`, `phase_duration_seconds`,
`queue_depth`, and `progress_time_seconds`. Phase observations always carry
`clock_domain=critical_path|worker_time`. Lifecycle and entity relationships
are structured events.

## Milestones

### M1: canonical Iris resource and telemetry health

Add a small typed convention module below Rigging and keep the foundation
transport unchanged. Iris owns authoritative execution identity and merges it
with explicit workload context:

```python
runtime_telemetry.configure(
    service="marin-rl",
    role=TelemetryRole.TRAINER,
    root_run_uid=config.run_id,
)
```

When `execution_uid` is omitted under Iris, derive it from the exact Iris job
and attempt. Do not synthesize node/device identity. Emit bounded root→job,
job→task, task→attempt, and attempt→worker/process link events. vLLM resources
carry `serving_job_id` so endpoint clients can join without duplicating engine
metrics.

Extend the process-local status snapshot with export attempts/failures/retries,
rejections, last success, and oldest queued record age. Publish the snapshot at
existing application progress cadences; do not start another health thread.
Emit a bounded shutdown-start event before the foundation's one bounded flush.

### M2: Levanter pre-training

Replace the telemetry-only `run` attribute with `root_run_uid` and trainer role.
Map existing step/throughput/MFU values to the common families while preserving
the existing operational scalar mirrors needed by current dashboards.

Add phase observations around:

```python
PHASE_DURATION.record(
    info.step_duration,
    attributes={"phase": "training", "clock_domain": "critical_path", "outcome": "succeeded"},
)
```

Emit data wait from loader timing, compilation/startup from the first step,
evaluation from eval callbacks, and checkpoint duration/outcome from the
checkpointer boundary. Add a moderate-cadence rank callback using a real
all-gather. Process zero emits min/median/max and bounded top-k/persistent
outlier events; individual rank samples remain cadence-bounded.

### M3: Marin RL

Configure coordinator, trainer, rollout, inference, environment, storage, and
weight roles from one root run. Record lifecycle and terminal events at entry
boundaries.

Trainer phases are exclusive `critical_path` observations: rollout wait, batch
assembly/sharding, forward/backward/optimizer, checkpoint/evaluation, and weight
publication. Rollout/inference/environment/storage/weight work uses
`worker_time` and may overlap.

Reuse current measured values instead of adding parallel timers:

```python
record_phase(timing.fetch_time, phase="rollout_wait", clock_domain="critical_path")
work_completed(total_output_tokens, work_kind="generated_token")
```

Publish train/policy/weight gauges, progress freshness, rollout/request/token
deltas, queue/capacity, rollout age, weight lag, transfer bytes, retries, and
normalized failures. Offline embedded vLLM keeps only the Marin-side bridge;
central served vLLM remains the canonical engine producer.

### M4: Zephyr and data generation

Configure the coordinator with `root_run_uid=execution_id` and configure worker
actors with the same root plus worker identity. Record stage/shard lifecycle,
stage duration, shard completion/retry, queue/inflight/worker-idle snapshots,
and progress freshness.

Translate completed shard snapshots into delta `work_completed` rows for
records and bytes. Keep current aggregate snapshots as explicitly tagged
current/cumulative source data. Emit bounded shard duration observations and a
top-k outlier event at stage completion instead of dashboard series for every
historical shard. Add spill/retry/error evidence where existing execution
state exposes it; do not infer missing I/O counters.

### M5: query and dashboard integration points

Add checked-in parameterized query templates for weekly pre-training, RL, and
Zephyr accounting; phase fractions; rank/shard outliers; failure timelines;
telemetry completeness; and reset-aware imported counters. Every template must
retain literal/raw `timestamp_ms` lower and upper predicates and at least one
entity predicate.

Add fleet/run/role/process/device dashboard seams and observe-only alert query
definitions for the signals available from M1–M4. Hardware/Ray/DCGM/NCCL and
Rigging inputs remain wired to stable input names. Integrate the approved vLLM
feeder commits in order and preserve their bounded per-replica freshness,
45-second reset/missing-predecessor lookback, coherent histogram-family
invalidation, and raw-sample KV peak semantics.

## Validation

- Behavior tests query exported records, phase exclusivity, identity joins,
  reset-aware deltas, and bounded outlier output.
- Unconfigured and failed telemetry must not change application outputs or exit
  status.
- No test sleeps; fake clocks/transports are used at I/O boundaries.
- Measure emission call latency and producer-side record counts/cardinality on
  bounded fixtures.
- Run focused Iris, Rigging, Levanter, Marin RL, Zephyr, and Grafana tests;
  `./infra/pre-commit.py --changed-files --fix`; then commit before one
  `./infra/pre-commit.py --review --agent-command='codex exec'` pass.
- Push coherent milestone commits, open one draft PR with `agent-generated`,
  monitor CI and existing review feedback, and publish the final Weaver
  evidence artifact.

## Deferred inputs

- Rigging hardware probe feeder: do not cherry-pick until the coordinator sends
  a corrected commit.
- vLLM dashboard/query feeder: integrate `eec3de4b72ab`, `06d2145febc7`, then
  `45d9b2bd3b67` in this PR without creating a feeder PR.
- No vLLM fork or repository changes.
