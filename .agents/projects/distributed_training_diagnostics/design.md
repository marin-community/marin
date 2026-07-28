# Distributed training diagnostics

_Capture a stalled distributed job before recovery removes the evidence._

NCCL RAS, thread stacks, GPU telemetry, and runtime versions should be available through one authenticated Iris command and persist in finelog. Grafana should warn when a running training job stops completing steps while its GPUs remain active. The alert opens an operator workflow; it does not restart the job.

[Research](research.md) records the current NEST-BURN-001 incident, repository paths, and the NCCL protocol details.

## Challenges

A collective hang looks healthy to most liveness checks. The Python process, Telltale server, CUDA kernels, Kubernetes pod, and NCCL RAS control threads can all remain responsive while the training step never returns. GPU utilization can stay at 100%; low power makes a collective wait plausible but does not prove it.

The useful evidence spans process and control-plane boundaries. NCCL RAS is a localhost service inside the task namespace. Thread captures require ptrace safety. GPU power comes from node-level DCGM. Training progress comes from a process-local tracker forwarded to finelog. The capture must survive the pod, while passive monitoring must not add a query fan-out from every rank.

## Costs / Risks

- A distributed capture adds one profile type across the Iris proto, RPC-worker, Kubernetes, CLI, dashboard, and finelog paths.
- Thread dumping briefly attaches to the target. The existing Iris watchdog and `SIGCONT` recovery remain mandatory.
- Stall thresholds can flag long compilation, evaluation, or checkpoint work. The first rule is warning-only and uses explicit phase metrics plus conservative grace periods.
- `NCCL_DEBUG=INFO` adds log volume. Normal jobs exclude `COLL`, `CALL`, and `TRACE`.

## Design

### Live and durable capture

Extend `ProfileTask` with `DistributedProfile`. `iris process profile distributed -t <task-or-job>` dispatches through the same authenticated controller path as CPU, memory, and thread profiles. A job target fans out with bounded concurrency; a task target captures one pod.

Each capture writes an `iris.profile` row with `type="distributed"`, `format="json"`, and the existing seven-day retention. The JSON envelope contains:

- NCCL RAS `VERBOSE STATUS`, requested as JSON and preserved as text on NCCL before 2.28.7;
- per-communicator collective-count differences, with the lagging ranks named;
- a py-spy thread dump using the existing attach watchdog;
- `nvidia-smi` identity, utilization, memory, power, power limit, and active process rows;
- `/proc/1/task/*/wchan`, selected `NCCL_*` variables, `XLA_FLAGS`, CUDA/JAX/NCCL versions, hostname, and capture errors.

Every collector has a five-second timeout except the existing 30-second thread-dump watchdog. Individual collector failures become structured errors in the envelope; they do not discard successful evidence. The full uncompressed payload is capped at 4 MiB.

The NCCL RAS client lives at `lib/iris/src/iris/cluster/runtime/nccl_ras.py`. It uses the documented localhost text protocol, requests `SET FORMAT json`, strips command acknowledgements, and falls back to text. It does not poll in the background.

Rigging Telltale gets no generic diagnostic-hook registry. Its routes are public by contract and mounted by services other than training. Iris already owns authorization, dispatch, timeouts, and persistence for on-demand captures.

### Passive progress and power signals

Levanter Telltale adds two gauges:

- `levanter_progress_time_seconds`: wall-clock time of the last completed training step;
- `levanter_phase`: `0=initializing`, `1=training`, `2=evaluation`, `3=checkpointing`, `4=finished`.

The tracker updates both outside the compiled step. Telltale continues forwarding every 15 seconds, so finelog freshness proves the auxiliary process thread is alive while `progress_time` distinguishes repeated scrapes from new training progress.

CoreWeave DCGM collection adds aggregate `gpu_power_limit_w` to `iris.worker`. The existing `worker_id` on Telltale rows associates each process with its node row. The Grafana bridge runs one fixed, bounded finelog query over fresh `telltale`, `iris.worker`, and `iris.task_state` rows and exposes `/finelog/marin/alerts/training_stalls`.

A warning candidate requires:

- no progress for 15 minutes in `training`, or 45 minutes in `initializing`;
- a fresh Telltale sample within 90 seconds;
- at least 75% of the run’s GPU nodes at mean utilization of 90% or more over five minutes;
- a fresh `RUNNING` job-state row.

Evaluation and checkpoint phases use configurable 60-minute thresholds initially. The bridge emits `classification="collective_like"` when median power divided by power limit is below 0.35. Grafana evaluates every minute and fires after five minutes. The alert links the job, training dashboard, `iris process profile distributed` command, and the collective-hang runbook. Severity remains `warning`; the Loom operator captures evidence and decides whether to kick the gang.

### Runtime flags and post-mortem logs

Multi-host launch helpers set:

```text
NCCL_RAS_ENABLE=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS
NCCL_DEBUG_FILE=/dev/stderr
```

Iris log shipping makes stderr durable. Short reproduction jobs may add `COLL,PROXY,NVLS,REG`. `TRACE` and `CALL` stay opt-in. Launch metadata records CUDA, JAX, and NCCL versions so matched arms cannot silently resolve different communication stacks.

Grafana remains read-only. It does not call `ProfileTask`, kick tasks, or hold an Iris mutation credential. Automatic recovery is out of scope.

## Testing

The NCCL client tests its socket wire request, acknowledgement-prefixed JSON, text fallback, and collective-count skew analysis against a local fake service.

The ProfileTask slice uses existing fake backends to assert that a successful distributed capture returns the JSON bytes and persists the same bytes to `iris.profile`; a partial collector failure still persists the other sections. A Kubernetes runtime test executes the bundle in a non-GPU pod and records explicit `unavailable` results without failing the task.

Levanter tracker tests use a fake clock to verify phase transitions and progress time. DCGM parser tests cover summed power limits. Grafana bridge tests use fixed Arrow tables for a progressing run, a high-utilization stall, stale Telltale, initialization grace, evaluation suppression, and missing power limits. Alert provisioning tests validate the warning rule and its no-data posture.

Rollout starts on one batch-priority multi-host smoke. Operators compare capture latency, payload size, and finelog query cost. The alert runs dashboard-only for one week before Slack/Loom notification is enabled.

## Open Questions

- Should a job-wide capture store one global RAS report plus per-task process sections, or accept duplicate RAS payloads in each task row for simpler provenance?
- Are 15 minutes for training and 45 minutes for initialization conservative enough for the largest JAX compilations and Paloma evaluation windows?
- Should a future Iris recovery command require an explicit `--skip-diagnostics` when a responsive GPU gang is kicked, or keep capture as a runbook convention?
