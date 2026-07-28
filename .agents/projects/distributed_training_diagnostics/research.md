# Distributed training diagnostics: research

## Background Research Brief

- Effort: medium
- Stop rule: stop when another source no longer changes the placement of live capture, durable storage, or alert evaluation.
- Date: 2026-07-28

### Question

Where should Marin expose on-demand NCCL RAS, thread, and GPU diagnostics, and how should it detect a training process that is alive but no longer completing steps?

### Current Marin context

The NEST-BURN-001 E256 run stopped producing training progress while every rank remained inside the same compiled JAX call. All 32 GPUs reported 100% utilization at roughly 190–230 W against a 1200 W cap. NCCL RAS reported every rank alive and every communicator `RUNNING` with no asynchronous error. Python thread dumps and NCCL RAS were available only through one-off `iris process profile` and `iris task exec` calls. The evidence would have disappeared with the pods if the operator had restarted the gang first.

Telltale already separates passive instrumentation from serving. The registry is scraped every 15 seconds on a daemon thread and forwarded to finelog ([`telltale.py:291-330`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/rigging/src/rigging/telltale.py#L291-L330)). Levanter publishes `levanter_step`, training scalars, and a human-readable status from tracker callbacks ([`tracker/telltale.py:117-153`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/levanter/src/levanter/tracker/telltale.py#L117-L153)). The training Grafana dashboard already plots loss, throughput, and step from the forwarded `telltale` namespace ([`training.json:31-243`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/infra/grafana/dashboards/training.json#L31-L243)).

Iris already has the authenticated control path needed for live captures. `ProfileTask` dispatches through the controller to RPC workers or Kubernetes, and successful captures become `iris.profile` rows. The shared capture code has bounded py-spy and memray watchdogs ([`runtime/profile.py:163-205`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/iris/src/iris/cluster/runtime/profile.py#L163-L205)); `IrisProfile` stores source, attempt, capture time, format, trigger, and payload ([`stats/tables.py:213-261`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/iris/src/iris/cluster/stats/tables.py#L213-L261)).

CoreWeave controllers already write node-level DCGM utilization and aggregate power to `iris.worker` ([`stats/tables.py:97-142`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/iris/src/iris/cluster/stats/tables.py#L97-L142)). The missing field is the power limit needed to compare hardware classes by a ratio instead of a model-specific watt threshold.

### Internal prior work

The `iris_profile_to_finelog` design established the capture path this proposal extends. It moved periodic and on-demand CPU, memory, and thread profiles into `iris.profile`, kept the controller RPC as the authorization and routing boundary, and set seven-day retention. A distributed diagnostic is another capture format, not a second profiling service.

The Grafana bridge already exposes fixed alert projections and leaves arbitrary finelog SQL to dashboards. Provisioned alert rules expect a string-label table with exactly one numeric `value`, return explicit zero rows when healthy, and page on query failure ([`rules.yaml:4-16`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/infra/grafana/provisioning/alerting/rules.yaml#L4-L16)). A training-stall projection belongs there.

### External prior art

NCCL RAS is enabled by default from NCCL 2.24 and listens on localhost port 28028. `VERBOSE STATUS` gathers a global communicator view through a text protocol. NCCL 2.28.7 added JSON output via `SET FORMAT json`; the JSON includes per-rank initialization state, async error state, missing ranks, and per-collective completed-call counters. NCCL 2.29 added a streaming monitor mode. Source: [NVIDIA NCCL RAS documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html).

NVIDIA documents `NCCL_DEBUG=INFO` as diagnostic logging and `TRACE` as replayable per-call logging. `NCCL_DEBUG_SUBSYS` can select initialization, network, graph, tuning, RAS, collective, proxy, NVLink SHARP, and other subsystems. `NCCL_DEBUG_FILE=/dev/stderr` is line-buffered; a `%h.%p` file must be unique and otherwise overwrites earlier output. Source: [NVIDIA NCCL environment-variable documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

### Decision

Do not add a generic web-hook registry to Rigging Telltale.

Telltale’s routes are intentionally `@public` and are mounted by actor and task services ([`telltale.py:401-455`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/rigging/src/rigging/telltale.py#L401-L455)). The training runtime binds them to `0.0.0.0` and registers the endpoint with Iris ([`runtime/telltale.py:79-130`](https://github.com/marin-community/marin/blob/1509b637aab6b7109cb0cfaa5a04ff211e4c18a8/lib/iris/src/iris/runtime/telltale.py#L79-L130)). A generic callback would need its own authorization semantics, timeout enforcement, response-size cap, and process-lifetime policy. It would also execute arbitrary application code on an HTTP request thread inside a training process.

Extend `ProfileTask` with an explicit distributed diagnostic type. It already supplies authenticated access, worker/Kubernetes dispatch, timeouts, finelog persistence, and a CLI. The task process needs no new inbound route. The NCCL query is a bounded localhost socket read from the container namespace.

Keep alerts passive. Telltale and DCGM feed finelog; the Grafana bridge evaluates a fixed stall projection; Grafana sends a warning to Slack and Loom. The operator or Loom session captures a diagnostic bundle before kicking the gang. Grafana does not receive an Iris mutation credential.

### Alert hypothesis

A job is stalled when all of these hold:

- `levanter_step` has not advanced for 15 minutes after at least one completed step, or it remains zero for 45 minutes after the first fresh Telltale sample.
- Telltale samples remain fresh within 90 seconds. This distinguishes a blocked training call from a dead exporter or lost finelog path.
- At least 75% of the job’s GPU nodes report mean utilization of 90% or more over the last five minutes.
- The job is still `RUNNING` in fresh `iris.task_state`.

The alert labels the stall `collective_like` when median `gpu_power_w / gpu_power_limit_w` is below 0.35. Low power is classification evidence, not an alert gate: input stalls and long compilations can consume little power without running a collective.

The 15-minute threshold is intentionally conservative. Levanter should publish a numeric phase (`initializing`, `training`, `evaluation`, `checkpointing`, `finished`) so the projection can use phase-specific thresholds and suppress expected long evaluation or checkpoint windows. Until that phase exists, the rule should remain a warning and require five consecutive one-minute evaluations.

### Runtime logging policy

Normal multi-host jobs:

```text
NCCL_RAS_ENABLE=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS
NCCL_DEBUG_FILE=/dev/stderr
```

`/dev/stderr` lets Iris log shipping preserve initialization and transport evidence before pod loss. Do not write normal-run diagnostics only to `/tmp`: files disappear on pod deletion, and `%h.%p` files require a separate collector.

Short reproduction jobs may add `COLL,PROXY,NVLS,REG`. `NCCL_DEBUG=TRACE` and `CALL` remain opt-in because they emit per-call data. All matched experiment arms must pin the same CUDA and NCCL builds; the current E256 and fixed25 retries loaded NCCL 2.28.9 against different CUDA minor builds, which weakens runtime comparisons even though communicators never span arms.

### Negative / failed leads

- A Telltale hook registry saves one RPC extension but creates a public, in-process execution surface and has no durable payload contract.
- Polling NCCL RAS from every process every 15 seconds is redundant: one response already gathers global communicator state, and a troubled communicator can take several seconds to report.
- GPU utilization alone cannot identify a collective. CUDA busy-poll kernels and real matmuls both report high utilization.
- Power alone cannot be compared across B200, GB200, H100, and TPU jobs. The metric needs a device power limit or a model-specific normalization.
- A Grafana alert that directly kicks a job destroys the evidence it is supposed to preserve and gives the monitoring bridge a mutation credential.

### Source ledger

| Source | Type | Claim used for | Confidence |
|---|---|---|---|
| `lib/rigging/src/rigging/telltale.py` | Marin code | Telltale route posture and forward loop | high |
| `lib/iris/src/iris/cluster/runtime/profile.py` | Marin code | bounded capture and durable profile path | high |
| `lib/levanter/src/levanter/tracker/telltale.py` | Marin code | current step and metric signals | high |
| `lib/iris/src/iris/cluster/stats/tables.py` | Marin code | existing GPU utilization/power and profile schemas | high |
| `infra/grafana/*` | Marin code | fixed alert projections and warning policy | high |
| NVIDIA NCCL RAS docs | official docs | RAS protocol, JSON counters, versions | high |
| NVIDIA NCCL env docs | official docs | debug levels, subsystems, file behavior | high |

### Handoff

- Implemented foundation: `lib/iris/src/iris/cluster/runtime/nccl_ras.py` queries JSON with a text fallback and reports per-communicator collective-count skew.
- Next implementation slice: add `DistributedProfile` to `ProfileTask`, bundle RAS/thread/GPU/process evidence, and write one `iris.profile` JSON row per task.
- Alert slice: add Levanter phase/progress metrics, DCGM power-limit collection, a fixed `/finelog/marin/alerts/training_stalls` projection, a warning rule, and a training-dashboard table.
