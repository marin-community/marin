# Sub-doc: Reconciliation Patterns Across Orchestrators

Companion to `spec.md` §3. Documents the prior-art findings that informed Iris' design choices.

## Decision matrix

| System | Direction | Identity model | State model | Liveness | Notification latency | Where it informs Iris |
|---|---|---|---|---|---|---|
| **K8s controller-runtime** | watch+push from etcd to controller; controller→kubelet via pod spec writes that kubelet pulls | `name + uid + generation` triple; controllerRef by UID | level-triggered (informer cache); resync every 10–30 min | Lease object (`coordination.k8s.io`), 15s/10s/2s | RTT (watch event) | UID/generation/observedGeneration model; level-triggered protocol |
| **Nomad** | worker-pull via long-blocking-query | `AllocID` (UUID) + `ModifyIndex` | level-triggered digest, edge-triggered fetch | TTL heartbeat (`Node.UpdateStatus`) | ~immediate via blocking query wake | digest-then-fetch (etags), the §4.2.1 pattern |
| **Mesos / Marathon** | server-push; framework calls `reconcile_tasks` for resync | `FrameworkID`/`AgentID`/`TaskID` (UUIDs) | edge-triggered + explicit reconcile | master pings agents; `--agent_reregister_timeout` | RTT | explicit reconciliation API as operator escape hatch |
| **Ray (raylet ↔ GCS)** | mixed: raylet pushes heartbeats; GCS pushes scheduling | `NodeID`/`WorkerID`/`ActorID` (28-byte) | level-triggered resources; edge-triggered actor state | 100ms heartbeats, 30s timeout | RTT | snapshot-and-reconcile loop with typed events |
| **Slurm** | server-push poll + slurmd self-register | `NodeName` (hostname); `JobID.StepID` int | level-triggered registration; lightweight pings | `SlurmdTimeout=300s`, pinged every ~100s | poll interval | one canonical "node registration" RPC for boot/restart/safety |
| **AWS ECS** | server-push over worker-initiated websocket | `TaskArn` (per-instance UUID); `ContainerInstanceArn` | hybrid: edge payload + level manifest | WebSocket keepalive | RTT | manifest+verification handshake (see Iris §4.6) |
| **Temporal** | worker-pull, long-poll | `WorkflowID + RunID`; activity `taskToken` per-attempt | edge-triggered events over event-sourced log | per-activity heartbeat; no worker reg | poll wait (~60s max) | per-attempt opaque token (UID equivalent) |
| **Borg / Omega** | server-push poll; Borglet reports full machine state | task instance + generation; cell-scoped IDs | full state from worker; delta from link shard to leader | poll cadence + missed-poll threshold | poll interval | **fail-open Borglet** — keep tasks alive through controller blips |

## Notes per system

### Kubernetes / controller-runtime

The canonical references:
- **Workqueue** ([client-go/util/workqueue/queue.go](https://github.com/kubernetes/client-go/blob/master/util/workqueue/queue.go)): three sets (`queue`, `dirty`, `processing`). `Add` is a no-op if key in `dirty`. Burst coalesces structurally.
- **Default rate limiter** ([default_rate_limiters.go](https://github.com/kubernetes/client-go/blob/master/util/workqueue/default_rate_limiters.go)): per-item exp 5ms→1000s, global 10qps/100 burst.
- **Reconciler return** ([controller-runtime controller.go](https://github.com/kubernetes-sigs/controller-runtime/blob/main/pkg/internal/controller/controller.go)): `err` → `AddRateLimited` (failure count grows). `RequeueAfter` → `Forget + AddAfter`. Success → `Forget`.
- **Leader election** ([leaderelection.go](https://github.com/kubernetes/client-go/blob/master/tools/leaderelection/leaderelection.go)): defaults 15s lease, 10s renew deadline, 2s retry, 1.2 jitter. Failed renew → `OnStoppedLeading`.
- **Bookmarks** ([KEP-956](https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/956-watch-bookmark/README.md)): apiserver emits no-payload events with fresh resourceVersion to prevent watcher cache from falling off the back of the bounded etcd watch history (~5 min). Lets long-quiet watchers survive.
- **Reflector list-then-watch** ([reflector.go](https://github.com/kubernetes/client-go/blob/master/tools/cache/reflector.go)): paginated list at `RV=""`, then watch from `listRV`, retry on 410 Gone.

What Iris steals: UID + generation + observedGeneration, level-triggered protocol shape, workqueue structural dedup as the right pattern for per-worker wake events.

What Iris doesn't steal: informer/watch (we have direct DB access, no apiserver intermediary), leader election (single-controller), bookmarks (no resourceVersion-indexed watch history).

### Nomad

- **Blocking queries** ([nomad/client/client.go `watchAllocations`](https://github.com/hashicorp/nomad/blob/main/client/client.go)): client calls `Node.GetClientAllocs` with `MinQueryIndex`; server holds until state-store index advances. Returns `map[AllocID]ModifyIndex`. Client diffs locally, fetches full allocs for new/changed IDs via `Alloc.GetAllocs`.
- **Failure mode** ([nomad#18267](https://github.com/hashicorp/nomad/issues/18267)): a follower whose Raft index lags can return a *shrunk* alloc map. Client doesn't strictly enforce `resp.Index ≥ req.MinQueryIndex` and may GC live tasks. The cautionary tale for digest-based reconciliation against a distributed backing store.
- **Plan** ([nomad/scheduler](https://github.com/hashicorp/nomad/blob/main/scheduler/)): scheduler proposes server-internal Plan; leader applies via Raft; per-node alloc index bumps wake client blocking queries.

What Iris steals: digest-then-fetch (etags). Spec is the heavy thing; etag is the cheap digest.

What Iris doesn't: Iris' DB is single-process SQLite, no Raft staleness, so we don't need the index-freshness defense.

### Mesos / Marathon

- **Reconciliation API** ([Mesos docs](https://mesos.apache.org/documentation/latest/reconciliation/)): two variants. Explicit reconciliation: framework sends `(TaskID, AgentID)` list, master returns latest known state. Implicit: empty list, master returns all known non-terminal tasks.
- **Framework failover**: `failover_timeout` (Marathon ≥1 week); `FrameworkID` preserved across restarts.
- **Agent failover**: `--agent_reregister_timeout` (minutes); tasks declared `TASK_LOST` if exceeded.

What Iris steals: the explicit `Reconcile([uid1, uid2])` shape is implicit in our spec already, but Mesos validates the design: this is a useful debug/recovery RPC. Add `Reconcile(force_full=true)` flag for operator use.

### Ray

- **Autoscaler v2** ([docs](https://docs.ray.io/en/latest/ray-core/internals/autoscaler-v2.html)): Reconciler reads pending demands + Ray nodes + cloud instances, emits typed `InstanceUpdateEvent`s, dispatches via Instance Manager. Separates *decide* from *do*.
- **Actor restart**: `ActorID` persists across restarts, `num_restarts` increments, `WorkerID` rolls (new process). Logical vs physical incarnation distinction.

What Iris steals: the typed-event output shape. Our `WorkerReconcileOutputs.db_writes: list[TransitionDelta]` is the moral equivalent — emit events from the pure function, apply them in a separate step.

### Slurm

- **Ping vs registration** ([ping_nodes.c](https://github.com/SchedMD/slurm/blob/master/src/slurmctld/ping_nodes.c)): `REQUEST_PING` (cheap, frequent) vs `MESSAGE_NODE_REGISTRATION_STATUS` (full state, on boot/restart). Same RPC used at boot, after slurmctld restart, and periodically.
- **No UID**: nodes identified by hostname, jobs by integer JobID. Slurm accepts the risk and validates protocol versions / IPs at the RPC layer.

What Iris steals: the "one canonical full-state exchange" idea. Iris' `Reconcile` is already this — it's used at every tick, on worker startup, after controller restart.

### AWS ECS

- **ACS protocol** ([amazon-ecs-agent/agent/acs](https://github.com/aws/amazon-ecs-agent/tree/master/agent/acs)): agent calls `ecs:DiscoverPollEndpoint`, upgrades to WebSocket via `ecs:Poll`. Control plane pushes JSON messages; agent ACKs and pushes state back on the same stream.
- **TaskManifestMessage**: periodic full `(TaskArn, desiredStatus)` list. Agent doesn't act destructively — it sends `TaskStopVerificationMessage` to confirm before stopping unknowns.

What Iris considers (rejected): verification handshake before destructive kill. See spec §4.6. Worth revisiting only if we move off SQLite to a distributed store.

### Temporal

- **Task tokens** ([docs](https://docs.temporal.io/encyclopedia/detecting-activity-failures)): every dequeued activity carries an opaque `taskToken`. Every state-change RPC (`RespondActivityTaskCompleted`, etc.) must present it. Server rejects stale tokens with `NotFound`.
- **Workflow vs Run ID**: `WorkflowID` is user-supplied logical key; `RunID` is server-minted per execution (continue-as-new bumps it).

What Iris steals: the `attempt_uid` is exactly this — opaque, per-incarnation, server-minted, validated on every observation.

### Borg / Omega

- **Borg paper** ([Verma et al., EuroSys 2015](https://s3.amazonaws.com/systemsandpapers/papers/borg.pdf)): Borglet → link shards report full machine state every tick; link shards → Borgmaster leader report only the delta. Bounds master CPU regardless of cluster size.
- **Fail-open**: Borglet continues running existing tasks indefinitely if Borgmaster is unreachable. Cell does not depend on master availability for steady state.

What Iris steals (in follow-up, not this project): fail-open worker semantics. Iris' current `_serve` deadline timer triggers a destructive `_reset_worker_state` on controller blip — wrong for long TPU jobs. Documented in spec §3.4 as out-of-scope.

What Iris does *not* steal: stateless link-shard sharding. Premature for 100–1000 worker scale.

## Synthesis: the patterns we adopt

1. **UID + generation** (k8s, Temporal, Nomad). `attempt_uid` is the routing primitive; `desired_generation` lets us mutate specs mid-flight.
2. **Level-triggered protocol** (k8s, Slurm, Mesos implicit-reconcile). `desired` is the complete expected set every tick, not a delta.
3. **Digest-then-fetch** (Nomad). Etag-based spec caching collapses steady-state payload to ~40 bytes/attempt.
4. **Pure compute + apply split** (Ray autoscaler v2, k8s controller-runtime). `reconcile_worker(inputs) → outputs`; `apply(outputs)` is the separate step.
5. **Structural dedup** (k8s workqueue). Per-worker wake events coalesce structurally, not via timers.
6. **Fail-open semantics** (Borg) — explicitly *deferred* but designed-around in the protocol so the eventual change is small.

## Patterns we deliberately reject

- **Streaming RPC / xDS-style ADS** ([Envoy xDS](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol)). Operational pitfalls (half-open TCP, sticky connections, retry storms) outweigh the latency gain at our scale. Compatible upgrade path preserved.
- **Pull with long-blocking-query** (Nomad, Temporal). Adds NAT-friendliness we don't need; loses controller-knows-who-needs-attention property.
- **Verification handshake before destructive GC** (ECS). Defense against a failure mode (stale distributed state) we don't have.
- **Watch bookmarks / informer cache** (k8s). We have direct SQL access, no need for a watch-cache intermediary.
- **Leader-election leases** (k8s `coordination.k8s.io`). Single-controller for now.
