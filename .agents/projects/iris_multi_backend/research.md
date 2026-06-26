# Multi-Backend — Research

Background for [`design.md`](./design.md). In-repo seam, prior art, what surprised us, the `codex`
review log, and the alternatives we rejected. Code permalinks are pinned to
`1013be215490cce01d095518ba3c07bbe0de0a7f` (origin/main at authoring time).

## The current seam (what we reuse vs. what's new)

Framed as **exact reuse** / **pattern reuse** / **new work**:

- **`TaskBackend` contract** —
  [`controller/backend.py:340`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/backend.py#L340):
  `schedule` (pure decision over a DB snapshot) / `reconcile` (stateless I/O) / `autoscale` +
  on-demand. **DB-less** — it never touches the controller DB. *Exact reuse of the contract; the whole
  backend relocates into the agent.* Two impls assembled by
  [`composer.py:46`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/composer.py#L46) /
  [`:158`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/composer.py#L158):
  `RpcTaskBackend`
  ([`backends/rpc/backend.py:133`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/backends/rpc/backend.py#L133),
  `{WORKER_DAEMON, IRIS_AUTOSCALER}`) and `K8sTaskProvider`
  ([`backends/k8s/tasks.py:1386`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L1386),
  `{CLUSTER_VIEW}`; Kueue schedules), selected by `config.provider_kind()`
  ([`config.py:506`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/config.py#L506)).
- **Three distinct concepts in `cluster/backends/` that must not be conflated** (this was a correction
  mid-design — see "What surprised us"):
  1. **`TaskBackend` = "the backend"** — `rpc/` (`RpcTaskBackend`, worker-daemon) and `k8s/tasks.py`
     (`K8sTaskProvider`). The two real backends.
  2. **`WorkerInfraProvider` = a "worker provider", NOT a backend** —
     [`protocols.py:94`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/backends/protocols.py#L94):
     `gcp/` (`GcpWorkerProvider`), `manual/`, `local/` (`create_slice`/`list_all_slices`). **GCP is the
     worker provider for the worker-daemon backend, not a backend of its own.** k8s has none (Kueue/the
     cluster autoscaler own nodes).
  3. **`ControllerProvider`** —
     [`protocols.py:21`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/backends/protocols.py#L21):
     how the controller/agent VM itself is provisioned. `factory.create_provider_bundle()` returns
     `{controller, workers}` by `platform_kind()`.

  So "GCP TPU" = the **worker-daemon backend** on the **gcp platform**; "CoreWeave" = the **k8s
  backend** on CoreWeave's k8s. The platform (gcp/coreweave/manual) is orthogonal to the backend kind
  (worker-daemon/k8s).
- **The substrate is already the record.** Worker daemon holds live attempt state; k8s etcd holds pods;
  the autoscaler treats cloud `list_all_slices()` as truth (not the `slices` table — the
  scale-group-rename cleanup). *This is what makes a DB-less agent recoverable.*
- **The worker wire is already one reconcile.**
  [`worker.proto:154`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/rpc/worker.proto#L154):
  `Reconcile(ReconcileRequest{desired: repeated DesiredAttempt}) -> (observed, health)`, where the
  worker kills any attempt **absent** from the desired set (the existing zombie-kill). The remote wire
  is this exact shape, lifted one level up. **No sync cursor exists today** — the worker re-sends the
  full desired set each tick; spec-caching already deltas the heavy payload, so `sync_id` generalizes
  that with an explicit version.
- **Constraints** (`scheduling/scheduler.py`): `Constraint{key, op, value(s), mode}` + a posting-list
  `ConstraintIndex`; `attempt_uid` exists and is `UNIQUE` — the idempotency/CAS key. *Matcher is
  pattern-reused for backend routing.* `route_demand`
  ([`autoscaler/routing.py:627`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/autoscaler/routing.py#L627))
  routes *task demand → scale groups* — **pattern reuse, not a drop-in** backend router.
- **Auth** — `_create_worker_jwt`
  ([`auth.py:462`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/auth.py#L462))
  mints only `system:worker`; minted in `create_controller_auth`
  ([`auth.py:322`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/controller/auth.py#L322)).
  No `system:controller` role / backend-scoped authz yet — *new work*.

## What surprised us

- **The agent does NOT carve the backend into an "I/O half."** An early framing said the agent runs the
  reconcile-I/O half while the root keeps the decision half. Wrong: the whole `RpcTaskBackend` /
  `K8sTaskProvider` (schedule *and* reconcile) runs wholesale inside the agent, because two-level
  scheduling puts the local task→worker `Scheduler` in the agent. The root's meta-scheduler
  (task→backend) is *new*, not the relocated half.
- **Backend ≠ worker-provider.** GCP is a worker *provider* for the worker-daemon backend, not a backend
  peer of k8s. The refactor adds a **transport axis only** (the rpc & k8s backends still exist, just
  tunneled over `Poll`); it does not create `backends/{remote,gcp,k8s}` peers.
- **Port reservations have no substrate footprint today.** `TaskAttempt.adopt()`
  ([`task_attempt.py:303`](https://github.com/marin-community/marin/blob/1013be215490cce01d095518ba3c07bbe0de0a7f/lib/iris/src/iris/cluster/worker/task_attempt.py#L303))
  rebuilds `ports={}` and Docker discovery lacks port metadata — so a worker-local fact violates the
  recoverability invariant. We must stamp allocated ports on the substrate object and re-reserve in
  `PortAllocator` on adopt. (This is a latent double-allocation bug **today**, independent of
  multi-backend.)
- **The substrate is durable for *running* work but not for *uncommitted terminal status*.** k8s GCs
  terminal gang pods ~60 s (a missing pod then reads as `WORKER_FAILED`); Docker cleans exited
  containers; the worker currently discards non-running containers. Hence ack-gated terminal retention.

## Prior art (distilled)

| Question | Finding | Decision |
|---|---|---|
| Push vs pull | Pull spans K8s + VM pools behind NAT without the hub hoarding inbound creds (Armada, HTCondor glidein, Temporal). | **Agent dials home, is the RPC client.** |
| Who owns state | Survivors keep their own store authoritative; **never mirror member runtime state** (KubeFed death, Ray GCS wall) — but **one** store is fine and simplest. | **Single root DB** for authoritative state; agents hold only a recoverable cache. |
| Safe re-placement | Dedup of observations ≠ stopping execution; need a fence the execution layer obeys. | Launch-lease + self-fence + epoch + CAS. |
| Affinity | Gangs stay in one cluster (Kueue all-or-nothing; Armada `gangNodeUniformityLabel`). | Root job pins to one backend; descendants inherit. |
| Overflow / scarcity | Central queue + late binding (Armada) beats speculative submit. | Queue centrally + atomic admission. |
| Cross-cluster preemption | No live migration anywhere; re-queue on a fresh attempt + cause-scoped retry (GCP exit 50001). | Preemption ⇒ attempt failure on the preemption budget; re-placement re-routes. |
| Identity | Globally-unique IDs encoding origin (Slurm). | Job/task IDs root-owned & global; infra IDs stamped with `attempt_uid` + `backend_id`. |

Per-system detail:

- **Armada** (the reference): central queue + **pull executors** with a capacity snapshot; ordered
  idempotent updates, run-ID dedup, **ack-gated lease/cleanup**, heartbeat-via-lease, executor-pushed
  token auth, atomic gang; lease-expiry terminal by default; millions of jobs/day. *Closest to our
  design's ack-gated, agent-dials-home, recoverable-cache model.*
- **MultiKueue** (closest K8s-native): hub-spoke, push, two-phase clone + `managedBy` inert mirror,
  first-worker-to-admit wins, GC-by-origin-label re-adoption, `WorkerLostTimeout` 15 min; not >1M
  jobs/day; no global quota/fairness.
- **Karmada**: second control plane; per-cluster push/pull `syncMode`; graceful eviction (off by
  default); ~100 clusters.
- **Slurm federation**: peer-to-peer push; sibling jobs (first-to-start wins, origin revokes); 6-bit
  origin job IDs; no live migration.
- **HTCondor glidein**: pull overlay (startd dials home; CCB) — NAT-traversal proof point.
- **Temporal**: workers long-poll task queues (outbound only); central event-history truth;
  deterministic replay; at-least-once + idempotency — the stateless-member proof point.
- **Nomad**: region isolation, clients pull from local region, authoritative region for ACLs.
- **Ray/KubeRay**: no federation; single GCS control store is the scaling wall — cautionary tale (one
  store can't sync every node's state; we keep the root store to *intent + observations*, not per-node
  sync).
- **KubeFed (archived 2023)**: died aggregating dynamic member state into one hub API — the "don't
  mirror runtime state, keep the hub store to intent" lesson.
- **AWS/GCP Batch**: cause-scoped retry (GCP exit 50001 = preemption) → maps to KILLED-vs-WORKER_FAILED.

## `codex` review log (converged to CLOSED)

| Round | Input | Outcome |
|---|---|---|
| 1 | Stateful subordinate controllers (rev 1) | 3 CRITICAL / 6 MAJOR: split-brain mitigation weak, needs durable event-log, transport reverse-RPC hard. → abandoned the stateful model. |
| 2 | Single DB + stateless agents | Right direction; 3 CRITICAL: stale-cache double-launch (gate the *launch*, not just the kill); "substrate is durable" false for uncommitted terminals; need a root leadership epoch. |
| 3 | Closure draft (4 mechanisms) | 0 CRITICAL; 5 MAJOR spec-completeness: K8s `attempt_uid` metadata + UID-keyed CAS; launch-lease post-create re-check; monotonic-duration leases; explicit ack disposition; wire must carry capacity/autoscale. |
| 4 | Closure delta | All five resolved. **CLOSED.** |
| 5 | Recoverable-cache worker model + rollout | 0 CRITICAL, 3 MAJOR: worker-local **port reservations** not recoverable (stamp + re-reserve on adopt); "preemption re-derived" holds only pre-side-effect (recover via terminal-retention); split rollout transport-only from substrate-recovery. |
| 6 | Round-5 MAJORs incorporated | **CLOSED.** |
| 7 | Wire collapse to one `Poll()` + `sync_id` + layout + config | 0 CRITICAL, 2 MAJOR: reroute must be explicitly two-phase (remove-from-old *first*, else every Poll renews the old lease and it never drains); lease renewal must be response-applied + root-conservative (lost response ⇒ over-wait, never under-wait). |
| 8 | Round-7 MAJORs incorporated | **CLOSED** — no remaining CRITICAL/MAJOR. |

Net: the "single DB, recoverable-cache backend" instinct was correct and is *simpler* than the stateful
model; the work was making it safe with four bounded mechanisms (epoch, monotonic launch-lease,
self-fence, ack-gated-terminal + CAS) — none a second database.

## Alternatives considered (and why the body design wins)

- **Option S — stateful subordinate controllers (rejected).** Each backend a full controller with its
  own DB + scheduler, reconciled against a durable per-backend `BackendEvent` log with cursor-ack
  exactly-once apply + full-resync. Rejected: the autonomy that buys fault-isolation also buys divergent
  authoritative state, N+1 databases to back up / migrate / version-skew, and a backend-DB-loss failure
  mode. Option C keeps the fault-isolation without a second authority.
- **Option H — minimal-state agent (durable outbound buffer only).** Folds into Option C: because the
  substrate is the durable record, any unreported observation is recoverable by re-listing on restart,
  and ack-gated retention covers the one case the substrate drops. The buffer is redundant.
- **Reverse-RPC transport (root dials backends; rejected).** Makes the root hoard inbound creds to every
  cluster, needs an inbound endpoint on each (NAT/firewall pain), and inverts the trust boundary. The
  agent-dials-home model is NAT-friendly and keeps one outbound identity per backend.
- **Speculative sibling submission (Slurm-style; deferred).** Wastes capacity under scarcity and needs
  cross-backend revoke races; central queue + late binding + queue-until-available gives the same "run
  wherever frees up first" without speculation.
- **C1 vs C2 worker-placement variants — both superseded.** C1 (root mirrors every backend's idle-worker
  inventory; one global scheduler) has a root-scheduler scaling ceiling. C2 (root holds no idle workers;
  agent re-derives placement *statelessly* each tick) is subtle — no local memory to avoid cross-tick
  double-assignment. Allowing the agent a *recoverable* local worker DB takes both off the table: it's
  effectively "C2 with a recoverable cache" — the agent *remembers* its assignments and that memory is
  disposable because it rebuilds from the substrate. No `ProposePlacements` round-trip (each task lands
  on one backend; launch-lease + generation provide safety).
