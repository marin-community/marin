# Federation: what crosses a cluster boundary

A cluster with `peers:` in its config may hand whole jobs to another cluster. `marin` (GCP:
TPU and CPU) peers with `cw-rno2a` and `cw-us-east-02a` (CoreWeave: H100), so a user submits
every job to `marin` and GPU work lands on CoreWeave.

This page is the job model. For the auth, networking, and DNS that carry a handoff, see
[`coreweave.md`](coreweave.md).

## Routing: classify at submit, place on the tick

Peer placement is not decided at submit. Submit only *classifies* a job; the controller's
single scheduling tick decides which peer a queued job lands on, alongside every local
scheduling decision. This keeps one thread of control for all placement and lets a job wait
for a peer to report free capacity instead of piling onto the first peer that merely *could*
host it.

`PeerRouter.classify` (`cluster/federation/router.py`) runs once per submission and returns one
of three dispositions:

1. **`QUEUE` on a `cluster=<peer>` pin.** `iris job run --target-cluster <peer>` sets it. The
   job queues for that peer even when it could run locally.
2. **`LOCAL` if the shape is locally feasible.** `job_feasibility` asks whether any local
   scaling group could in principle host the job.
3. **`QUEUE` if some reachable peer could host it** — its last capability heartbeat shows a
   backend whose advertised attributes satisfy **every routing constraint** on the job.
4. **`REJECT` otherwise**, so the job fails fast as unschedulable rather than wedging.

A `QUEUE` disposition parks the job in the parent's `federated_jobs` table in the
`QUEUED_HANDOFF` state. It is *not* yet assigned to a peer.

## Placement: the control tick drains the queue against availability

Each control tick, `FederationManager.plan_federation` (`cluster/federation/manager.py`) runs
the pure pass in `cluster/federation/availability.py` over the queued jobs and the peers' most
recent availability, and emits `(job → peer, backend)` promotions. A promotion is applied as a
conditional CAS (`promote_queued_handoff`): it advances the job to a pending handoff only if the
job is still queued, not cancelled, and non-terminal — so a cancel or terminalize racing the
tick can never be overtaken by a promotion. A confirmed promotion then delivers over the same
handoff machinery as before.

Placement translates a job's device request into an **availability gate**: N replicas of an
8×`h100` job becomes `ge(available:h100, N*8)`, and a peer backend hosts the job only when its
advertised free capacity meets the gate. This mirrors the `availability:<variant>` *EXISTS*
constraints the scheduler already uses for reservations, but the `available:<token>` metric is
*numeric* (a count), not a boolean.

Two properties keep placement honest without pretending to be exact:

- **Never summed across backends.** A job pins to one backend, so 6 free on one backend plus 4
  on another does not host an 8-GPU job. Availability is evaluated per backend.
- **A reservation ledger bounds cross-tick over-assignment.** The tick runs on every submit
  wake — far more often than the 30s heartbeat — so a naive per-tick read would re-spend the
  same advertised number every tick. The ledger records capacity already promoted against a
  peer backend *since its last heartbeat* (keyed on the heartbeat's `observation_epoch_ms`), and
  effective availability is `advertised − reserved`. A strictly newer heartbeat — whose number
  already reflects the delivered jobs — resets the ledger. Over-assignment is thus bounded to a
  peer's advertised free capacity per observation, which the issue explicitly tolerates; the
  peer's own scheduler (and a requeue) is the backstop.

`max_federation_handoffs_per_cycle` (controller config) caps promotions per tick, a second
bound on a burst against stale metrics.

## What the router matches on

Routing constraints are the subset of constraints marked `routing=True` in
`CONSTRAINT_REGISTRY` (`cluster/constraints.py`): `device-type`, `device-variant`,
`preemptible`, `region`, `zone`.

Two consequences catch people out:

- **`gpu-count` is not a routing constraint.** It is a consumable, checked against a worker's
  free GPUs when the peer schedules the job. `H100x1` and `H100x8` route identically. GPUs
  pack: several tasks share one 8-GPU node, unlike a TPU VM, which is atomic.
- **A peer that advertises no `region` satisfies no `region` constraint.** An advertised
  attribute the peer omits makes every constraint on that key fail. The CoreWeave backends
  advertise only `device-type` and `device-variant`, so any job carrying a region or zone
  constraint stays local.

That second point bites sub-jobs specifically. `IrisClient.submit` (`iris/client/client.py`)
gives a child job its parent worker's region unless the child names a region itself, which
keeps a child near its data. A GPU sub-job must opt out with fray's `ANY_REGION` sentinel —
`ResourceConfig(..., regions=[ANY_REGION])` — a region-EXISTS marker that suppresses the
inheritance and is then dropped before the wire.

## Only whole root jobs are federated

A peer runs a handed-off job under the same, cluster-invariant job id. A child job's id names
a parent the peer does not have, so a peer can only accept a root. `launch_job` refuses a
non-root job that routes to a peer, naming `--target-cluster` as the remedy.

So a job tree lives entirely on one cluster. A coordinator on `marin` cannot dispatch its
training sub-job to CoreWeave; pin the coordinator instead, and its whole tree runs there.

The identity rule reinforces this. A federated job must carry an accountable user: the peer's
`auth.allowed_submitters` gates on the submitter, and a `local_admin` (CIDR/loopback)
identity is refused before the handoff. In-cluster workers authenticate by network location,
so a job submitted by a worker is `local_admin` — a root submitted by a logged-in user is the
only thing that federates.

## What the peer checks when a handoff arrives

A handoff is authorized as a peer-to-peer delivery, not as a user submission. The peer
verifies the requesting cluster's signed identity and its own `allowed_submitters` against the
principal the parent asserts, then admits the job under the parent's job id.

What it does *not* re-run is the client-freshness gate. That gate makes humans upgrade a stale
`marin-iris` CLI, and the wire client here is the parent controller, which re-encodes the
request from its own stored job state — the submitter's `client_revision_date` is not part of
that state and does not survive the round-trip. The parent already gated the submitter's client
when it accepted the job, and a queued job may wait longer than the freshness window before it
is delivered, so gating on arrival would reject handoffs for a staleness the peer cannot
actually observe.

## What travels with the job

`FederationManager._inline_blobs` (`cluster/federation/manager.py`) carries the workspace
bundle and any offloaded workdir file into the handoff as bytes, because a peer reads its own
bundle store and cannot resolve a content id minted by the parent.

Environment travels too, and it wins over the peer's own defaults: a child inherits its
parent's `env`, and a job's explicit env overrides a cluster's `defaults.task_env`. Passing
`-e MARIN_PREFIX gs://…` to a job bound for CoreWeave therefore delivers a `gs://` path to a
pod holding only S3 credentials.

## Credentials do not travel

Each cluster's task pods carry that cluster's credentials, and only those:

| Cluster | Task pods can read |
| --- | --- |
| `marin` (GCP) | `gs://` via the `iris-worker` service account |
| `cw-*` (CoreWeave) | `s3://` via the `iris-task-env` secret (CoreWeave AI Object Storage) |

There is no cross-cloud identity. A job on CoreWeave cannot read `gs://`, and a job on
`marin` cannot read `s3://marin-us-east-02a` unless it is handed AWS credentials explicitly.
Every artifact a federated job touches must live in the peer's object store.

## Observing federation

There is no `iris peers` command. Reachability, advertised shapes, and free capacity come
from the `ListPeers` RPC; handoff state lives in the parent's `federated_jobs` table.

```bash
# Which peers are reachable, what each advertises, and how much is free
uv run iris --cluster=marin rpc controller list-peers

# Where a job was handed off, to whom, and under which principal
uv run iris --cluster=marin query \
  "SELECT job_id, peer_id, owner_principal, handoff_state FROM federated_jobs"
```

Each backend in the `list-peers` output carries an `availability` block — the capacity metric
the queue gates on (`{"version": 1, "observation_epoch_ms": …, "amounts": {"h100": 504},
"total_amounts": {"h100": 512}}`). `amounts` is the free count the gate evaluates;
`total_amounts` is the matching denominator, shown on the dashboard's Backends page as
free/total meters per device variant. A backend with **no** `availability` block supplies no
metric and is matched on shape alone, so jobs route to it without a capacity check.

A queued job and a delivered one are both `pending` on the parent, but they are waiting on
different things, and `job list` says which:

| Pending reason | Meaning |
| --- | --- |
| `Queued for a federation peer to report free capacity` | In the queue, no peer chosen yet |
| `Queued for peer X to report free capacity` | In the queue, pinned to `X` (`--target-cluster`) |
| `Awaiting acceptance by peer X` | Promoted and delivered; awaiting the peer's admission |
| `Handed off to peer X; awaiting first status report` | Admitted; the first sync has not landed |

A job that sits on the first two lines is waiting for capacity, not stuck: compare its device
request against the peer's advertised `amounts`. A job cancelled while queued never reaches the
peer and terminates as `Cancelled before handoff`.

A federated job's tasks live on the peer and are mirrored back, so `iris job summary` reports
its state from `marin`. Its logs are relayed asynchronously into `marin`'s finelog and lag
behind a log-heavy job; `job summary` is the reliable liveness answer.
