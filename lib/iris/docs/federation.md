# Federation: what crosses a cluster boundary

A cluster with `peers:` in its config may hand whole jobs to another cluster. `marin` (GCP:
TPU and CPU) peers with `cw-rno2a` and `cw-us-east-02a` (CoreWeave: H100), so a user submits
every job to `marin` and GPU work lands on CoreWeave.

This page is the job model. For the auth, networking, and DNS that carry a handoff, see
[`coreweave.md`](coreweave.md).

## Routing: one decision, at submit

`PeerRouter.decide` (`cluster/federation/router.py`) runs once per submission, in order:

1. **A `cluster=<peer>` pin wins.** `iris job run --target-cluster <peer>` sets it. The job
   goes to that peer even when it could run locally.
2. **Otherwise local, if the shape is locally feasible.** `job_feasibility` asks whether any
   local scaling group could in principle host the job.
3. **Otherwise the first peer that can host it**, by peer id, ascending.
4. **Otherwise local**, so the job fails as locally unschedulable rather than wedging.

A peer "can host" a job when its last capability heartbeat (a 30s `ListBackends` probe) shows
a backend whose advertised attributes satisfy **every routing constraint** on the job.

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

There is no `iris peers` command. Reachability and advertised shapes come from the
`ListPeers` RPC; handoff state lives in the parent's `federated_jobs` table.

```bash
# Which peers are reachable, and what each advertises
uv run iris --cluster=marin rpc controller list-peers

# Where a job was handed off, to whom, and under which principal
uv run iris --cluster=marin query \
  "SELECT job_id, peer_id, owner_principal, handoff_state FROM federated_jobs"
```

A federated job's tasks live on the peer and are mirrored back, so `iris job summary` reports
its state from `marin`. Its logs are relayed asynchronously into `marin`'s finelog and lag
behind a log-heavy job; `job summary` is the reliable liveness answer.
