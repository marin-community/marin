# Priority Bands

Iris ranks pending tasks by **priority band** before per-user fairness. Four
bands exist (defined in [`job.proto`](../src/iris/rpc/job.proto)):
`SYSTEM`, `PRODUCTION`, `INTERACTIVE`, and `BATCH`. Choose the right band for
what you are running; higher bands can disrupt running work below them.

| Band | Selected via | Behavior |
|---|---|---|
| `SYSTEM` | `--priority system --system-reason=<reason>` | Admin-only Iris, Finelog, and hero work. Runs before and can preempt every other band. Never downgraded by the budget system. |
| `PRODUCTION` | `--priority production` | Admin-only critical organizational work. Yields to SYSTEM; preempts INTERACTIVE/BATCH. Never downgraded by the budget system. |
| `INTERACTIVE` | default (or `--priority interactive`) | Normal work. Yields to SYSTEM/PRODUCTION; preempts BATCH. |
| `BATCH` | `--priority batch` | Opportunistic. Yields to every higher band. Safe to launch in bulk. |

## When to use each band

### SYSTEM

Use only for Iris and Finelog infrastructure or hero runs. SYSTEM submission
requires admin authorization. CLI submissions must include
`--system-reason=<reason>` with `hero`, `finelog`, or `iris` as a complete word;
the reason remains in the job's submission argv for audit.

### PRODUCTION

Use for critical organizational work that should reclaim ordinary cluster
capacity but must yield to Iris, Finelog, and hero runs. PRODUCTION submission
also requires admin authorization.

### INTERACTIVE

The default band. Use for everyday research: training runs, ad-hoc evaluation,
debugging, single-shot experiments. Most jobs belong here.

### BATCH

Use for work you are happy to have preempted by anyone else. Equivalent to
`sc-loprio` on the NLP cluster. Good candidates:

- Hyperparameter sweeps
- Batch inference / offline evaluation
- Large fan-out experiments where any individual run can be retried
- Anything you want to run *a lot* of without crowding out the cluster

BATCH jobs are the polite default when you don't strictly need a result soon.

## How preemption is enforced

The band a job runs at maps to a Kubernetes PriorityClass
(`iris-{system,production,interactive,batch}`, values 10000/1000/10/0) stamped
on every pod. How that band turns into actual preemption depends on the backend:

- **K8s GPU clusters (CoreWeave).** Every pod is admitted through Kueue. Kueue reads
  the pod's PriorityClass as the Workload priority and, with the ClusterQueue's
  `preemption.withinClusterQueue: LowerPriority` policy, evicts lower-priority
  Workloads when binding GPU or RDMA quota is exhausted. Topology-Aware
  Scheduling checks that those victims make a compatible placement available.
  This is what lets a higher-priority multi-host gang reclaim full nodes from
  running `batch` gangs instead of waiting behind non-binding quota.
  Preemption is whole-Workload (gang-aware): Kueue evicts a full lower-priority gang,
  not a stray pod out of it.
  CoreWeave health-check Pods bypass Kueue and run at Kubernetes priority `-1`;
  their unconstrained-topology annotation keeps them out of TAS capacity accounting.
  Once Kueue admits an Iris workload, kube-scheduler can preempt those Pods because
  every Iris PriorityClass, including `iris-batch`, uses `PreemptLowerPriority`.
- **VM/TPU clusters.** There is no Kueue; the Iris controller's own scheduler ranks
  pending tasks by band and reclaims slices directly. A preemptor only replaces one
  running slice of its own variant and at least its size, so on its own it cannot
  free a reservation held by smaller slices. See
  [Reservation reclaim](#reservation-reclaim-vmtpu).

## Reservation reclaim (VM/TPU)

A reserved pool is the set of scale groups with `capacity_type: reserved` that
share one `quota_pool` (every size of a `tpu_pools` entry, for example
`v4-reserved/us-central2-b`). When smaller, lower-band slices hold the whole
reservation, a larger slice for a higher-band job waits on the cloud and
preemption cannot help it. Reservation reclaim deletes lower-band slices so the
waiting slice can be fulfilled. It is **off** until a pool is listed:

```yaml
defaults:
  autoscaler:
    reservation_reclaim:
      quota_pools: [v4-reserved/us-central2-b]
      min_wait: { seconds: 600 }     # waiting slice's minimum time on the cloud before reclaim
      cooldown: { seconds: 1800 }    # minimum time between reclaims in one pool
      highest_victim_band: interactive   # or batch, to spare interactive slices
```

Each autoscaler pass, per listed pool:

- **Trigger:** a slice of the pool is still waiting on the cloud (BOOTING) after
  `min_wait`, and coscheduled demand is routed to its group. The highest-band,
  then oldest, waiting slice is served.
- **Victims:** READY slices of other groups in the pool whose running tasks are
  all strictly below the waiting demand's band and no higher than
  `highest_victim_band`. A slice with one task outside that range is skipped whole.
- **Order:** idle slices first, then the lowest band, then the fewest chips, then
  the youngest. Reclaim stops once the freed chips reach the waiting slice's chips.
- **All or nothing:** if the candidates cannot free enough chips, nothing is
  deleted and the autoscaler logs `reclaim_blocked` with reason `too_few_chips`.
- **Effect:** the controller fails the victims' workers (their tasks become
  `WORKER_FAILED` and retry on `max_retries_preemption`) and deletes the slices.
  Other groups in the pool may not scale up until the waiting slice stops waiting
  (READY, failed, or gone), for at most `cooldown`, so the freed chips go to it.
- **Once per waiting slice:** the pool reclaims nothing more while that slice
  still waits on the cloud. Each pass logs `reclaim_blocked` with reason
  `awaiting_capacity` for it, so a reservation the cloud never fills is not
  drained again and again. After the slice stops waiting, the pool may reclaim
  again once `cooldown` has passed.
- **Same-tick safety:** workers that receive a task in the same tick count as busy
  at that task's band, so a task placed this tick is never a victim.
- **Record:** a `reservation_reclaim` autoscaler action names the waiting slice,
  the band, the chips needed and freed, and every victim slice. Each victim worker's
  failure reason names the pool and the waiting slice.

Iris cannot see how many chips are free in the cloud reservation, so reclaim
assumes none are and frees the waiting slice's full chip count. It may delete more
than strictly needed, never less. The cooldown is held in memory, so a controller
restart resets it.

A preempted job surfaces as described in [`task-states.md`](task-states.md) and is
requeued for retry.

On Kubernetes, single-task CPU coordinators have a PodDisruptionBudget whose
availability policy follows the job's band. SYSTEM and PRODUCTION coordinators
use `minAvailable: 1`, so a voluntary node drain waits for operator action.
INTERACTIVE and BATCH coordinators use `maxUnavailable: 1`, so a drain may
evict the singleton pod. Iris records that eviction as `PREEMPTED` and retries
it within `max_retries_preemption`.

An evicted coordinator loses in-memory and node-local state. INTERACTIVE and
BATCH coordinators must keep durable progress outside the pod and make repeated
external writes safe. A PDB cannot protect SYSTEM or PRODUCTION from a hard
node failure.

## How band selection interacts with budgets

Per-user budget tracking lives in
[`controller/budget.py`](../src/iris/cluster/controller/budget.py). When a user
exceeds their budget, INTERACTIVE submissions are silently downgraded to BATCH.
SYSTEM and PRODUCTION are exempt.

See [shared budgets](federation.md#shared-budgets) for federation configuration.

### Max-band caps and unlisted users

Each user who is listed in the cluster config's `user_budgets` tier list has a
`budget_limit` and `max_band` recorded in the `user_budgets` table.
**Unlisted** users don't get a row; they inherit `UserBudgetDefaults` at read
time — a small INTERACTIVE budget that silently falls through to BATCH once
exceeded. Submissions at a higher band than `max_band` are **rejected** (not
downgraded) with `PERMISSION_DENIED`. The tiers reconciled from the cluster
config at startup are:

- Admins — `SYSTEM`, `PRODUCTION`, and everything below; large budget.
- Listed researchers — `INTERACTIVE` (plus `BATCH`), large budget.
- Everyone else (including new/unlisted users) — `INTERACTIVE` with a small
  default budget; jobs run INTERACTIVE while within budget and degrade to
  BATCH once exceeded. SYSTEM and PRODUCTION submissions are rejected.

Budget rows are keyed by the job owner stored in `jobs.user_id`: `/power/train`
uses the `power` budget. Spend, priority limits, scheduling fairness, task caps,
and dashboard counts use this same owner. Non-admin top-level submissions use the
authenticated identity as their owner; admins can choose a different owner.
The authenticated submitter remains in `jobs.submitting_user` for federation
authorization, but does not determine the budget.

If a higher-band submission is rejected:

1. **Use the appropriate lower band.** Ordinary research should run at
   INTERACTIVE or BATCH.
2. **Check your budget identity.** The error reports the job owner used for
   budget lookup. Confirm that exact value appears in the cluster config.
3. **Request an uplift.** If your work needs INTERACTIVE budget headroom or an
   admin-only band, ping [@Helw150](https://github.com/Helw150).

## See also

- [`task-states.md`](task-states.md) — how preemption surfaces in task state
- [`OPS.md`](../OPS.md) — operator-side scheduler inspection
