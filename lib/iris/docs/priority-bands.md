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
  Workloads to admit a higher-priority one — including when Topology-Aware
  Scheduling can't otherwise place it on full nodes. This is what lets a
  higher-priority multi-host gang reclaim capacity from running `batch` gangs.
  Preemption is whole-Workload (gang-aware): Kueue evicts a full lower-priority gang,
  not a stray pod out of it.
- **VM/TPU clusters.** There is no Kueue; the Iris controller's own scheduler ranks
  pending tasks by band and reclaims slices directly.

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

Budget rows are keyed by the authenticated principal stored in
`jobs.submitting_user`. IAP and JWT submissions use the verified email or
service-account identity. The friendly owner in a job path remains independent:
an admin may submit `/power/train` while Iris accounts the job to
`russell.power@openathena.ai`. Child jobs inherit the root submitter, and a
federated handoff carries the same principal to the receiving cluster. Trusted
local submissions (`local_admin`) and rows with an empty `submitting_user` use the
job-path owner as a fallback budget key.

If a higher-band submission is rejected:

1. **Use the appropriate lower band.** Ordinary research should run at
   INTERACTIVE or BATCH.
2. **Check your budget identity.** The error reports the authenticated email or
   service account used for budget lookup. Confirm that exact value appears in
   the cluster config. Trusted local submissions instead use the nickname at
   the start of the job path.
3. **Request an uplift.** If your work needs INTERACTIVE budget headroom or an
   admin-only band, ping [@Helw150](https://github.com/Helw150).

## See also

- [`task-states.md`](task-states.md) — how preemption surfaces in task state
- [`OPS.md`](../OPS.md) — operator-side scheduler inspection
