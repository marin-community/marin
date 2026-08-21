# Priority Bands

Iris ranks pending tasks by **priority band** before per-user fairness. Four
bands exist (defined in [`job.proto`](../src/iris/rpc/job.proto)):
`PRODUCTION`, `PRIORITY`, `INTERACTIVE`, and `BATCH`. Choose the right band for
what you are running; higher bands can disrupt running work below them.

| Band | Selected via | Behavior |
|---|---|---|
| `PRODUCTION` | `--priority production --production-needed=<reason>` | Admin-only operational and hero work. Runs before and can preempt every other user band. Never downgraded by the budget system. |
| `PRIORITY` | `--priority priority` | Important time-sensitive work. Yields to PRODUCTION; preempts INTERACTIVE/BATCH. |
| `INTERACTIVE` | default (or `--priority interactive`) | Normal work. Yields to PRODUCTION/PRIORITY; preempts BATCH. |
| `BATCH` | `--priority batch` | Opportunistic. Yields to every higher band. Safe to launch in bulk. |

## When to use each band

### PRODUCTION

Use **only** for Iris and Finelog infrastructure, hero runs, or similarly
critical organizational work. Production submission requires admin
authorization. The CLI also requires `--production-needed=<reason>` and stores
that reason in the job's submission argv for audit.

### PRIORITY

Use for important, time-sensitive research that should reclaim ordinary cluster
capacity but must still yield to infrastructure and hero runs. This is the
appropriate replacement for most historical uses of PRODUCTION.

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
(`iris-{production,priority,interactive,batch}`, values 1000/100/10/0) stamped
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
availability policy follows the job's band. PRODUCTION coordinators use
`minAvailable: 1`, so a voluntary node drain waits for operator action.
PRIORITY, INTERACTIVE, and BATCH coordinators use `maxUnavailable: 1`, so a
drain may evict the singleton pod. Iris records that eviction as `PREEMPTED`
and retries it within `max_retries_preemption`.

An evicted coordinator loses in-memory and node-local state. PRIORITY,
INTERACTIVE, and BATCH coordinators must keep durable progress outside the pod
and make repeated external writes safe. Use PRODUCTION only when the coordinator
must block voluntary maintenance; a PDB cannot protect it from a hard node failure.

## How band selection interacts with budgets

Per-user budget tracking lives in
[`controller/budget.py`](../src/iris/cluster/controller/budget.py). When a user
exceeds their budget, PRIORITY and INTERACTIVE submissions are silently
downgraded to BATCH. PRODUCTION is exempt.

### Max-band caps and unlisted users

Each user who is listed in the cluster config's `user_budgets` tier list has a
`budget_limit` and `max_band` recorded in the `user_budgets` table.
**Unlisted** users don't get a row; they inherit `UserBudgetDefaults` at read
time — a small INTERACTIVE budget that silently falls through to BATCH once
exceeded. Submissions at a higher band than `max_band` are **rejected** (not
downgraded) with `PERMISSION_DENIED`. The tiers reconciled from the cluster
config at startup are:

- Admins — `PRODUCTION` (and everything below), large budget.
- Listed researchers — `INTERACTIVE` (plus `BATCH`), large budget. Operators
  can grant `PRIORITY` through `max_band` without granting PRODUCTION.
- Everyone else (including new/unlisted users) — `INTERACTIVE` with a small
  default budget; jobs run INTERACTIVE while within budget and degrade to
  BATCH once exceeded. PRODUCTION submissions are rejected.

If you hit `User <name> cannot submit PRODUCTION jobs (max band: INTERACTIVE)`:

1. **Use the appropriate lower band.** Important research can use
   `--priority priority`; ordinary work should run at INTERACTIVE or BATCH.
2. **Check your username.** The `max_band` cap is keyed on the verified
   identity the controller sees. If the username in the error message isn't
   what you expect — e.g. it's an email local-part or an SSO id rather than
   your GitHub handle — your identity probably doesn't match the `user_id`
   listed in the cluster config, and you'll land on the default tier.
3. **Request an uplift.** If your work needs INTERACTIVE budget headroom or
   PRIORITY or PRODUCTION, ping [@Helw150](https://github.com/Helw150) to be added to the
   appropriate tier in `marin.yaml`.

## See also

- [`task-states.md`](task-states.md) — how preemption surfaces in task state
- [`OPS.md`](../OPS.md) — operator-side scheduler inspection
