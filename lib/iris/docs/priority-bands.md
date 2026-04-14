# Priority Bands

Iris ranks pending tasks by **priority band** before per-user fairness. Three
bands exist (defined in
[`job.proto`](../src/iris/rpc/job.proto)): `PRODUCTION`, `INTERACTIVE`, and
`BATCH`. Choose the right band for what you are running — picking the wrong one
either delays your work or disrupts other people's.

| Band | Selected via | Behavior |
|---|---|---|
| `PRODUCTION` | `--priority production` | Always scheduled before lower bands. Can preempt INTERACTIVE/BATCH. Never downgraded by the budget system. |
| `INTERACTIVE` | default (or `--priority interactive`) | Normal work. Yields to PRODUCTION; preempts BATCH. |
| `BATCH` | `--priority batch` | Opportunistic. Yields to anything else. Safe to launch in bulk. |

## When to use each band

### PRODUCTION

Use **only** for work that has been discussed at a weekly meeting or directly
with the PI (Percy) as high priority for the whole org and blocked on compute.
For Stanford folks: equivalent to `sphinx` queues on the NLP cluster.

Submitting to PRODUCTION without a prior conversation is antisocial — you are
preempting other researchers' running jobs. The CLI logs a warning when you
pass `--priority production` to remind you of this contract.

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

## How band selection interacts with budgets

Per-user budget tracking lives in
[`controller/budget.py`](../src/iris/cluster/controller/budget.py). When a user
exceeds their budget, INTERACTIVE submissions are silently downgraded to BATCH.
PRODUCTION is exempt — another reason to reserve it for vetted work.

## See also

- [`task-states.md`](task-states.md) — how preemption surfaces in task state
- [`OPS.md`](../OPS.md) — operator-side scheduler inspection
