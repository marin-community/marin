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
preempting other researchers' running jobs.

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

### Max-band caps and unlisted users

Each user who is listed in the cluster config's `user_budgets` tier list has a
`budget_limit` and `max_band` recorded in the `user_budgets` table.
**Unlisted** users don't get a row; they inherit `UserBudgetDefaults` at read
time — a small INTERACTIVE budget that silently falls through to BATCH once
exceeded. Submissions at a higher band than `max_band` are **rejected** (not
downgraded) with `PERMISSION_DENIED`. The tiers reconciled from the cluster
config at startup are:

- Admins — `PRODUCTION` (and everything below), large budget.
- Listed researchers — `INTERACTIVE` (plus `BATCH`), large budget.
- Everyone else (including new/unlisted users) — `INTERACTIVE` with a small
  default budget; jobs run INTERACTIVE while within budget and degrade to
  BATCH once exceeded. PRODUCTION submissions are rejected.

If you hit `User <name> cannot submit PRODUCTION jobs (max band: INTERACTIVE)`:

1. **Retry without `--priority production`.** Most research workloads do not
   need PRODUCTION and run fine at INTERACTIVE (or opportunistically at BATCH).
2. **Check your username.** The `max_band` cap is keyed on the verified
   identity the controller sees. If the username in the error message isn't
   what you expect — e.g. it's an email local-part or an SSO id rather than
   your GitHub handle — your identity probably doesn't match the `user_id`
   listed in the cluster config, and you'll land on the default tier.
3. **Request an uplift.** If your work needs INTERACTIVE budget headroom or
   PRODUCTION, ping [@Helw150](https://github.com/Helw150) to be added to the
   appropriate tier in `marin.yaml`.

## See also

- [`task-states.md`](task-states.md) — how preemption surfaces in task state
- [`OPS.md`](../OPS.md) — operator-side scheduler inspection
