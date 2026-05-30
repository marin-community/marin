---
name: grug-moe-experiment
description: Run and evaluate Grug MoE experiment variants in experiments/grug/moe using Marin infrastructure. Use when proposing, implementing, launching, monitoring, or summarizing a Grug/MoE model change that must pass the d512/d768 effective-speedup gate, the d1024/d1280 scaling-law gate, or the associated GitHub issue, W&B, and Iris workflow.
---

# Skill: Grug MoE Experiment

Use this skill with `.agents/skills/change-grug/SKILL.md` for Grug code
changes and `.agents/skills/run-research/SKILL.md` for logbooks, W&B tracking,
snapshot tags, and experiment issue updates.

## Autonomy

Run the workflow end-to-end without human confirmation once the user asks for a
Grug MoE experiment. You may:

- Create branches, commits, pushes, GitHub experiment issues, and comments.
- Submit Iris jobs and kill only jobs submitted by you.
- Run both evaluation gates and report failures with evidence.

Do not stop to ask for confirmation during normal experiment execution. If a
step fails, diagnose and retry when reasonable; otherwise report the concrete
blocker in the GitHub issue or research logbook.

## Objective

Determine whether a proposed MoE change outperforms the baseline in
`experiments/grug/moe/README.md`. Compare against the baseline table there.

Pull final metrics from W&B only after `run.state` is `finished`:

- `eval/paloma/macro_loss` final value
- `throughput/tokens_per_second` averaged over the last 100 steps
- `throughput/total_tokens` final value

Use the baseline scaling law with asymptote pinned at 1.6:

```text
loss(C) = 1.6 + 95.18 * C^(-0.0941)
```

## Evaluation Gates

### Gate 1: Effective Speedup

Run the variant at:

- `d512` with `2.19e17` FLOPs
- `d768` with `1.70e18` FLOPs

At each scale, compare the variant's final loss to the baseline final
`macro_loss` at that scale. The variant passes gate 1 only if effective speedup
is greater than 1 at both scales.

Use this calculation:

```python
def effective_speedup(baseline_loss, baseline_tps, variant_loss, variant_tps, budget):
    """Return wall-clock speedup of variant over baseline."""
    loss_inf = 1.6
    alpha = 0.0941

    baseline_a = (baseline_loss - loss_inf) * budget**alpha
    baseline_compute_needed = (baseline_a / (variant_loss - loss_inf)) ** (1 / alpha)

    baseline_time = baseline_compute_needed / baseline_tps
    variant_time = budget / variant_tps
    return baseline_time / variant_time
```

### Gate 2: Scaling Projection

Run the two larger scales:

- `d1024` with `9.00e18` FLOPs
- `d1280` with `2.83e19` FLOPs

Combine those results with `d512` and `d768`. The variant passes gate 2 only if:

- Effective speedup is greater than 1 at all four scales.
- A fit of `loss(C) = 1.6 + A * C^(-alpha)` on the four variant optima projects
  lower loss than the baseline at both budgets:
  - `1e21`: baseline loss `2.606`
  - `1e23`: baseline loss `2.252`

## Implementation Surface

Keep each experiment scoped. Most promotable Grug MoE changes land in one of:

- `experiments/grug/moe/model.py` for architecture changes such as routing,
  norms, attention, activation functions, expert layout, and expert parallelism.
- `experiments/grug/moe/heuristic.py` for scaling heuristics such as learning
  rate formula coefficients, depth/width formulae, GQA ratio, epsilon, or beta2.
- `experiments/grug/moe/optimizer.py` for AdamH internals, parameter group
  partitioning, per-group learning rates, weight decay, or optimizer variants.

Prefer copy-local experiment code over new reusable framework APIs unless the
same abstraction is already needed in multiple places.

## GitHub And Research Tracking

Create a branch for each experiment issue from `main`. Title experiment issues:

```text
Agent MoE Experiment: <description>
```

Include the exact user prompt that initiated the experiment in the issue body.
Follow `run-research` for the issue body, research logbook, W&B links, snapshot
tags, and update cadence.

After creating an experiment issue, add it as a sub-issue of #4281 with GitHub
GraphQL. This is required.

```bash
gh api graphql -f query='
query {
  repository(owner: "marin-community", name: "marin") {
    parent: issue(number: 4281) { id }
    child: issue(number: <NEW_ISSUE_NUMBER>) { id }
  }
}'
```

Then call `addSubIssue` with the returned IDs:

```bash
gh api graphql -f query='
mutation {
  addSubIssue(input: {issueId: "<PARENT_ID>", subIssueId: "<CHILD_ID>"}) {
    issue { number }
    subIssue { number }
  }
}'
```

## Authentication

Assume the user has already completed these before job submission:

- `WANDB_API_KEY` is set in the environment.
- `gcloud auth login` and `gcloud auth application-default login` have run.

If these prerequisites are missing, record the blocker and the exact failed
check. Do not attempt to work around missing credentials with new secrets.

## Job Submission

Submit jobs to Iris on a `v5p-8` reservation:

```bash
.venv/bin/iris --cluster=marin job run \
  --no-wait \
  --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.launch
```

Swap the module path for the specific launch script when using another module
in `experiments/grug/moe`.

## Monitoring

Runs may take time to find a TPU and 5-10 minutes to start after scheduling.
After confirming progress in W&B, wait about 15 minutes between status checks.
Do not poll tightly.

Reconnect to logs:

```bash
.venv/bin/iris --cluster=marin job logs -f JOB_ID
```

List your jobs:

```bash
.venv/bin/iris --cluster=marin job list | grep "$(whoami)"
```

Check W&B runs by matching the launch prefix:

```python
import wandb

api = wandb.Api()
runs = api.runs(
    "marin-community/marin_moe",
    filters={"displayName": {"$regex": "^<PREFIX>"}},
    order="-created_at",
)
for run in runs:
    step = run.summary.get("global_step", "n/a")
    print(f"{run.name:<50} state={run.state:<10} step={step}")
```

## Reporting

For each gate result, report:

- Commit SHA and launch command.
- Hardware reservation and run IDs.
- Baseline loss/TPS and variant loss/TPS at each scale.
- Effective speedup at each completed scale.
- Gate decision and next action.

For gate 2, include the fitted `A`, fitted `alpha`, and projected losses at
`1e21` and `1e23` FLOPs.
