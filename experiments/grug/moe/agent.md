# Agent Guide: experiments/grug/moe

## Autonomy

This workflow is designed to run end-to-end without human confirmation. The
agent is authorized to:

- Create branches, commit, and push without asking
- Create GitHub experiment issues and post comments
- Submit Iris jobs and kill only jobs submitted by self
- Run experiments through both gates autonomously

Do not stop to ask for confirmation at any step. If something fails, diagnose
and retry or report the failure — do not block waiting for input.

## Objective

Determine whether a proposed change outperforms the baseline. Baseline results
are in `experiments/grug/moe/README.md` — compare against the table there.

**Metrics (from wandb):**
- `eval/paloma/macro_loss` (final value)
- `throughput/tokens_per_second` (averaged over the last 100 steps)
- `throughput/total_tokens` (final value)

**Baseline scaling law** (L∞ pinned at 1.6):

```
loss(C) = 1.6 + 95.18 · C^(-0.0941)
```

### Gate 1: effective speedup at two small scales

Run the variant at `d512` (2.19e17 FLOPs) and `d768` (1.70e18 FLOPs).

For each scale, compute the **effective speedup** at a fixed macro_loss target
(use the baseline's final macro_loss at that scale as the target). The variant
passes gate 1 if it shows an effective speedup at **both** scales.

### Gate 2: scaling law projection

Run the variant at the two larger scales: `d1024` (9.00e18) and `d1280`
(2.83e19). Combine with the gate 1 results (d512, d768) for four total points.

The variant passes gate 2 if:
1. It shows an effective speedup at **all four** scales.
2. Fit a new scaling law `loss(C) = 1.6 + A · C^(-alpha)` (asymptote pinned
   at 1.6) on the variant's four optima. Project to 1e21 and 1e23 FLOPs.
   The variant's projected loss must be lower than the baseline's at both
   budgets (baseline: 2.606 at 1e21, 2.252 at 1e23).

### Effective speedup calculation

Given baseline and variant results at the same compute budget:

```python
import numpy as np

def effective_speedup(baseline_loss, baseline_tps, variant_loss, variant_tps, budget):
    """Compute effective wall-clock speedup at the baseline's final loss.

    Returns > 1 if the variant is faster to reach the same loss.
    """
    target_loss = baseline_loss

    # Invert scaling law: C needed to reach target_loss
    # loss(C) = 1.6 + 95.18 * C^(-0.0941) => C = (95.18 / (loss - 1.6))^(1/0.0941)
    C_baseline = (95.18 / (target_loss - 1.6)) ** (1 / 0.0941)

    # The variant achieves variant_loss at the same budget. Fit the same
    # power-law shape shifted vertically: the variant's curve passes through
    # (budget, variant_loss) with the same exponent.
    # variant_loss = 1.6 + A_var * budget^(-0.0941)
    A_var = (variant_loss - 1.6) / budget ** (-0.0941)
    C_variant = (A_var / (target_loss - 1.6)) ** (1 / 0.0941)

    # Wall-clock = compute / throughput
    wall_baseline = C_baseline / baseline_tps
    wall_variant = C_variant / variant_tps
    return wall_baseline / wall_variant
```

### Example: effective speedup at a fixed loss target

Suppose at d768 / 1.70e18 FLOPs:
- **Baseline**: macro_loss = 3.43, tok/s = 200,000
- **Variant A**: macro_loss = 3.40, tok/s = 180,000 (better loss, 10% slower)

To reach macro_loss = 3.43 (the baseline's final loss), how much compute does
each method need?

```python
# Invert the scaling law: C(L) = (95.18 / (L - 1.6))^(1/0.0941)
target_loss = 3.43
C_baseline = (95.18 / (target_loss - 1.6)) ** (1 / 0.0941)  # = 1.70e18

# Variant A reaches 3.40 at 1.70e18 FLOPs. It would have hit 3.43 at some
# smaller C. Assume the same scaling law shape, shifted by the improvement:
# variant_loss = 1.6 + A_var * budget^(-0.0941)
A_var = (3.40 - 1.6) / (1.70e18) ** (-0.0941)
C_variant = (A_var / (target_loss - 1.6)) ** (1 / 0.0941)
```

But compute alone isn't wall-clock time — variant A is 10% slower per step.
The wall-clock to reach the target is `C / tok_per_sec`:

```python
wall_baseline = C_baseline / 200_000
wall_variant = C_variant / 180_000
speedup = wall_baseline / wall_variant
```

If `speedup > 1`, variant A reaches the target loss faster in real time despite
being slower per step. Report this as "X% effective speedup (or slowdown) at
macro_loss = Y". This is the key number for deciding whether to promote a
change.

## Implementation

Most promotable changes will land in one of three files:

- `model.py` — architecture tweaks (routing, norms, attention, activation functions, expert layout, etc.).
- `heuristic.py` — scaling heuristics (LR formula coefficients, depth/width formula, GQA ratio, per-batch-size epsilon/beta2 scaling).
- `optimizer.py` — optimizer internals (AdamH components, parameter-group partitioning, per-group learning rates, weight decay).

## Documentation & GitHub Issues

Create a new branch for each experiment issue. Branch off `main`.

Follow `.agents/skills/agent-research/SKILL.md` for all documentation, logbooks,
W&B tracking, and GitHub experiment issue management tied to work in this
directory. Pay attention to this file carefully.

Experiment issues should be titled `Agent MoE Experiment: [description]`.
Include the exact prompt from the user that initiated the experiment in the
issue body.

After creating the issue, **add it as a sub-issue of #4281** (April 2026 MoE
scaling tracking issue) using the GitHub GraphQL API. This is required — do not skip it. First get the node IDs, then
call `addSubIssue`:

```bash
# 1. Get node IDs for the parent and the new issue
gh api graphql -f query='
query {
  repository(owner: "marin-community", name: "marin") {
    parent: issue(number: 4281) { id }
    child: issue(number: <NEW_ISSUE_NUMBER>) { id }
  }
}'

# 2. Add the sub-issue relationship
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
- `WANDB_API_KEY` set in the environment
- `gcloud auth login` and `gcloud auth application-default login`

## Job Submission

Jobs in this directory are submitted to **Iris** on a **v5p-8**.

### Submission command

```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.launch
```

Swap the module path (`experiments.grug.moe.launch`) for whichever launch
script in this directory you are running.

### Monitoring

Runs may take time to find a TPU, and 5–10 minutes to start once scheduled.
After confirming the run is progressing on wandb, jobs typically take over an
hour to complete. Sleep at reasonable intervals (e.g. 15 minutes) before
checking status — do not poll in a tight loop.

Reconnect to logs:
```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job logs -f JOB_ID
```

List your jobs:
```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job list | grep "$(whoami)"
```

Check runs in wandb (match `<PROJECT>` and `<PREFIX>` to `launch.py`):
```python
import wandb
api = wandb.Api()
runs = api.runs('marin-community/<PROJECT>', filters={'displayName': {'$regex': '^<PREFIX>'}}, order='-created_at')
for r in runs:
    print(f'{r.name:<50} state={r.state:<10} step={r.summary.get("global_step", "n/a")}')
```
