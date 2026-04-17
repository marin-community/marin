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
- `run.state` must be `finished` before pulling final metrics

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

The question: how much faster (in wall-clock time) does the variant reach
its final loss compared to how long the baseline would need to reach that
same loss?

The scaling law `loss(C) = 1.6 + A · C^(-0.0941)` has a fixed exponent but
the coefficient A varies. We recenter A to pass through the baseline's
actual (budget, loss) point, then invert to find how much compute the
baseline would need to match the variant's loss. Finally, compare wall-clock
times.

```python
def effective_speedup(baseline_loss, baseline_tps, variant_loss, variant_tps, budget):
    """Wall-clock speedup of the variant over the baseline.

    Returns > 1 if the variant reaches its loss faster in real time.
    """
    L_inf = 1.6
    alpha = 0.0941

    # 1. Fit the scaling law coefficient through the baseline's data point.
    #    loss = L_inf + A_bl * C^(-alpha)  =>  A_bl = (loss - L_inf) * C^alpha
    A_bl = (baseline_loss - L_inf) * budget ** alpha

    # 2. How much compute would the baseline need to reach variant's loss?
    C_needed = (A_bl / (variant_loss - L_inf)) ** (1 / alpha)

    # 3. Compare wall-clock times:
    #    baseline_time = C_needed / baseline_tps
    #    variant_time  = budget / variant_tps
    return (C_needed / baseline_tps) / (budget / variant_tps)
```

### Example

Suppose at d512 / 2.19e17 FLOPs:
- **Baseline**: macro_loss = 3.81, tok/s = 405,000
- **Variant**: macro_loss = 3.79, tok/s = 380,000 (better loss, 6% slower)

```python
# 1. Fit A through baseline point
A_bl = (3.81 - 1.6) * (2.19e17) ** 0.0941  # ≈ 94.7

# 2. Compute baseline needs to reach 3.79
C_needed = (94.7 / (3.79 - 1.6)) ** (1 / 0.0941)  # > 2.19e17

# 3. Wall-clock comparison
baseline_time = C_needed / 405_000
variant_time  = 2.19e17 / 380_000
speedup = baseline_time / variant_time
```

If the quality improvement (needing more baseline compute to match) outweighs
the throughput cost (variant taking longer per FLOP), speedup > 1.

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

Check runs in wandb (match `<PREFIX>` to the run_id pattern in `launch.py`):
```python
import wandb
api = wandb.Api()
runs = api.runs('marin-community/marin_moe', filters={'displayName': {'$regex': '^<PREFIX>'}}, order='-created_at')
for r in runs:
    print(f'{r.name:<50} state={r.state:<10} step={r.summary.get("global_step", "n/a")}')
```
