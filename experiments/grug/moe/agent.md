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

Determine whether a proposed change outperforms the baseline.

**Gate 1 comparison points:** `d512` at `2.19e17` FLOPs, and `d768` at
`1.70e18` FLOPs (compute-optimal for each dim).

**Gate 2 (only if Gate 1 is promising):** `d1024` at `9.00e18` FLOPs, and
`d1280` at `2.83e19` FLOPs (compute-optimal for each dim).

**Metrics (from wandb):**
- `eval/paloma/c4_en/bpb` (final value)
- `throughput/tokens_per_second` (averaged over the last 100 steps)
- `throughput/total_tokens` (final value)

The key quantity is `total_tokens / tokens_per_second` — wall-clock time to
reach a given loss. A change that improves bpb but slows throughput is only
worthwhile if the net wall-clock to match baseline loss is lower.

A change will typically move *both* metrics. To decide whether the tradeoff is
worthwhile, use the baseline compute-loss formula:

```
loss(C) = 1.6 + 95.18 · C^(-0.0941)
```

Running slower corresponds to a smaller effective `C` (fewer FLOPs per unit
wall-clock). Compare the proposed (bpb, tok/s) pair against the baseline by
converting throughput changes into an effective-`C` shift and checking whether
the measured bpb beats what the baseline curve would predict at that `C`.

## Implementation

Most promotable changes will land in one of three files:

- `model.py` — architecture tweaks (routing, norms, attention, activation functions, expert layout, etc.).
- `heuristic.py` — scaling heuristics (LR formula coefficients, depth/width formula, GQA ratio, per-batch-size epsilon/beta2 scaling).
- `optimizer.py` — optimizer internals (AdamH components, parameter-group partitioning, per-group learning rates, weight decay).

## Documentation & GitHub Issues

Create a new branch for each experiment issue. Branch off `grug_moe_heuristic`
(the current best recipe branch).

Follow `.agents/skills/agent-research/SKILL.md` for all documentation, logbooks,
W&B tracking, and GitHub experiment issue management tied to work in this
directory. Pay attention to this file carefully.

Experiment issues should be titled `Agent MoE Experiment: [description]` and
reference #4281 as the parent tracking issue.

## Authentication

Assume the user has already completed these before job submission:
- `WANDB_API_KEY` set in the environment
- `gcloud auth login` and `gcloud auth application-default login`

## Job Submission

Jobs in this directory are submitted to **Iris** on a **v5p-8**.

**Concurrency limit:** Each GitHub experiment issue should have **at most 2
concurrent runs** active on the cluster at a time.

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
