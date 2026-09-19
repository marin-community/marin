# Agent Guide: experiments/grug/fast_track

## Autonomy

This workflow is designed to run end-to-end without human confirmation. The
agent is authorized to:

- Create branches, commit, and push without asking
- Create GitHub experiment issues and post comments
- Submit Iris jobs and kill only jobs submitted by self
- Run the ladder and compare against the baseline autonomously

Do not stop to ask for confirmation at any step. If something fails, diagnose
and retry or report the failure — do not block waiting for input.

## Objective

Determine whether a proposed change outperforms the baseline. The baselines are
the dense and MoE runs recorded in `experiments/grug/fast_track/README.md` —
compare against the table there (the `fasttrack-baseline`-tagged W&B runs).

**Metrics (from wandb, project `marin-community/marin_moe`):**
- `eval/paloma/macro_bpb` (final value) — primary quality metric (lower is better)
- `eval/uncheatable_eval/macro_bpb` (final value) — secondary quality metric
- `throughput/tokens_per_second` (averaged over the last 100 steps)
- `run.state` must be `finished` before pulling final metrics

The ladder is data-matched by default (`--match data`): a variant trains on the
same tokens as its baseline at each size, so bpb is an equal-data comparison. Use
`--match compute` for an equal-FLOPs comparison instead.

### Gate 1: two small scales

Run the variant at `d512` and `d768` (both dense and MoE, as the change
requires). The variant passes gate 1 if its Paloma bpb beats the baseline at
**both** scales.

### Gate 2: full ladder

Run the variant at `d1024` and `d1280`, combining with gate 1 for four points.
The variant passes gate 2 if it beats the baseline Paloma bpb at **all four**
scales. Once four points exist, fit `bpb(C) = L_inf + A · C^(-alpha)` on the
variant's optima and on the baseline's, and compare the projections — but fit
`L_inf`/`alpha` from these runs; do not reuse constants from other variants
(different vocab and metric).

### Effective speedup (throughput-aware refinement)

When a variant trades quality for throughput (or vice versa), compare wall-clock
time to reach a fixed bpb rather than bpb alone. Given a fitted exponent `alpha`
and asymptote `L_inf`:

```python
def effective_speedup(baseline_bpb, baseline_tps, variant_bpb, variant_tps, budget, *, L_inf, alpha):
    """Wall-clock speedup of the variant over the baseline (>1 = variant reaches its bpb faster)."""
    A_bl = (baseline_bpb - L_inf) * budget ** alpha          # fit A through the baseline point
    C_needed = (A_bl / (variant_bpb - L_inf)) ** (1 / alpha)  # compute baseline needs to match variant bpb
    return (C_needed / baseline_tps) / (budget / variant_tps)
```

## Implementation

Most promotable changes land in one of these files:

- `model.py` — architecture: attention, GatedNorm, SConv, QB-routed MoE, activations, expert layout.
- `heuristic.py` — scaling heuristics (LR / beta2 / epsilon formulas, depth/width, GQA ratio).
- `optimizer.py` — optimizer config: LR groups, hyperball step, weight decay.
- `grugmuon_stacked.py` — Newton-Schulz orthogonalization (the Muon direction).
- `adamh.py` — AdamH scale transform (the `adamh` LR group).
- `router_metrics.py` — routing-stats telemetry (logging-only; never feeds the loss).
- `launch.py` — ladder rungs, budget resolution (`--match`), Iris/W&B wiring.

## Documentation & GitHub Issues

Create a new branch for each experiment issue, off `main`. Follow
`.agents/skills/research/SKILL.md` for explicitly requested research programs and
their existing issue, W&B, or session records.

Experiment issues should be titled `Agent fast_track Experiment: [description]`
and include the exact prompt from the user that initiated the experiment.

## Authentication

Assume the user has already set `WANDB_API_KEY` in the environment before job
submission. Iris derives `MARIN_PREFIX` per-cluster (region metadata) — it is not
a shell variable. No `gcloud` / TPU auth is needed; this variant runs on H100.

## Job Submission

Jobs run on **Iris**, one 8×H100 node per run. Two H100 clusters are available:
`cw-rno2a` and `cw-us-east-02a`. Use the `irun` wrapper from the README (target it
with `IRIS_CLUSTER=<cluster>`), or submit directly:

```bash
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster "${IRIS_CLUSTER:-cw-rno2a}" --priority interactive --job-name "<name>-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT marin_moe \
  -- python -m experiments.grug.fast_track.launch --run-id <name> --size <size> [--dense] --version <v> --run
```

`--size` (d512/d768/d1024/d1280) and `--run-id` are required; `--dense` selects
the dense baseline. The step budget derives from the variant baseline (`--match`,
default `data`) unless `--num-steps` is given; `--batch-size` defaults to the
rung's baseline batch. `--version` sets the checkpoint version (use `dev` for
scratch, a calendar `YYYY.MM.DD` for coordinated runs); `--run` submits (without
it the lowered plan is printed and nothing runs). Omit `--no-save-checkpoints` for
runs whose final model you want to keep.

## Monitoring

Runs take a few minutes to schedule and compile, then run from minutes (small
dense) to ~12 hr (d1280 MoE). Sleep at reasonable intervals (e.g. 15 minutes)
before checking status — do not poll in a tight loop.

List your jobs:
```bash
uv run iris --config lib/iris/config/marin.yaml job list | grep "$(whoami)"
```

Reconnect to logs:
```bash
uv run iris --config lib/iris/config/marin.yaml job logs -f JOB_ID
```

Check runs in wandb (match `<PREFIX>` to the `--run-id` pattern, e.g. `fasttrack-`):
```python
import wandb
api = wandb.Api()
runs = api.runs('marin-community/marin_moe', filters={'displayName': {'$regex': '^<PREFIX>'}}, order='-created_at')
for r in runs:
    print(f'{r.name:<50} state={r.state:<10} bpb={r.summary.get("eval/paloma/macro_bpb", "n/a")}')
```

## Final metrics to log

When a run reaches `state=finished`, record one record per run (in the experiment
issue; for baselines, also the README table). Capture:

**Description & config**
- run description (1–3 lines): what the run is / what changed vs the baseline
- size + variant (e.g. `d768` / moe), match mode (`data` / `compute`)
- batch size, sequence length (`SEQ_LEN` = 4096), num steps
- tokenizer + vocab size (`hero-bpe-v16384` / 16384), hardware (8×H100, 1 node)

**Scale**
- num tokens trained (= batch × steps × seq_len)
- num parameters (total)
- num active parameters (per `_active_params`)
- num lm_head parameters (= hidden_dim × vocab_size)
- total training FLOPs excluding the lm_head (≈ 6 · active_params · tokens)
- total training FLOPs including the lm_head (adds 6 · lm_head_params · tokens;
  a large share at small hidden_dim / 16k vocab)

**Results** (from wandb — final value, or last-100-step average for throughput)
- Paloma macro loss and macro bpb (`eval/paloma/macro_bpb`)
- uncheatable macro bpb (`eval/uncheatable_eval/macro_bpb`)
- MFU, throughput (`throughput/tokens_per_second`), wall-clock runtime

**Provenance**
- git commit SHA the run executed at, and a link to `launch.py` on that branch @ that commit
- W&B run URL
- permanent (final) checkpoint S3 path, when checkpoints were saved
