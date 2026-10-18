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

The ladder and gates below are for model and optimizer changes.
For data changes, use the [Hero-mix track](README.md#add-a-dataset) or the
[quality-classifier track](README.md#improve-a-quality-classifier).
[Run a data-track command](README.md#run-a-data-track-command) gives their Iris command, and
[Compare data-track runs](README.md#compare-data-track-runs) gives their gate. All tracks use the same metric.

Determine whether a proposed change outperforms the baseline. The baselines are
the dense and MoE runs recorded in `experiments/grug/fast_track/README.md` —
compare against the table there (the W&B runs linked below the table).

**Metrics (from wandb, project `marin-community/marin_moe`):**
- `eval/paloma/macro_loss` (final value) — **the** primary quality metric for all tracks (lower is better).
  Report and compare in Paloma macro loss, not bpb. Uncheatable macro loss
  (`eval/uncheatable_eval/macro_loss`) and per-dataset Paloma loss are guardrails.
- `throughput/tokens_per_second` (averaged over the last 100 steps)
- `run.state` must be `finished` before pulling final metrics

The ladder is data-matched by default (`--match data`): a variant trains on the
same tokens as its baseline at each size, so macro loss is an equal-data comparison. Use
`--match compute` for an equal-FLOPs comparison instead.

### Gate 1: two small scales

Run the variant at `d512` and `d768` (both dense and MoE, as the change
requires). The variant passes gate 1 if its Paloma macro loss beats the baseline at
**both** scales.

### Gate 2: full ladder

Run the variant at `d1024` and `d1280`, combining with gate 1 for four points.
The variant passes gate 2 if it beats the baseline Paloma macro loss at **all four**
scales. Once four points exist, fit `L(C) = L_inf + A · C^(-alpha)` on the
variant's optima and on the baseline's, and compare the projections — but fit
`L_inf`/`alpha` from these runs; do not reuse constants from other variants
(different vocab and metric).

### Effective speedup (throughput-aware refinement)

When a variant trades quality for throughput (or vice versa), compare wall-clock time to reach a
fixed macro loss rather than loss alone. The one rule: **`budget`/`C` and the throughput must be the
same currency** — both tokens, or both FLOPs — or the wall-clock ratio silently gains a spurious
`flops_per_token` factor.

For the default **data-matched** ladder, work in tokens: `budget` is the shared token count both runs
trained (data-match guarantees they match, so one `budget` applies to both) and `*_tps` is
`throughput/tokens_per_second`. Then `compute / throughput` is genuinely seconds and this is exact
(fit `alpha`/`L_inf` as loss-vs-tokens):

```python
def effective_speedup(baseline_loss, baseline_tps, variant_loss, variant_tps, budget, *, L_inf, alpha):
    """Wall-clock speedup of the variant over the baseline (>1 = variant reaches its loss faster).
    DATA-matched: `budget` = shared token count; `*_tps` = tokens/sec; alpha/L_inf fit as loss-vs-tokens."""
    A_bl = (baseline_loss - L_inf) * budget ** alpha           # fit A through the baseline point
    C_needed = (A_bl / (variant_loss - L_inf)) ** (1 / alpha)  # tokens the baseline needs for variant_loss
    return (C_needed / baseline_tps) / (budget / variant_tps)
```

For **`--match compute`** (or any architecture change that alters FLOPs/token), tokens/sec is the
wrong denominator: `budget`/`C` are FLOPs but the two runs process *different* token counts, so
dividing FLOPs by tokens/sec drops the `flops_per_token` ratio — a +18% FLOPs/token variant that is
genuinely equal gets reported as ~0.85. Keep `budget`/`C` in FLOPs but pass throughput as **FLOP/s**
(`flops_per_second = tokens_per_second * flops_per_token`, with training
`flops_per_token = _compute_flops(model_config=cfg)[0] / SEQ_LEN`, which is three times the forward-only
summary value `throughput/flops_per_token_analytic`)
for `*_tps`; then `FLOPs / (FLOPs/sec)` is seconds and the FLOPs/token difference cancels correctly.

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

Experiment issues should be titled `[fast-track] <description>` and include the exact prompt from
the user that initiated the experiment. Label them `fast-track` and `agent-generated`.

## Authentication

Assume the user has already set `WANDB_API_KEY` in the environment before job
submission. Iris derives `MARIN_PREFIX` per-cluster (region metadata) — it is not
a shell variable. No `gcloud` / TPU auth is needed; this variant runs on H100.

## Job Submission

Training jobs run on **Iris**, with one 8×H100 node per run. `fast-track` trains on the frozen
Hero cache in `s3://marin-us-east-02a`. Without `--submit`, it prints the plan.

```bash
uv run fast-track --submit --run-id <name> --size <size> [--dense] --version <v>
```

`--size` and `--run-id` are required. Use `--version dev` for scratch runs and `--version YYYY.MM.DD`
for coordinated runs. Pass `--save-checkpoints` to keep the final model.
[Launch commands](README.md#launch-commands) gives the other flags and defaults.

## Monitoring

Runs take a few minutes to schedule and compile, then run from minutes (small
dense) to about 10 hr (d1280 MoE). Sleep at reasonable intervals (e.g. 15 minutes)
before checking status — do not poll in a tight loop.

List your jobs:
```bash
uv run iris --cluster marin job list --prefix "/$(whoami)"
```

Reconnect to logs:
```bash
uv run iris --cluster marin job logs -f JOB_ID
```

Check runs in wandb (match `<PREFIX>` to the `--run-id` pattern, e.g. `fasttrack-`):
```python
import wandb
api = wandb.Api()
runs = api.runs('marin-community/marin_moe', filters={'displayName': {'$regex': '^<PREFIX>'}}, order='-created_at')
for r in runs:
    print(f'{r.name:<50} state={r.state:<10} loss={r.summary.get("eval/paloma/macro_loss", "n/a")}')
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
- Paloma macro loss (`eval/paloma/macro_loss`) — the primary metric. Also record Uncheatable macro loss
  (`eval/uncheatable_eval/macro_loss`) as a guardrail.
- MFU, throughput (`throughput/tokens_per_second`), wall-clock runtime

**Provenance**
- git commit SHA the run executed at, and a permalink to `launch.py` pinned at that commit
- W&B run URL
- permanent (final) checkpoint S3 path, when checkpoints were saved
