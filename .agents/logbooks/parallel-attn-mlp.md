# Parallel attn/MLP block layout: Research Logbook

Experiment ID prefix: `MOE-PAM`

Linked issue: marin-community/marin#5932 (sub-issue of #5358).

## Scope
- Goal: Determine whether running attention and MLP in parallel against the
  residual stream (instead of the baseline serial wiring) improves the
  effective speedup of the May MoE Recipe.
- Primary metric: `eval/paloma/macro_loss` (effective speedup per
  `experiments/grug/moe/agent.md` §Effective speedup calculation).
- Secondary metric: `throughput/tokens_per_second`.
- Constraints: keep all recipe knobs (XSA, GatedNorm, PKO, QB routing, etc.)
  identical to the May Recipe baseline. Only the per-block layout changes.

## Variants (gate 1)
| tag | parallel_mode | parallel_second_half_only |
| --- | --- | --- |
| `parallel-all` | `PARALLEL` | False |
| `parallel-half` | `PARALLEL` | True |
| `parallel-merged` | `PARALLEL_MERGED_NORM` | False |

Each runs at both gate-1 scales (d512 / 2.19e17, d768 / 1.70e18) — 6 child
training jobs total.

## Baseline
- Branch baseline (per user override): `grug_moe_may_recipe` (not `main`).
- Code refs: `experiments/grug/moe/model.py` (`Block`), `direct_launch.py`.
- Baseline numbers: pull from latest `grug-moe-direct-d{512,768}-*` runs on
  `marin_moe`.

## Kickoff
- Branch: `parallel_attn_mlp` (off `grug_moe_may_recipe`).
- Submission: single `iris job run --no-wait --zone us-east5-a` invoking
  `python -m experiments.grug.moe.parallel_attn_mlp_sweep`. The coordinator
  fans out the six training jobs via `ThreadPoolExecutor` and blocks on all
  of them.
- Stop criteria for this kickoff: per the updated `agent.md`, stop after the
  Iris coordinator is submitted; gate-1 analysis and any follow-up runs
  require explicit user direction.

## Experiment Log
### Kickoff
- Hypothesis: parallel attn/MLP loses a small amount of loss quality but
  gains enough throughput to be net positive on effective speedup; merged
  norm narrows the loss gap further by saving parameters.
- Command: see Kickoff section.
- Result: TBD — gate 1 in flight.
