---
topic: norm-preserving-residual
issue: https://github.com/marin-community/marin/issues/8860
description: Evaluate learned norm-preserving residual mixing against the July MoE baseline.
author: kaiyuew
---

# Norm-Preserving Residual: Task Logbook

## Current TL;DR

Gate-1 code is ready on snapshot `959fdb2b2`. The d512 and d768 cells preserve
the canonical July recipe from #6882 and change only the two residual merges in
each layer. No training result exists yet.

## Scope

- Goal: test `sqrt(1 - beta_l / L) * residual + sqrt(beta_l / L) * hidden`.
- Primary metrics: final `eval/paloma/macro_loss`, final-100-step
  `throughput/tokens_per_second`, and effective wall-clock speedup against the
  matching July cell.
- Constraints: canonical July model/data/optimizer/budget/eval; v5p-8; one
  learned scalar per layer; no Gate 2 unless both Gate-1 cells pass.
- Coordinating issue: https://github.com/marin-community/marin/issues/8860

## Baseline

- Date: 2026-07-02 to 2026-07-04.
- Code refs: #6882; branch `july_baseline`; commit
  `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- Baseline numbers:

  | Cell | Layers | Budget | Batch | Steps | Paloma macro | Tokens/s |
  |---|---:|---:|---:|---:|---:|---:|
  | d512 | 6 | 3.82e17 | 16 | 10,980 | 3.5667 | 352,609 |
  | d768 | 8 | 2.81e18 | 32 | 16,875 | 3.2272 | 249,954 |

## Hypothesis Queue

### Active

- `MOE-NPR-001/002`: learned norm-preserving residual mixing improves
  effective wall-clock speed at both July Gate-1 scales. Next test: launch the
  matched d512 and d768 cells from snapshot `959fdb2b2`.

### Blocked

None.

### Falsified / Dead End

None.

### Promoted

None.

## Entry Log

### 2026-09-02 12:16 - Gate-1 implementation validated

- Hypothesis: a depth-scaled, learned residual mixture improves quality enough
  to offset any throughput cost at both July Gate-1 widths.
- Commit Hash: `959fdb2b22b9387c5191aad72d4c6b5aa41b7cd8`.
- Commands:
  - `uv run --with pytest --with pytest-timeout pytest -q experiments/grug/moe_norm_preserving_residual/test_optimizer.py experiments/grug/moe_norm_preserving_residual/test_norm_preserving_residual.py`
  - `uv run --with pytest --with pytest-timeout pytest -q tests/test_grug_variant_contracts.py`
  - `./infra/pre-commit.py --all-files`
- Config: `beta_l = softplus(theta_l)` with `theta_l = 0`; one fp32 scalar per
  layer, shared by the attention and MoE merges. Effective `beta_l` is capped
  at `L * (1 - 1e-6)` before the square root. At initialization, d512 uses
  residual/hidden coefficients `0.940466/0.339889`; d768 uses
  `0.955697/0.294353`.
- Result: 3 focused tests passed; 16 Grug contract tests passed; all repository
  pre-commit checks passed.
- Interpretation: the requested equation is implemented, the square-root
  domain is finite for large positive `theta_l`, and the variant lowers through
  the shared Grug contracts. Training quality remains unknown.
- Next action: submit `MOE-NPR-001-d512` and `MOE-NPR-002-d768` on v5p-8 and
  monitor loss, throughput, and per-layer `train/residual/layer_<l>/beta`.

## Background Research Brief

- Effort: low.
- Stop rule: stop after the canonical July source/result, exact in-repo
  residual sites, and the closest fixed and learned residual-scaling papers
  establish the smallest controlled experiment.
- Date: 2026-09-02.

### Question

Does a per-layer learned convex-in-squared-coefficients residual mixture improve
the canonical July MoE baseline?

### Current Marin Context

The July model is a pre-norm MoE Transformer with two additive residual merges
per layer. d512 has 6 layers and d768 has 8. The experiment shares one
`theta_l` across both merges in layer `l`, adds 6 or 8 scalar parameters, and
leaves the July heuristic and training schedule unchanged.

### Internal Prior Work

- #6882 defines the canonical July source and reports d512/d768 Paloma macro
  losses of `3.5667/3.2272` and throughputs of `352,609/249,954` tokens/s.
- No existing Marin experiment or model path matched the requested
  `sqrt(1-beta/L), sqrt(beta/L)` equation.

### External Prior Art

- [DeepNorm](https://arxiv.org/abs/2203.00555) modifies Transformer residual
  scaling using depth-derived fixed coefficients and paired initialization.
- [ReZero](https://arxiv.org/abs/2003.04887) learns one scalar gate per residual
  connection, initialized to zero, and reports stable training at large depth.
- [LayerScale](https://arxiv.org/abs/2103.17239) learns residual-branch scaling
  in deep image Transformers with small initial coefficients.

These papers support testing residual scaling but do not test this equation,
the `softplus(0)` initialization, shared attention/MoE parameters, or Marin's
6/8-layer MoE regime.

### Negative / Failed Leads

- A positive-only softplus parameterization does not guarantee `beta_l < L`.
  An uncapped implementation can produce NaNs when `theta_l` grows large.
- The exact requested parameterization was not found in the local repository or
  the three closest primary papers.

### Evidence Map

#### Claim: the July experiment is a controlled baseline comparison

- Support:
  - #6882: source commit, recipe, run IDs, budgets, loss, and throughput.
  - Snapshot `959fdb2b2`: variant copied from the canonical July commit.
- Contradictions: current `main` has evolved beyond the July code, so using its
  active MoE variant would not isolate residual mixing.
- Directness to Marin: exact model family, data recipe, hardware, and metrics.
- Confidence: stable baseline; exploratory variant.
- Action: launch only the matched d512/d768 cells.

#### Claim: learned depth-scaled residual mixing may improve optimization

- Support:
  - DeepNorm, ReZero, and LayerScale each report benefits from residual scaling
    or gating in deeper Transformer-like networks.
- Contradictions: their equations, initialization, tasks, and depths differ
  from this experiment.
- Directness to Marin: low to medium.
- Confidence: exploratory.
- Action: require an effective-speedup win at both widths before scaling up.

### Recommended Next Experiments

#### 1. Gate 1 at d512 and d768

- Minimum experiment: one seed-0 run at each canonical July Gate-1 cell.
- Baseline/control: `moe_may_july_baseline_d512` and
  `moe_may_july_baseline_d768` from #6882.
- Expected signal: lower Paloma macro loss with near-neutral throughput; learned
  beta values remain well below `L`.
- Falsifier: effective wall-clock speedup is at or below 1 at either width, a
  run becomes non-finite, or beta saturates at the cap.
- Cost/risk: two v5p-8 training runs; expected hours, with no Gate 2 spend until
  both pass.
- Sources: #6882, DeepNorm, ReZero, LayerScale.

### Hypothesis Queue Update

- Add: `MOE-NPR-001/002` Gate 1.
- Revise: none.
- Falsify / stop: stop the series if either Gate-1 effective speedup is at or
  below 1.
- Promote: Gate 2 only after both Gate-1 cells pass.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| July Baseline #6882 | GitHub issue | https://github.com/marin-community/marin/issues/6882 | exact config and baseline metrics | high | canonical Marin evidence |
| July source | Marin code | commit `52d8a9eb8` | exact architecture and launch recipe | high | branch copied directly |
| DeepNorm | paper | https://arxiv.org/abs/2203.00555 | depth-dependent residual scaling | medium | different equation and depth |
| ReZero | paper | https://arxiv.org/abs/2003.04887 | learned scalar residual gates | medium | zero initialization |
| LayerScale | paper | https://arxiv.org/abs/2103.17239 | learned residual-branch scaling | medium | vision models, per-channel gate |

### Handoff

- Suggested issue `Prior work` block: residual scaling has positive precedent in
  DeepNorm, ReZero, and LayerScale, but this exact equation and shallow MoE
  regime are untested.
- Suggested logbook entry: launch exact d512/d768 Gate 1 from `959fdb2b2`.
- Open questions: whether one shared scalar per layer is better than separate
  attention/MoE scalars; whether `softplus(0)` overweights the hidden branch.
- Stop reason: the remaining uncertainty requires training evidence.
