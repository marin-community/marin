# Debugging log for DS-RE-CEQ two-phase many-domain diagnostics

Investigate why DS-RE-CEQ fits the 241-run two-phase-many sweep poorly and create diagnostics that expose whether the failure is on the frontier, in parameter identifiability, or in the quality-split structure.

## Initial status

The current benchmark on the two-phase-many sweep shows that DS-RE-CEQ is weak on the main objective `lm_eval/mmlu_5shot/bpb`:

- `R^2 = 0.0624`
- `RMSE = 0.0884`
- `Spearman = 0.2569`
- `Regret@1 = 0.1146`
- `P = 162`

This is not just “imperfect.” It is bad enough that the fitted optimum is clearly off the observed frontier.

## Hypothesis 1

The model is smoothing away the frontier. It may fit the bulk weakly, but the main practical failure is likely ranking the best schedules poorly.

## Changes to make

- Add a dedicated diagnostics subdir under `experiments/domain_phase_mix/exploratory/two_phase_many/`.
- Build plots for:
  - OOF predicted vs actual BPB
  - best actual BPB among top-k predicted
  - OOF residuals against coarse schedule aggregates
  - full-fit parameter heatmaps
  - restart-stability diagnostics

## Hypothesis 2

The parameterization is weakly identified in the 2-phase / 39-domain setting. In particular, the learned gate and interference may be partly redundant, and the model may lack enough sharing across the high/low quality bucket pairs.

## Changes to make

- Inspect full-fit parameters directly from `final_p`.
- Plot the effective 2-phase interference `g_1 * lambda_{1,d}` instead of raw gate and lambda separately.
- Compare across-seed optimum schedules and parameter spread.

## Results

Diagnostics were written under:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/dsre_ceq_debug/`

Main artifacts:

- `dsre_ceq_oof_actual_vs_predicted.png`
- `dsre_ceq_topk_best_actual.png`
- `dsre_ceq_residual_panels.png`
- `dsre_ceq_parameter_heatmaps.png`
- `dsre_ceq_restart_stability.png`
- `dsre_ceq_debug_summary.json`

Key results:

- OOF fit is still weak:
  - `R^2 = 0.0767`
  - `RMSE = 0.0878`
  - `Spearman = 0.2590`
  - `Regret@1 = 0.0991`
- Frontier retrieval is poor:
  - top-1 predicted run has actual BPB `2.2491`
  - top-5 best actual only improves to `2.1643`
  - top-10 best actual is still `2.1643`
  - global best is `2.1032`
- The OOF scatter shows that DS-RE-CEQ compresses predictions into a narrow band and smooths away the best observed region.
- The coarse residual panels are nearly flat:
  - Spearman values are close to zero for the six simple aggregates that were tested.
  - That suggests the failure is not explained by one obvious coarse aggregate.
- The full-fit parameter heatmap shows strong per-domain irregularity, including a very large effective-interference spike on `dolmino_synth_qa`.
- Restart stability is poor even when predicted optimum BPB is numerically stable:
  - predicted optimum BPB std across seeds: `0.0142`
  - observed regret@1 std across seeds: `0.0371`
  - mean pairwise TV distance between predicted optima: `0.4306`
  - max pairwise TV distance between predicted optima: `0.4991`
- In other words: different seeds predict similarly optimistic BPB values, but they reach them with very different schedules and choose materially different observed candidates.

Interpretation:

- The model is not just underfitting globally; it is weakly identified on the frontier.
- Quality-bucket structure is still a plausible problem, but the first-pass residual panels suggest the main issue is not a single missing coarse feature.
- The strongest current warning sign is instability and parameter pathology, especially in the effective interference term.

## Future Work

- [ ] Add high-vs-low local swap diagnostics within each retained CC topic
- [ ] Compare DS-RE-CEQ against the existing in-file ablations on the same top-k retrieval metric
- [ ] Add grouped or hierarchical residual diagnostics by topic family
- [ ] Add a parameter-trajectory view across restart seeds for the top 10 frontier domains
