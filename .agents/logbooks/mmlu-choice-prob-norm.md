# MMLU choice_prob_norm Surrogates: Research Logbook

## Scope
- Goal: Test whether GRP no-L2 fits `choice_prob_norm` better than MMLU BPB on the 300M qsplit swarm.
- Primary metrics: OOF RMSE, OOF Spearman, top-1 regret, raw-optimum geometry.
- Constraints: Use the existing 300M `qsplit240_core` panel; do not launch training or eval jobs.

## Baseline
- Date: 2026-04-28
- Code refs:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_mmlu_choice_prob_norm.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_mmlu_choice_prob_norm_20260428/report.md`
- Baseline numbers:
  - Prior 300M MMLU BPB GRP fit had OOF Spearman about `0.30-0.35` after optimizer debugging.

## Experiment Log
### 2026-04-28 - Fit standard and SL-Verb MMLU choice_prob_norm
- Hypothesis: `choice_prob_norm` should be easier to model than MMLU BPB because it is a soft multiple-choice correctness metric.
- Command:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_mmlu_choice_prob_norm.py
```

- Config: GRP power-family-penalty no-L2, `scale=300m_6b`, `run_set=qsplit240_core`, expanded start bank with 24 random starts, top-8 Powell refinement.
- Result:
  - Standard MMLU `choice_prob_norm` logit fit: OOF RMSE `0.003501`, OOF Spearman `0.578384`.
  - Standard MMLU `choice_prob_norm` raw fit: OOF RMSE `0.003528`, OOF Spearman `0.576762`.
  - SL-Verb `choice_prob_norm` logit fit: OOF RMSE `0.000529`, OOF Spearman `0.312399`.
  - SL-Verb `choice_prob_norm` raw fit: OOF RMSE `0.000533`, OOF Spearman `0.291175`.
- Interpretation:
  - Standard `choice_prob_norm` is a better target than MMLU BPB for GRP ranking, but the selected observed row still has substantial top-1 regret.
  - SL-Verb has much lower absolute variance, so RMSE looks tiny but rank signal is weaker.
  - Standard raw optima collapse almost entirely to broad text and are far from observed rows, so use this as a diagnostic signal rather than a deployment mixture.
- Next action: If this objective remains interesting, test constrained optima and benchmark-aggregate variants rather than trusting raw simplex optima.

### 2026-04-28 - Debug GRP assumptions for standard MMLU choice_prob_norm
- Hypothesis: The remaining fit gap may come from benchmark-mismatched GRP assumptions: family partition,
  probability transform, monotone NNLS head, or high/low pair aggregation.
- Command:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_mmlu_choice_prob_norm.py \
  --only-slug-prefix mmlu_5shot_choice_prob_norm \
  --family-scheme synthetic_tech \
  --block-variant full
```

- Additional probes:
  - Family-scheme sweep for standard MMLU logit `choice_prob_norm`.
  - Linear head probe comparing NNLS, OLS, and RidgeCV.
  - Feature block probe comparing full, no-pairs, signals-only, and family-only variants.
- Result:
  - Best family scheme: `synthetic_tech`, OOF Spearman `0.598493`, OOF RMSE `0.003467`.
  - Default family scheme: OOF Spearman `0.578384`, OOF RMSE `0.003501`.
  - Best linear head diagnostic: RidgeCV, OOF Spearman `0.604756`, OOF RMSE `0.003441`.
  - OLS was worse than NNLS, so monotonicity is not the primary issue.
  - Family-only features failed badly, so domain-level signal is still required.
- Interpretation:
  - `synthetic_tech + logit(choice_prob_norm)` is the current GRP diagnostic baseline.
  - Raw simplex optima remain unreliable: the surrogate predicts overly optimistic corner mixtures outside the observed panel.
- Links:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_mmlu_choice_prob_norm_20260428/debug_iteration_report.md`
  - `docs/debug-log-grp-mmlu-choice-prob-norm-fit.md`

### 2026-04-28 - Add constrained optimum diagnostics for choice_prob_norm
- Hypothesis: `choice_prob_norm` is a better fit target than MMLU BPB, but raw simplex optimization is
  unsafe because the target is bounded, narrow-range, and accuracy-like.
- Command:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_mmlu_choice_prob_norm.py \
  --only-slug-prefix mmlu_5shot_choice_prob_norm_logit \
  --family-scheme synthetic_tech \
  --block-variant full
```

- Result:
  - Standard MMLU accuracy vs standard `choice_prob_norm` Spearman: `0.866`.
  - Standard MMLU accuracy vs standard `choice_logprob_norm` Spearman: `-0.026`.
  - Standard MMLU BPB vs standard `choice_logprob_norm` Spearman: `-0.970`.
  - Standard MMLU BPB vs standard accuracy Spearman: `-0.069`.
  - Best observed standard-MMLU `choice_prob_norm`: `0.274843`.
  - Raw optimum predicted metric: `0.340919`, optimism `+0.066076`, nearest observed TV `0.748`.
  - Top-8-actual hull predicted metric: `0.268796`, nearest observed TV `0.002`.
  - Trust-blend cap `0.05` predicted metric: `0.274535`, optimism `-0.000307`, nearest observed TV `0.049`.
  - Trust-blend cap `0.10` predicted metric: `0.280172`, optimism `+0.005330`, nearest observed TV `0.100`.
- Interpretation:
  - The response to BPB-vs-choice-probability geometry is not a different raw optimizer; it is target-aware
    deployment. For benchmark choice probabilities, use logit target fitting plus top-actual hull or
    small trust-blend moves until raw benchmark optima have direct validation.
- Links:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_mmlu_choice_prob_norm_20260428/optimum_diagnostics.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_mmlu_choice_prob_norm_20260428/report.md`
