# Debugging log for GRP 300M MMLU BPB fit

## Initial Status

The no-L2 GRP fit on `300m_6b` `lm_eval/mmlu_5shot/bpb` is poor:

- `qsplit240_core`: OOF RMSE `0.061997`, OOF Spearman `0.295815`, top-1 regret `0.039156`.
- `all`: OOF RMSE `0.061442`, OOF Spearman `0.286749`, top-1 regret `0.039156`.

The best observed row is `run_00018` at MMLU BPB `2.041565`, while the fitted model picks `run_00162`
at `2.080722`.

## Hypothesis 1: Optimizer Settings Are Too Weak

The first fit used the existing no-L2 metric-objective tuning path with a small start bank, top-3 starts,
and Powell refinement. Since previous GRP 300M debug work found optimizer sensitivity and boundary
saturation, the next step is to compare stronger optimizer settings before blaming the model class.

## Changes To Make

No production code changes yet. Run a local diagnostic sweep against the same materialized 300M MMLU BPB
tables:

- current deterministic start bank
- expanded deterministic and random log-space starts
- Powell with larger `maxiter`
- Nelder-Mead and L-BFGS-B local alternatives
- train-RMSE objective as a capacity check

## Results

Optimizer sweep artifacts:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_mmlu_5shot_bpb_20260428/optimizer_debug/`

Best CV-objective settings:

| setting | OOF RMSE | OOF Spearman | OOF R2 | top-1 regret |
|---|---:|---:|---:|---:|
| original `base9_powell80_cv` | 0.061997 | 0.295815 | 0.029157 | 0.039156 |
| longer `base9_powell240_cv` | 0.061214 | 0.321929 | 0.053509 | 0.039156 |
| extreme starts + Powell | 0.060291 | 0.337832 | 0.081838 | 0.043038 |
| random starts + Powell | 0.059905 | 0.349961 | 0.093544 | 0.037506 |
| random starts + Nelder-Mead | 0.060147 | 0.361895 | 0.086206 | 0.039156 |

Best train-RMSE objective:

| setting | train RMSE | OOF RMSE | OOF Spearman | top-1 regret |
|---|---:|---:|---:|---:|
| random starts + Powell train RMSE | 0.052787 | 0.064238 | 0.280676 | 0.037506 |
| extreme + random starts + Powell train RMSE | 0.052745 | 0.061927 | 0.323209 | 0.033320 |

Interpretation:

The optimizer was leaving a small amount of performance on the table, but not enough to explain the bad
fit. Stronger starts and longer Powell improve OOF RMSE from `0.061997` to `0.059905` and OOF Spearman
from `0.295815` to `0.349961`. Optimizing train RMSE can reduce in-sample RMSE to `0.052745`, but it does
not improve held-out prediction. This suggests the current GRP form has weak MMLU-specific signal and/or
the 300M MMLU BPB target is noisy relative to mixture features.

## Hypothesis 2: NNLS Sign Constraints Are The Bottleneck

If the GRP features are useful but the monotone NNLS head has the wrong sign constraints for MMLU BPB,
unconstrained linear or ridge heads on the same design should improve OOF fit.

## Results

Using the best random-start GRP feature parameters:

| head | OOF RMSE | OOF Spearman | OOF R2 | top-1 regret |
|---|---:|---:|---:|---:|
| mean baseline | 0.062921 | NaN | 0.000000 | 0.110814 |
| NNLS same features | 0.059905 | 0.349961 | 0.093544 | 0.037506 |
| unconstrained OLS | 0.061864 | 0.339760 | 0.033301 | 0.037506 |
| unconstrained RidgeCV | 0.061724 | 0.339415 | 0.037670 | 0.037506 |
| raw phase weights RidgeCV | 0.063421 | 0.122959 | -0.015981 | 0.078999 |

The design matrix is ill-conditioned (`cond ~= 6.8e7`), but relaxing NNLS does not help. The monotone
sign constraint is not the primary failure.

## Hypothesis 3: MMLU BPB Is A Different / Noisier Target Than The Loss Metrics GRP Was Designed For

On the `300m_6b` qsplit rows:

- `lm_eval/mmlu_5shot/bpb` has std `0.0629`.
- `eval/uncheatable_eval/bpb` has std `0.0183`.
- MMLU BPB vs uncheatable BPB Spearman is only `0.032`.
- MMLU BPB vs Paloma macro BPB Spearman is only `0.084`.
- MMLU BPB vs MMLU accuracy Spearman is `-0.069`.
- MMLU BPB vs MMLU choice logprob Spearman is `-0.982`, so the BPB label itself is internally
  consistent with the smooth logprob proxy.

Interpretation:

MMLU BPB is smooth enough as a metric, but its across-mixture variation is mostly not the same signal as
uncheatable or Paloma BPB, and it is not strongly reflected in MMLU accuracy at this scale. The current
GRP body was built for validation/perplexity-style loss targets; it does not capture much of the
benchmark-specific direction.

## Conclusion

Optimizer settings explain only a small part of the poor `300m_6b` MMLU BPB fit. The best local optimizer
sweep improves OOF Spearman from `0.296` to `0.350`, but OOF R2 remains under `0.10`. Unconstrained heads
do not fix it. The main issue appears to be target/model mismatch: MMLU BPB has a benchmark-specific
mixture signal that this GRP feature family does not represent well.

