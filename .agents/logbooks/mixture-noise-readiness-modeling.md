# Mixture Noise and Optimization Readiness Modeling

## Scope

- Goal: model mixture-dependent noise, controllability, and optimization readiness well enough to decide which metrics can be optimized, which should be guardrails, and how to build aggregate objectives that remain deployable.
- Primary outputs: per-metric readiness labels, perturbation confidence diagnostics, heteroskedastic/noise-floor estimates, and readiness-weighted aggregate objectives.
- Fieldbook experiment: `exp_01kv93eqg3e759wwv6zqe2bfka`.
- Constraints: use existing 300M/proportional-controllability and repeated-anchor data first; do not submit new validation mixtures until the objective gives a plausible trust-region candidate rather than only an extrapolative raw optimum.

## Baseline

- Existing v4-style aggregate baseline: collaborator-style 5-factor aggregate over the selected 300M metric panel.
- Existing diagnostics feeding this thread:
  - Proportional controllability 300M: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/heteroskedastic_optimization_readiness_20260616/`
  - DCLM latent auxiliary modeling: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_calibrated_auxiliary_noise_floor_20260616/`, `dclm_calibrated_auxiliary_anchored_regression_20260616/`, and `dclm_leave_task_out_prediction_20260616/`
  - StarCoder repeated anchors: `.agents/logbooks/starcoder-heteroskedastic-snr.md`

## Experiment Log

### 2026-06-16 - Readiness-weighted aggregate DSP

- Hypothesis: heteroskedastic/readiness diagnostics can improve aggregate-objective construction by downweighting metrics that are noisy, weakly controlled, or directionally unreliable, while preserving stable BPB anchors as coverage constraints.
- Command:

```bash
uv run python experiments/domain_phase_mix/exploratory/two_phase_many/fit_readiness_weighted_aggregate_dsp.py
```

- Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/readiness_weighted_aggregate_dsp_20260616/`
- Key result:

| target | active metrics | OOF Spearman | OOF R2 | raw optimum TV to proportional | raw optimum phase max weights |
|---|---:|---:|---:|---:|---:|
| `original_factor` | 0 | 0.8996 | 0.8017 | 0.8163 | 0.2672 / 0.4496 |
| `strict_steerable` | 2 | 0.8694 | 0.6923 | 0.9578 | 0.9131 / 0.9854 |
| `steerable_guardrail` | 3 | 0.8908 | 0.8014 | 0.7378 | 1.0000 / 0.3034 |
| `steerable_guardrail_stabilized` | 26 | 0.9055 | 0.8535 | 0.6536 | 0.2014 / 0.8990 |
| `broad_screened` | 35 | 0.9148 | 0.8638 | 0.6801 | 0.2653 / 0.7095 |

- Interpretation:
  - This is a positive modeling result: `broad_screened` improves OOF fit over the original factor while incorporating readiness diagnostics.
  - Direct-readiness-only optimization is too narrow; `strict_steerable` collapses to a two-metric, near-single-domain objective.
  - Stable Paloma/uncheatable BPB anchors are necessary to prevent objective collapse and keep coverage pressure in the aggregate.
  - The raw optima remain extrapolative, so the result supports trust-region/path candidate construction, not direct submission of the unconstrained raw optimum.
- Next action: construct candidates on the proportional-to-`broad_screened` and proportional-to-`steerable_guardrail_stabilized` paths, score all selected metrics and guardrails, and only then decide whether a MoE scaling validation is justified.

### 2026-06-16 - Noise-shape critique: heteroskedasticity is not automatically skew

- Question: should the nuisance term around a fixed mixture be modeled as symmetric heteroskedastic noise, or as asymmetric/spike-prone noise from rare valuable training chunks appearing in a sampled training input?
- CC review: `ctc ask` session `6097ef5d-600e-4aca-8754-f31ff015ccd5`, `claude-opus-4-8`, `env -u ANTHROPIC_API_KEY`, read-only/no-shell permissions.
- Main critique:
  - Heteroskedasticity and skew/heavy tails are separate assumptions. A Gaussian with mixture-dependent variance is heteroskedastic but symmetric.
  - The floor-plus-upward-spikes mechanism is plausible near corners or low expected relevant-content count, but should not be treated as a global property of all metrics or mixtures.
  - Bounded metrics near a floor can show mechanical right-skew from finite-item binomial effects even without rare-data inclusion; diagnostics should be run on transformed scales such as logit where applicable.
  - Fixing the simulated-epoch subset estimates a conditional objective, not the marginal deployment objective over training randomness, and can bias or flip the argmax.
  - Winner's curse from selecting the best noisy mixture is a separate problem that exists even under symmetric noise and should be corrected or guarded against.
- Modeling implication:
  - Keep symmetric heteroskedastic models as an interior baseline.
  - Add diagnostics for conditional skew/tail asymmetry before introducing skewed likelihoods.
  - For optimization, prefer robust decision rules: lower-confidence-bound, median/quantile/CVaR, sign-stability, and fresh-seed validation of selected candidates.
- Next diagnostics to implement:
  - Within-mixture skewness and median-minus-mean by anchor, with uncertainty.
  - Upper-tail versus lower-tail asymmetry using quantile gaps or semivariance.
  - Skew/variance versus a rarity proxy such as expected relevant-domain mass or effective support.
  - Logit-scale versions for bounded metrics to separate rare-data inclusion from finite-item floor artifacts.
  - Winner's-curse correction for selected optima and path candidates.

### 2026-06-16 - Noise-shape and winner's-curse diagnostics

- Hypothesis: existing repeated-anchor data can tell whether we should replace the symmetric heteroskedastic noise approximation with a skewed/spike-prone likelihood, and whether the `broad_screened` aggregate remains useful after correcting for selection optimism.
- Command:

```bash
uv run python experiments/domain_phase_mix/exploratory/two_phase_many/analyze_noise_shape_and_winner_curse.py
```

- Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/noise_shape_winner_curse_20260616/`
- Plots:
  - `proportional_noise_skew_distribution.html`
  - `bounded_metric_raw_vs_logit_skew.html`
  - `starcoder_skew_vs_exposure_key_metrics.html`
  - `starcoder_sd_vs_exposure_key_metrics.html`
  - `starcoder_skew_rarity_association_top.html`
  - `readiness_path_gain_vs_winner_curse.html`
  - `winner_curse_corrections.html`
- Key result:

| diagnostic | value |
|---|---:|
| proportional repeat rows | 10 |
| proportional metric diagnostics | 518 |
| proportional median utility skew | -0.0670 |
| proportional share positive utility skew | 0.4691 |
| bounded raw-vs-logit skew Spearman | 0.7849 |
| bounded skew sign change after logit | 0.0927 |
| StarCoder repeated rows | 50 |
| StarCoder anchor-metric diagnostics | 580 |

- Path/winner's-curse result:

| target | first positive p50-adjusted path point | robust moderate-TV point | raw optimum p50/p90/p95-adjusted gain |
|---|---:|---:|---:|
| `broad_screened` | `t=0.25`, TV `0.1700`, gain `0.0412` | `t=0.50`, TV `0.3400`, p95-adjusted gain `0.2944` | `1.6162` / `1.3751` / `1.1755` |
| `steerable_guardrail_stabilized` | `t=0.33`, TV `0.2157`, gain `0.0872` | `t=0.50`, TV `0.3268`, p95-adjusted gain `0.2261` | `1.3240` / `1.1116` / `1.0354` |

- Interpretation:
  - The global floor-plus-upward-spike noise story is not supported as a blanket likelihood. The proportional-repeat skew distribution is centered near zero, and bounded-metric logit checks mostly preserve skew sign.
  - The StarCoder repeated-anchor panel strongly supports mixture-dependent variance, but not a simple rarity-implies-positive-skew law. Many generic BPB metrics show larger noise scale at high StarCoder exposure, not at low exposure.
  - The useful modeling move is not to globally switch to a skewed likelihood. Keep symmetric heteroskedastic noise as the default interior approximation, then add robust decision layers: path/trust-region constraints, lower-confidence or quantile objectives, sign stability, and fresh-seed validation for selected candidates.
  - Winner's-curse correction matters. Small path moves do not survive selection correction, but `broad_screened` becomes credible around `t=0.5` and remains strongly positive at the raw optimum even after conservative path-level correction.
- Next action: carry `broad_screened` forward as the most promising aggregate objective, but select deployable candidates from trust-region/path surfaces and validate them with fresh runs rather than directly trusting the unconstrained raw optimum.

### 2026-06-17 - StarCoder heteroskedasticity mechanism follow-up

- Question: what is the strongest defensible explanation for the observed StarCoder mixture-dependent variance, and how should the theory document frame heteroskedasticity without overclaiming a skewed/noisy-spike likelihood?
- Command:

```bash
uv run python experiments/domain_phase_mix/exploratory/two_phase_many/analyze_starcoder_heteroskedasticity_mechanisms.py
```

- Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_heteroskedasticity_mechanisms_20260617/`
- CC review: `ctc ask` session `a3c38fbf-68e1-483c-9679-6d28c922c7ab`, `claude-opus-4-8`, `env -u ANTHROPIC_API_KEY`, read-only/no-shell/no-write permissions.
- Key measurements:

| diagnostic | value |
|---|---:|
| repeat rows | 50 |
| anchors | 10 |
| scalar eval metrics | 58 |
| Brown-Forsythe reject share at p<0.05 | 0.9310 |
| max key-metric std ratio | 134.6123 |
| max key-metric std ratio excluding `starcoder_only` | 13.4643 |

- Key metric examples:

| metric | std max/min | std max/min excluding `starcoder_only` | strongest log-std predictor |
|---|---:|---:|---|
| `eval/bpb` | 128.2185 | 6.4808 | `log_mean`, R2 0.8545 |
| `eval/uncheatable_eval/bpb` | 75.9066 | 11.5550 | `log_mean`, R2 0.7958 |
| `eval/uncheatable_eval/github_python/bpb` | 26.2537 | 12.8496 | `log_mean`, R2 0.9141 |

- Interpretation:
  - Heteroskedasticity is robustly present in this repeated-anchor panel: most scalar eval metrics reject equal variance across anchors, and key BPB metrics retain large local-std ratios even excluding the `starcoder_only` vertex.
  - The strongest observed predictor of local log standard deviation is mean performance scale, not phase concentration alone. Log transforms reduce variance ratios but do not eliminate them, so mean-variance coupling is important but incomplete.
  - The `starcoder_only` vertex has the highest variance for many metrics even though between-bucket interleaving is degenerate there. This points away from a pure mixture-composition/interleaving story and toward within-bucket subset choice, repetition/effective support, and trajectory sensitivity.
  - Cross-BPB repeat residual correlations are very high, suggesting a shared upstream source of repeat noise rather than independent benchmark measurement noise.
  - Rare high-leverage chunk inclusion remains plausible when the training subset is resampled across seeds, but it is only one mechanism. It should not be promoted to a global right-skewed likelihood without direct skew/tail diagnostics.
- Theory update:
  - Updated `theory.md` Section 6 to frame conditional heteroskedasticity as a second-moment property, separate it from skew/heavy-tail claims, introduce between-bucket versus within-bucket randomness, use the law of total variance as an experimental-control diagnostic, and state optimization implications via heteroskedastic weighting, variance-stabilizing transforms, local replication, and lower-confidence/risk-aware selection.
