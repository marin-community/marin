# Round 5: why `weibull_softplus_unscaled` beats matched-seed OLMix by so little on Table 9 (2026-09-04)

Scope: the frozen successor `weibull_softplus_unscaled` (WSPU), fitted per Table-9 component on the canonical 280-run
Delphi 3e18 panel with the heldout inner-fold protocol, against the OLMix KL=0.005 cap-4 baseline at data seed 662009
and trainer seed 0. All development evidence comes from the 247-coordinate Table-9 bank of the canonical registry
(246 for the learned remedies, which drop the coordinate measured by the matched-seed OLMix run; section 6)
(`reference_outputs/single_phase_heldout_benchmark_20260902/`, manifest `6cce727e…`; the pending refresh has not
landed). Nothing was launched. Nothing was tuned to the matched seed: every learned remedy is scored
leave-one-source-out on the bank, and the matched-seed runs are used only as the held-out check of the frozen model.

## 0. TLDR

- The fitted heads did not predict the 15 systematic regressions. At cap 7 they predict 8 of the 15 components to get
  worse and, on average, predict those 15 to improve by 0.017 BPB; they got worse by 0.048. Across all 51 components the
  heads predict a macro gain of 0.078 BPB (cap 7) against an observed 0.009: 88% of the predicted gain is optimism, and
  the residual is proportional to the predicted delta (correlation −0.82; family explains 35% of the residual
  variance, the predicted delta 67%).
- The optimism has two named sources, both visible in the bank before the runs. (a) The heads credit 18 small Common
  Crawl buckets (0.5–4% each, 31% of the WSPU mixture against 20% under OLMix) with a summed −0.043 BPB macro gain,
  −0.046 on the 15 regressors; the bank shows no such gain (regressor residual correlates +0.50 with the CC-low share).
  (b) The stack benefit curve is too steep: the code family is predicted to gain −0.141 and gains −0.033; the bank's
  code residual correlates +0.89 with the stack share and is +0.116 on coordinates with stack_edu at ≥ 5 epochs. The
  heads did know that cutting synthetic QA and OLMOCR hurts the regressors (+0.046 predicted for that part alone) and
  were roughly right; the fantasy CC gain cancelled it.
- The error does not come from epoch-cap activity (cap 8 has no active cap and the largest residual), from the cap
  binding, from distance to the panel as such (OLMix is further from the panel, TV 0.51 vs 0.39, and is predicted to
  within 0.006), or from component noise (|residual| vs panel repeat SD +0.12; the matched- and original-seed
  residual RMS agree to 0.001 at caps 6 and 7 and 0.004 at cap 8, while the seed gap is 0.008–0.014 per component). Aggregation and component-name
  handling are clean: the harness macro is the unweighted 51-component mean and reproduces the W&B macro on all four
  runs to 1e-6; the 51 names map one-to-one; the sweep's stored components equal the W&B evaluations bit for bit.
- No offline remedy improves selection on the bank. Descriptor and kernel residual calibration remove the bias
  (archive bias −0.025 → +0.001, RMSE 0.038 → 0.020) and leave regret@1 at 0.014–0.016 with the frontier ranked
  worse (10th → 13th–47th); reliability weighting, family-mean and code-excluded objectives, exposure clamping all tie
  or lose; a share floor of 2% (buckets under it count as absent) is the only rule that improves best-of-5 regret
  (0.0132 → 0.0086, P(better) 0.66) and it costs frontier rank and Spearman. Fifth demonstration on this bank that
  calibration and selection are different properties.
- Candidate set: interpolations between OLMix and cap 7, successor optima under synthetic-QA/OLMOCR floors, and the
  bank's own best region. The empirical evidence (bank top coordinates, 1.058–1.067, all with synthetic QA 0.10–0.20,
  OLMOCR 0.02–0.05, stack 0.18–0.30, CC-high 0.23–0.32, curated 0.13–0.17) points at the `floor_qa0.2_olmocr0.08_cap7`
  and `interp_cap7_0.5` candidates as the ones that restore synthetic QA/OLMOCR mass while keeping the code gain;
  the replicated frontier-centre mixture (26 seeds, 1.0639 ± 0.0041) remains the best-measured Table-9 mixture and
  beats every WSPU cap. Any of these needs a run to be validated; none is launched.

## 1. Inputs

- Sweep artifacts: `reference_outputs/delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902/` (candidate weights,
  runtime predictions 1.008099 / 1.004632 / 1.003945 for caps 6/7/8, measured components).
- OLMix weights: W&B training run `marin-community/marin/olmix_onephase_table9_d001_kl0p005_cap4_3e18-eff7f7`, config
  `data.train_weights[0][1]` (62 keys; the 23 Paloma keys carry zero mass; the 39 panel buckets sum to 1).
- Evaluations (project `marin-community/marin-eval`, keys `olmo_base_easy/table9/<component>/bpb` and the macro):
  matched seed OLMix `o518aq9w` 1.0768647, cap 6 `077oz9yd` 1.0727456, cap 7 `esr6cjuw` 1.0678004, cap 8 `22eqjh7q`
  1.0795197; original seed cap 6/7/8 `nsepufp5`, `w2mw7ur4`, `8vwwl4sx` 1.0721714 / 1.0729639 / 1.0736207.
- The in-process refit of the 51 heads reproduces the sweep's runtime predictions exactly (1.0080986 / 1.0046324 /
  1.0039446), so the audited model is the frozen one. Per-bucket curves reconstructed from the fitted heads reproduce
  the heldout shards to 1e-15.

Scripts (all in `two_phase_many/`): `single_phase_round5_olmix_gap_20260904.py` (51-row tables, bucket decomposition,
extrapolation flags), `single_phase_round5_dose_curves_20260904.py` (fitted single-bucket dose curves),
`single_phase_round5_remedies_20260904.py` (offline remedies scored on the bank),
`single_phase_round5_candidates_20260904.py` (candidate set). Outputs in
`reference_outputs/single_phase_observatory_benchmark_20260902/olmix_gap_round5/`. Tests in
`tests/test_single_phase_round5_scripts_20260904.py`.

## 2. Did the heads predict the regressions?

No. Family means of the OLMix-to-WSPU delta (BPB; negative = WSPU better; residual = observed − predicted):

| family (n) | cap 6 predicted / observed / residual | cap 7 | cap 8 |
|---|---|---|---|
| arc (2) | −0.062 / +0.058 / +0.121 | −0.054 / +0.050 / +0.104 | −0.047 / +0.077 / +0.124 |
| mmlu (4) | +0.010 / +0.028 / +0.018 | +0.019 / +0.037 / +0.018 | +0.024 / +0.060 / +0.036 |
| qa_reading (8) | −0.066 / +0.023 / +0.089 | −0.052 / +0.036 / +0.088 | −0.042 / +0.067 / +0.108 |
| commonsense (5) | −0.019 / −0.004 / +0.015 | −0.014 / −0.005 / +0.009 | −0.011 / +0.003 / +0.013 |
| basic_skills (6) | −0.068 / −0.005 / +0.063 | −0.076 / −0.035 / +0.041 | −0.080 / +0.004 / +0.084 |
| math (7) | −0.044 / −0.010 / +0.034 | −0.046 / −0.020 / +0.026 | −0.046 / −0.021 / +0.025 |
| code (19) | −0.125 / −0.026 / +0.099 | −0.141 / −0.033 / +0.108 | −0.149 / −0.036 / +0.113 |
| macro (51) | −0.074 / −0.004 | −0.078 / −0.009 | −0.079 / +0.003 |

- Components predicted worse: 6 / 10 / 10 of 51 at caps 6/7/8; observed worse: 15 / 17 / 22. Sign agreement 42 / 42 /
  37 of 51. Of the 15 systematic regressors the heads predict 6 / 8 / 8 to get worse; their mean predicted delta is
  −0.027 / −0.017 / −0.010 against observed +0.043 / +0.048 / +0.074.
- The MMLU aggregates are the only family the heads predicted to regress, and by less than half the observed amount.
  ARC-challenge (predicted −0.108, observed +0.052), NaturalQuestions (−0.160 / +0.046), DROP (−0.123 / +0.075) and
  MMLU social sciences (−0.070 / +0.043) are the largest sign errors. The per-component tables for all three caps are
  in Appendix A.
- Levels: the OLMix mixture itself is predicted at 1.0825 (observed 1.0769, −0.006) but with opposite family errors:
  the regressor families are predicted too pessimistically at OLMix (arc −0.091, qa_reading −0.055, mmlu social
  sciences −0.106) and code too optimistically (+0.027). At the WSPU mixtures every family is predicted too
  optimistically (+0.03 to +0.14). The delta error is the sum of the two.

## 3. Where the predicted gain came from

The successor is additive in per-bucket materialized epochs, so the OLMix-to-WSPU prediction decomposes exactly per
bucket (hybrid mixtures, additivity gap 7e-15). Cap 7, macro and regressor contributions:

| bucket group | OLMix share → WSPU share | predicted macro contribution | on the 15 regressors | on code |
|---|---|---|---|---|
| synthetic QA + OLMOCR (cut) | 0.463 → 0.162 | +0.027 | +0.046 | +0.003 |
| stack_edu + stack_edu_fim (raised to cap) | 0.169 → 0.296 | −0.035 | +0.002 | −0.082 |
| 18 CC buckets at 0.5–4% each | 0.196 → 0.310 | −0.043 | −0.046 | −0.050 |
| curated, math, other synthetic | | −0.027 | −0.019 | −0.012 |
| total | | −0.078 | −0.017 | −0.141 |

- The fitted single-bucket dose curves (`dose_curves_weibull_softplus_unscaled.csv`) show the mechanism. For the
  regressors, synthetic QA is worth −0.104 at 1 epoch and −0.153 at 4 epochs with no harm upturn (panel maximum 2.34
  epochs; the OLMix value of 4 epochs is extrapolated, and the heads extrapolate a continued benefit). Each CC-low bucket
  is worth −0.006 to −0.013 at 1–2 epochs for the regressors; eighteen of them add up. Stack_edu is worth −0.204 at 3
  epochs and −0.261 at 6 for code, essentially zero for the regressors, with no harm inside 10 epochs.
- Extrapolation in epochs is limited: cap 7 places only stack_edu beyond the panel's maximum exposure (6.98 vs 6.85);
  cap 8 both stack buckets (7.8 and 7.6 vs 6.9 and 7.1); OLMix places synthetic QA (4.0 vs 2.34) and OLMOCR (4.0 vs
  3.50) beyond it. Clamping exposures at the panel maximum changes the cap-7 prediction by 0.0002. The panel has 8 rows
  with stack_edu above 4 epochs, 3 above 6, one with synthetic QA above 2 epochs, none with stack above 20% or
  synthetic QA above 25%: the WSPU direction is thinly supported in share space, the OLMix direction almost not at all.
- The panel's effective bucket count is 22–39 (median 28); WSPU cap 7 has 20, OLMix 10. The credit for spreading mass
  over many small buckets is learned from rows that are all spread, where per-bucket benefits at small exposure are
  poorly separated.

## 4. Correlates of the error

Per-component residuals at the matched seed (cap 7 unless stated):

- Allocation change: the residual is a function of what the head predicted, not of noise: correlation of residual with
  predicted delta −0.86 / −0.82 / −0.78 (caps 6/7/8); every family is over-predicted in proportion to its predicted
  gain. The bank residuals (247 coordinates, observed − predicted, positive = optimistic) put the same structure in
  share space: macro residual correlates +0.68 with the stack share and −0.66 with the CC-high share; the code residual
  +0.89 with stack; the regressor residual +0.50 with the CC-low share and −0.34 with OLMOCR. Joint OLS on shares:
  macro residual = −0.03 + 0.21·stack + 0.17·cc_low − 0.02·synth_qa − 0.08·olmocr.
- Epoch-cap activity: cap 6 has four active caps, cap 7 three, cap 8 none; the residual grows with the cap (RMS 0.090,
  0.096, 0.104). The cap does not bind the successor's optimum (7.8 epochs), so activity is not the driver.
- Distance from the panel: OLMix is the furthest of the four mixtures (TV 0.51 to the nearest panel row, vs 0.385 /
  0.394 / 0.401) and the best predicted. The bank's nearest coordinates to cap 7 are the WSPU sweep's own original-seed
  runs (TV 0–0.10) with macro residuals +0.059 to +0.070, code +0.12 to +0.14, regressors +0.02: the optimism was
  measurable in development data before the matched-seed runs.
- Component noise: |residual| vs panel repeat SD +0.18 / +0.12 / +0.36; vs the WSPU seed gap (original minus matched
  seed, median 0.008–0.014 per component) −0.04 / −0.01 / +0.06. Residual RMS at the original seed equals the matched
  seed to 0.001 at caps 6 and 7 (0.090 vs 0.091, 0.096 vs 0.096) and to 0.004 at cap 8 (0.104 vs 0.100). The regressions are not seed noise: 13 of the 15 regress at both seeds and all three caps.
- Task family explains 35% of the cap-7 residual variance; the predicted delta alone explains 67%.

## 5. Aggregation and component-name audit

- Harness Table-9 aggregation is the unweighted mean of 51 components (weights all 1/51); it equals the W&B macro
  `table9_51_component_macro_bpb` on all seven runs to 1e-6.
- The 51 W&B component keys (`olmo_base_easy/table9/<name>/bpb`) map one-to-one onto the harness names
  (`olmo_base_eval/easy_bpb/<name>/bpb`); the sweep's `measured_table9_components.csv` equals the W&B original-seed
  evaluations to 1e-16; the registry's epoch-cap sources store short names and are remapped (round-3 fix, unchanged).
- The successor optimizes the same unweighted macro it is scored on. Nothing in the aggregation explains the gap.

## 6. Offline remedies, scored by selection value on the bank

Every remedy transforms the frozen successor's per-component bank predictions (or its fitted curves). The bank
coordinate measured by the matched-seed OLMix evaluation (`3c5d1d0d…`, W&B `o518aq9w`) is dropped from every remedy fit
and score, leaving 246 coordinates; the WSPU matched-seed evaluations are not in the registry. Learned remedies
(descriptor and kernel residual calibration, bank-derived reliability weights) are fitted leave-one-source-out over 16
primary groups, holding out every coordinate with any membership in the held-out source (multi-source coordinates
are held out with each of their sources). Archive stratum = 157 non-dose coordinates; the frontier is the HPR-280 tied
control (1.0579, n=1). Bootstrap: 1000 coordinate resamples, paired against the successor. The ridge values and kernel
bandwidths are pre-specified grids reported side by side; none is selected on the pooled out-of-fold scores.

| remedy | regret@1 (archive) | best-of-5 regret | frontier predicted rank | Spearman | bias | RMSE | regret@1 (dose) | Δregret@1 vs successor [95% CI] | P(better) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| successor | 0.0157 | 0.0132 | 10 | 0.900 | -0.0252 | 0.0378 | 0.0015 | reference |  |
| residual_mean_shift | 0.0157 | 0.0132 | 10 | 0.896 | -0.0066 | 0.0295 | 0.0015 | +0.0000 [+0.0000, +0.0000] | 0.00 |
| residual_calibration@ridge1 | 0.0143 | 0.0143 | 43 | 0.897 | +0.0006 | 0.0206 | 0.0357 | -0.0005 [-0.0014, +0.0035] | 0.71 |
| residual_calibration_by_family@ridge1 | 0.0143 | 0.0143 | 43 | 0.897 | +0.0006 | 0.0206 | 0.0357 | -0.0005 [-0.0014, +0.0035] | 0.71 |
| kernel_residual@tv0.1 | 0.0151 | 0.0143 | 13 | 0.889 | +0.0009 | 0.0194 | 0.0015 | -0.0004 [-0.0014, +0.0023] | 0.57 |
| kernel_regression@tv0.1 | 0.0140 | 0.0132 | 13 | 0.823 | +0.0075 | 0.0225 | 0.1940 | +0.0115 [-0.0025, +0.0486] | 0.67 |
| reliability_panel_repeat | 0.0143 | 0.0143 | 31 | 0.927 | -0.1567 | 0.1579 | 0.0015 | -0.0006 [-0.0014, +0.0023] | 0.71 |
| reliability_bank_residual | 0.0143 | 0.0143 | 23 | 0.926 | -0.0794 | 0.0823 | 0.0015 | -0.0006 [-0.0014, +0.0035] | 0.71 |
| family_mean_objective | 0.0143 | 0.0143 | 31 | 0.855 | +0.0956 | 0.0992 | 0.0015 | -0.0004 [-0.0014, +0.0023] | 0.61 |
| macro_excluding_code | 0.0168 | 0.0143 | 50 | 0.766 | +0.1683 | 0.1707 | 0.0015 | +0.0016 [-0.0014, +0.0111] | 0.23 |
| clamp_exposure_panel_max | 0.0151 | 0.0132 | 11 | 0.916 | -0.0227 | 0.0344 | 0.0015 | -0.0004 [-0.0014, +0.0000] | 0.57 |
| clamp_exposure_panel_p95 | 0.0168 | 0.0143 | 56 | 0.783 | +0.0012 | 0.0256 | 0.0015 | +0.0034 [-0.0014, +0.0119] | 0.09 |
| no_credit_below_share0.01 | 0.0157 | 0.0143 | 42 | 0.892 | +0.0223 | 0.0423 | 0.0015 | +0.0001 [+0.0000, +0.0023] | 0.00 |
| no_credit_below_share0.02 | 0.0151 | 0.0086 | 20 | 0.810 | +0.0860 | 0.0926 | 0.0015 | -0.0006 [-0.0072, +0.0022] | 0.52 |
| no_credit_below_share0.02+residual_calibration@ridge1 | 0.0160 | 0.0086 | 48 | 0.603 | -0.0068 | 0.0315 | 0.0015 | +0.0002 [-0.0072, +0.0091] | 0.25 |

- Residual calibration fixes the bias and not the ordering: descriptor ridge (13 mixture descriptors) and kernel
  residual smoothing (TV bandwidth 0.1) bring the archive bias to +0.001 and the RMSE to 0.020, and every one of them
  still picks a WSPU-sweep coordinate (measured 1.072–1.073) while ranking the frontier 13th–47th. On the dose stratum
  the descriptor calibration is harmful (regret 0.0015 → 0.036); kernel residuals are neutral there.
- Reliability weighting (panel repeat variance or bank residual variance) and the family-mean objective change the
  pick by one neighbour (regret 0.0143, P(better) 0.71, difference −0.0007) at the cost of large bias; excluding the
  code family is worse (regret 0.0168, frontier 50th).
- Exposure clamping is a no-op (panel max) or harmful (95th percentile). Treating buckets under a 2% share as absent
  is the one rule with a best-of-5 gain (0.0132 → 0.0086, P(better) 0.66) but a worse frontier rank (20th) and
  Spearman (0.81); combining it with calibration loses the gain. Its own optimum (`rule_share_floor0.02_cap7`) is not
  a sensible candidate (synthetic QA 0.10).
- Pure kernel regression on the bank (no model) at bandwidth 0.1 selects as well as the successor (regret 0.0140) and
  fails at larger bandwidths; the bank is too sparse to replace the model.

## 7. Candidate set

Weights in `candidates_weights.csv`; per-component predicted effects (plain successor) in
`candidates_component_effects.csv`. "descriptor-calibrated Δ" applies the full-bank descriptor residual model (ridge 1);
"kernel-corrected macro" adds the TV-0.2 kernel mean of bank residuals; both are fitted on the 246-coordinate bank without
the matched-seed OLMix coordinate. Both are bias corrections that did not improve selection, reported for scale only.
Cap 7 is a per-bucket upper bound of 7 epochs on the 1/2048 block grid; "at the bound" means the returned block count
equals the cap's block count (sweep candidates report their stored activity). Box candidates round the ±0.05 box
inward on the grid.

| candidate | synth QA | OLMOCR | stack | eff. buckets | max epochs | buckets at the 7-epoch bound | TV to OLMix | TV to nearest panel row | TV to nearest bank coord (its measured BPB) | predicted macro (Δ vs OLMix) | descriptor-calibrated Δ | kernel-corrected macro |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---:|
| olmix | 0.333 | 0.130 | 0.169 | 10.0 | 4.00 | none | 0.000 | 0.514 | 0.046 (1.0851) | 1.0825 (+0.0000) | +0.0000 | 1.1052 |
| wspu_cap6 | 0.131 | 0.042 | 0.254 | 21.5 | 6.00 | dolma3_stack_edu, dolmino_stack_edu_fim, dolmino_synth_code, dolmino_synth_math | 0.377 | 0.385 | 0.000 (1.0722) | 1.0081 (-0.0744) | -0.0487 | 1.0385 |
| wspu_cap7 | 0.124 | 0.038 | 0.296 | 20.1 | 6.99 | dolma3_stack_edu, dolmino_stack_edu_fim, dolmino_synth_math | 0.396 | 0.394 | 0.000 (1.0730) | 1.0046 (-0.0779) | -0.0481 | 1.0367 |
| wspu_cap8 | 0.118 | 0.036 | 0.326 | 19.2 | 7.81 | none | 0.412 | 0.401 | 0.000 (1.0736) | 1.0039 (-0.0786) | -0.0453 | 1.0373 |
| interp_cap6_0.25 | 0.283 | 0.108 | 0.191 | 13.4 | 4.50 | none | 0.094 | 0.462 | 0.112 (1.0769) | 1.0444 (-0.0381) | -0.0300 | 1.0718 |
| interp_cap6_0.5 | 0.232 | 0.086 | 0.212 | 16.5 | 5.00 | none | 0.189 | 0.423 | 0.167 (1.0769) | 1.0256 (-0.0569) | -0.0428 | 1.0554 |
| interp_cap6_0.75 | 0.182 | 0.064 | 0.233 | 19.3 | 5.50 | none | 0.283 | 0.400 | 0.094 (1.0722) | 1.0139 (-0.0687) | -0.0475 | 1.0445 |
| interp_cap7_0.25 | 0.281 | 0.107 | 0.201 | 13.2 | 4.75 | none | 0.099 | 0.465 | 0.118 (1.0769) | 1.0420 (-0.0405) | -0.0308 | 1.0699 |
| interp_cap7_0.5 | 0.228 | 0.084 | 0.233 | 16.0 | 5.49 | none | 0.198 | 0.428 | 0.177 (1.0769) | 1.0218 (-0.0607) | -0.0435 | 1.0525 |
| interp_cap7_0.75 | 0.176 | 0.061 | 0.264 | 18.4 | 6.24 | none | 0.297 | 0.399 | 0.094 (1.0722) | 1.0097 (-0.0728) | -0.0470 | 1.0416 |
| floor_qa0.2_olmocr0.08_cap7 | 0.200 | 0.080 | 0.296 | 16.3 | 6.99 | dolma3_stack_edu, dolmino_stack_edu_fim, dolmino_synth_math | 0.313 | 0.422 | 0.119 (1.0730) | 1.0096 (-0.0730) | -0.0484 | 1.0433 |
| floor_qa0.2_olmocr0.13_cap7 | 0.200 | 0.130 | 0.290 | 15.0 | 6.96 | dolmino_synth_math | 0.279 | 0.447 | 0.147 (1.0711) | 1.0151 (-0.0674) | -0.0489 | 1.0487 |
| floor_qa0.25_olmocr0.08_cap7 | 0.250 | 0.080 | 0.290 | 14.4 | 6.96 | dolmino_synth_math | 0.278 | 0.447 | 0.145 (1.0711) | 1.0141 (-0.0684) | -0.0457 | 1.0480 |
| floor_qa0.25_olmocr0.13_cap7 | 0.250 | 0.130 | 0.277 | 13.4 | 6.94 | dolmino_synth_math | 0.243 | 0.471 | 0.177 (1.0711) | 1.0204 (-0.0621) | -0.0449 | 1.0535 |
| floor_qa0.3_olmocr0.08_cap7 | 0.300 | 0.080 | 0.277 | 12.8 | 6.94 | dolmino_synth_math | 0.242 | 0.471 | 0.177 (1.0711) | 1.0202 (-0.0623) | -0.0410 | 1.0535 |
| floor_qa0.3_olmocr0.13_cap7 | 0.300 | 0.130 | 0.264 | 11.8 | 6.94 | dolmino_synth_math | 0.208 | 0.489 | 0.214 (1.0711) | 1.0273 (-0.0552) | -0.0412 | 1.0594 |
| rule_share_floor0.02_cap7 | 0.099 | 0.028 | 0.296 | 21.0 | 6.99 | dolma3_stack_edu, dolmino_stack_edu_fim, dolmino_synth_math | 0.447 | 0.354 | 0.102 (1.0730) | 1.0216 (-0.0609) | -0.0233 | 1.0522 |
| rule_clamp_panel_max_cap7 | 0.124 | 0.038 | 0.293 | 20.3 | 6.99 | dolmino_stack_edu_fim, dolmino_synth_math | 0.395 | 0.392 | 0.003 (1.0730) | 1.0048 (-0.0778) | -0.0465 | 1.0367 |
| box0.05_around_olmix_cap7 | 0.284 | 0.081 | 0.269 | 13.7 | 6.94 | dolmino_synth_math | 0.242 | 0.452 | 0.174 (1.0711) | 1.0188 (-0.0638) | -0.0418 | 1.0515 |
| box0.05_around_bank_top5_cap7 | 0.123 | 0.038 | 0.296 | 20.2 | 6.99 | dolma3_stack_edu, dolmino_stack_edu_fim, dolmino_synth_math | 0.398 | 0.389 | 0.008 (1.0730) | 1.0047 (-0.0778) | -0.0482 | 1.0366 |
| bank_top1 | 0.101 | 0.024 | 0.301 | 20.8 | 16.03 | dolma3_stack_edu, dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_math, dolmino_synth_thinking | 0.523 | 0.340 | 0.000 (1.0579) | 1.0342 (-0.0484) | -0.0167 | 1.0618 |
| bank_top2 | 0.131 | 0.037 | 0.201 | 23.1 | 11.89 | dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_code, dolmino_synth_math | 0.446 | 0.309 | 0.000 (1.0639) | 1.0394 (-0.0432) | -0.0350 | 1.0633 |
| bank_top3 | 0.100 | 0.025 | 0.282 | 21.7 | 15.59 | dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_math, dolmino_synth_thinking | 0.521 | 0.327 | 0.000 (1.0660) | 1.0339 (-0.0486) | -0.0198 | 1.0601 |
| bank_top4 | 0.145 | 0.039 | 0.231 | 21.0 | 13.32 | dolma3_finemath_3plus, dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_code, dolmino_synth_math, dolmino_synth_thinking | 0.430 | 0.349 | 0.000 (1.0663) | 1.0382 (-0.0443) | -0.0331 | 1.0656 |
| bank_top5 | 0.201 | 0.048 | 0.178 | 17.0 | 7.99 | dolma3_finemath_3plus, dolmino_synth_math, dolmino_synth_thinking | 0.353 | 0.438 | 0.000 (1.0664) | 1.0499 (-0.0326) | -0.0194 | 1.0801 |
| bank_top5_mean | 0.135 | 0.034 | 0.239 | 22.6 | 11.37 | dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_math, dolmino_synth_thinking | 0.450 | 0.318 | 0.070 (1.0663) | 1.0255 (-0.0571) | -0.0382 | 1.0526 |
| bank_replicated_n26 | 0.131 | 0.037 | 0.201 | 23.1 | 11.89 | dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_code, dolmino_synth_math | 0.446 | 0.309 | 0.000 (1.0639) | 1.0394 (-0.0432) | -0.0350 | 1.0633 |
| bank_replicated_n8 | 0.079 | 0.028 | 0.232 | 23.8 | 16.54 | dolma3_finemath_3plus, dolma3_wikipedia, dolmino_stem_heavy_crawl, dolmino_synth_code, dolmino_synth_math, dolmino_synth_thinking | 0.528 | 0.323 | 0.000 (1.0739) | 1.0527 (-0.0299) | -0.0135 | 1.0774 |

Predicted family-mean deltas from OLMix (plain successor):

| candidate | arc | basic_skills | code | commonsense | math | mmlu | qa_reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| olmix | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| wspu_cap6 | -0.0621 | -0.0678 | -0.1252 | -0.0191 | -0.0435 | +0.0104 | -0.0659 |
| wspu_cap7 | -0.0535 | -0.0764 | -0.1407 | -0.0141 | -0.0460 | +0.0185 | -0.0518 |
| wspu_cap8 | -0.0472 | -0.0801 | -0.1486 | -0.0106 | -0.0461 | +0.0244 | -0.0415 |
| interp_cap6_0.25 | -0.0620 | -0.0269 | -0.0470 | -0.0160 | -0.0159 | -0.0142 | -0.0645 |
| interp_cap6_0.5 | -0.0718 | -0.0454 | -0.0803 | -0.0200 | -0.0284 | -0.0120 | -0.0767 |
| interp_cap6_0.75 | -0.0712 | -0.0589 | -0.1059 | -0.0207 | -0.0376 | -0.0036 | -0.0765 |
| interp_cap7_0.25 | -0.0589 | -0.0311 | -0.0541 | -0.0145 | -0.0178 | -0.0121 | -0.0609 |
| interp_cap7_0.5 | -0.0672 | -0.0522 | -0.0920 | -0.0174 | -0.0313 | -0.0082 | -0.0701 |
| interp_cap7_0.75 | -0.0650 | -0.0672 | -0.1203 | -0.0169 | -0.0406 | +0.0021 | -0.0666 |
| floor_qa0.2_olmocr0.08_cap7 | -0.0558 | -0.0751 | -0.1288 | -0.0129 | -0.0374 | +0.0024 | -0.0493 |
| floor_qa0.2_olmocr0.13_cap7 | -0.0506 | -0.0741 | -0.1198 | -0.0112 | -0.0346 | +0.0006 | -0.0401 |
| floor_qa0.25_olmocr0.08_cap7 | -0.0549 | -0.0710 | -0.1203 | -0.0124 | -0.0309 | -0.0031 | -0.0470 |
| floor_qa0.25_olmocr0.13_cap7 | -0.0492 | -0.0691 | -0.1089 | -0.0112 | -0.0285 | -0.0057 | -0.0385 |
| floor_qa0.3_olmocr0.08_cap7 | -0.0520 | -0.0655 | -0.1089 | -0.0122 | -0.0246 | -0.0080 | -0.0431 |
| floor_qa0.3_olmocr0.13_cap7 | -0.0472 | -0.0629 | -0.0960 | -0.0101 | -0.0211 | -0.0107 | -0.0348 |
| rule_share_floor0.02_cap7 | -0.0171 | -0.0660 | -0.1313 | -0.0063 | -0.0477 | +0.0426 | +0.0014 |
| rule_clamp_panel_max_cap7 | -0.0539 | -0.0761 | -0.1398 | -0.0145 | -0.0460 | +0.0180 | -0.0527 |
| box0.05_around_olmix_cap7 | -0.0589 | -0.0663 | -0.1079 | -0.0130 | -0.0273 | -0.0093 | -0.0491 |
| box0.05_around_bank_top5_cap7 | -0.0552 | -0.0762 | -0.1408 | -0.0144 | -0.0446 | +0.0186 | -0.0518 |
| bank_top1 | -0.0122 | -0.0707 | -0.1163 | +0.0114 | -0.0225 | +0.0523 | +0.0103 |
| bank_top2 | -0.0358 | -0.0572 | -0.0779 | +0.0025 | -0.0334 | +0.0309 | -0.0262 |
| bank_top3 | -0.0181 | -0.0683 | -0.1111 | +0.0074 | -0.0230 | +0.0477 | +0.0015 |
| bank_top4 | -0.0206 | -0.0644 | -0.0926 | +0.0146 | -0.0378 | +0.0404 | -0.0052 |
| bank_top5 | +0.0067 | -0.0413 | -0.0659 | -0.0096 | -0.0264 | +0.0083 | +0.0028 |
| bank_top5_mean | -0.0433 | -0.0702 | -0.1052 | -0.0028 | -0.0341 | +0.0258 | -0.0318 |
| bank_replicated_n26 | -0.0358 | -0.0572 | -0.0779 | +0.0025 | -0.0334 | +0.0309 | -0.0262 |
| bank_replicated_n8 | -0.0032 | -0.0462 | -0.0843 | +0.0160 | -0.0263 | +0.0678 | +0.0245 |

- What the bank says (measured, not predicted): the top five coordinates (1.058–1.066) have synthetic QA 0.10–0.20,
  OLMOCR 0.02–0.05, stack 0.18–0.30, CC-high 0.23–0.32, curated 0.13–0.17 and 17–23 effective buckets; the replicated
  frontier centre (26 seeds, 1.0639 ± 0.0041) has synthetic QA 0.13, OLMOCR 0.04, stack 0.20, curated 0.17. Relative
  to it, WSPU cap 7 has too much stack (0.30) and too little curated mass (0.07); OLMix has too much synthetic QA
  (0.33) and OLMOCR (0.13). The successor predicts every one of these coordinates 0.02–0.03 too optimistically and
  still ranks the frontier 10th.
- Recommended for validation, in order: `floor_qa0.2_olmocr0.08_cap7` (synthetic QA 0.20, OLMOCR 0.08, stack 0.30 at
  cap, 16 effective buckets, TV 0.12 to a measured 1.0730 neighbour); `interp_cap7_0.5` (halfway to OLMix, synthetic
  QA 0.23, stack 0.23, no active cap); `floor_qa0.25_olmocr0.13_cap7`. All three keep the predicted code and math gains
  (−0.09 to −0.13 code) while the heads predict the MMLU family flat instead of worse. Two seeds each would resolve
  them against the 0.004 seed SD. The frontier-centre mixture itself is the strongest measured comparator and should
  be the control for any such run.
- Not recommended: caps ≥ 8 (worst matched-seed result, no active cap, largest residual), the share-floor optimum, and
  the box around OLMix (predicted gains come from the same CC spread).

## 8. Caveats

- One matched seed per mixture; per-component seed SD is 0.008–0.014 BPB, macro 0.004. Family-level conclusions are
  well above that; single components (e.g. basic_skills_string_operations, seed gap 0.126) are not.
- Remedies are transforms of the frozen heads. A refit with bank rows in training was tested in round 3 (Table-9
  frontier rank got worse) and not repeated. The pending registry refresh (408 T9 coordinates) will re-run the heldout
  stage automatically; the residual audit should be repeated on it.
- The final model-versus-baseline comparison stays on the 280-run panel; nothing here changes the frozen successor.

## Appendix A: 51-row tables per cap


#### Cap 6: predicted vs observed OLMix-to-WSPU component deltas (BPB; negative = WSPU better)

| component | family | predicted Δ | observed Δ (matched seed) | residual | observed Δ (original seed) | panel repeat SD | systematic regressor |
|---|---|---:|---:|---:|---:|---:|:---:|
| arc_challenge | arc | -0.1150 | +0.0585 | +0.1735 | +0.0516 | 0.0155 | yes |
| arc_easy | arc | -0.0091 | +0.0584 | +0.0675 | +0.0305 | 0.0132 | yes |
| basic_skills_arithmetic | basic_skills | -0.1021 | -0.0357 | +0.0664 | -0.0469 | 0.0274 |  |
| basic_skills_coding | basic_skills | -0.1628 | -0.0682 | +0.0946 | -0.0197 | 0.0270 |  |
| basic_skills_common_knowledge | basic_skills | -0.0309 | +0.0657 | +0.0966 | -0.0041 | 0.0364 | yes |
| basic_skills_logical_reasoning | basic_skills | -0.0029 | +0.0451 | +0.0480 | +0.0410 | 0.0187 | yes |
| basic_skills_pattern | basic_skills | -0.0176 | -0.0174 | +0.0002 | +0.1333 | 0.0531 |  |
| basic_skills_string_operations | basic_skills | -0.0907 | -0.0193 | +0.0713 | -0.0386 | 0.0714 |  |
| codex_humaneval | code | -0.0595 | -0.0325 | +0.0270 | -0.0270 | 0.0185 |  |
| mbpp | code | -0.0999 | -0.0358 | +0.0641 | -0.0262 | 0.0089 |  |
| mt_mbpp_bash | code | -0.2222 | -0.0427 | +0.1795 | -0.0480 | 0.0283 |  |
| mt_mbpp_c | code | -0.1138 | -0.0109 | +0.1030 | -0.0198 | 0.0088 |  |
| mt_mbpp_cpp | code | -0.1063 | -0.0144 | +0.0919 | -0.0264 | 0.0069 |  |
| mt_mbpp_csharp | code | -0.0856 | -0.0124 | +0.0732 | -0.0089 | 0.0048 |  |
| mt_mbpp_go | code | -0.1439 | -0.0398 | +0.1040 | -0.0171 | 0.0098 |  |
| mt_mbpp_haskell | code | -0.1021 | -0.0290 | +0.0731 | -0.0297 | 0.0233 |  |
| mt_mbpp_java | code | -0.0837 | -0.0125 | +0.0711 | -0.0239 | 0.0061 |  |
| mt_mbpp_javascript | code | -0.1369 | -0.0313 | +0.1057 | -0.0329 | 0.0071 |  |
| mt_mbpp_matlab | code | -0.0649 | -0.0134 | +0.0515 | -0.0121 | 0.0150 |  |
| mt_mbpp_php | code | -0.0977 | -0.0221 | +0.0755 | -0.0257 | 0.0123 |  |
| mt_mbpp_python | code | -0.1032 | -0.0325 | +0.0707 | -0.0256 | 0.0088 |  |
| mt_mbpp_r | code | -0.1288 | -0.0116 | +0.1172 | -0.0328 | 0.0099 |  |
| mt_mbpp_ruby | code | -0.1890 | -0.0332 | +0.1557 | -0.0481 | 0.0181 |  |
| mt_mbpp_rust | code | -0.2428 | -0.0456 | +0.1972 | -0.0326 | 0.0328 |  |
| mt_mbpp_scala | code | -0.1187 | -0.0247 | +0.0940 | -0.0351 | 0.0206 |  |
| mt_mbpp_swift | code | -0.1618 | -0.0275 | +0.1343 | -0.0536 | 0.0216 |  |
| mt_mbpp_typescript | code | -0.1185 | -0.0267 | +0.0918 | -0.0403 | 0.0063 |  |
| csqa | commonsense | -0.0052 | -0.0110 | -0.0058 | -0.0888 | 0.0286 |  |
| hellaswag | commonsense | -0.0039 | -0.0056 | -0.0017 | -0.0046 | 0.0016 |  |
| piqa | commonsense | -0.0642 | -0.0175 | +0.0466 | -0.0197 | 0.0076 |  |
| socialiqa | commonsense | +0.0129 | +0.0167 | +0.0038 | +0.0256 | 0.0148 | yes |
| winogrande | commonsense | -0.0351 | -0.0046 | +0.0305 | +0.0219 | 0.0088 |  |
| minerva_math_algebra | math | -0.0473 | -0.0124 | +0.0349 | -0.0123 | 0.0080 |  |
| minerva_math_counting_and_probability | math | -0.0350 | -0.0107 | +0.0242 | -0.0101 | 0.0062 |  |
| minerva_math_geometry | math | -0.0556 | -0.0177 | +0.0380 | +0.0017 | 0.0046 |  |
| minerva_math_intermediate_algebra | math | -0.0425 | -0.0051 | +0.0374 | +0.0011 | 0.0097 |  |
| minerva_math_number_theory | math | -0.0395 | -0.0065 | +0.0331 | -0.0110 | 0.0055 |  |
| minerva_math_prealgebra | math | -0.0427 | -0.0088 | +0.0339 | -0.0122 | 0.0062 |  |
| minerva_math_precalculus | math | -0.0418 | -0.0091 | +0.0327 | +0.0012 | 0.0074 |  |
| mmlu_humanities | mmlu | +0.0062 | +0.0149 | +0.0087 | +0.0271 | 0.0163 | yes |
| mmlu_other | mmlu | +0.0100 | +0.0455 | +0.0355 | +0.0493 | 0.0064 | yes |
| mmlu_social_sciences | mmlu | -0.0752 | +0.0409 | +0.1161 | +0.0370 | 0.0123 | yes |
| mmlu_stem | mmlu | +0.1007 | +0.0110 | -0.0897 | +0.0188 | 0.0106 | yes |
| coqa | qa_reading | -0.1541 | -0.0720 | +0.0821 | -0.0739 | 0.0253 |  |
| drop | qa_reading | -0.1287 | +0.0404 | +0.1691 | -0.0155 | 0.0359 | yes |
| jeopardy | qa_reading | +0.0392 | +0.0750 | +0.0357 | +0.0640 | 0.0249 | yes |
| lambada | qa_reading | +0.0104 | +0.0288 | +0.0184 | +0.0313 | 0.0123 | yes |
| medmcqa | qa_reading | -0.0037 | +0.0586 | +0.0623 | +0.0663 | 0.0161 | yes |
| naturalqs | qa_reading | -0.1692 | +0.0266 | +0.1958 | +0.0565 | 0.0222 | yes |
| sciq | qa_reading | -0.0486 | +0.0591 | +0.1078 | +0.0452 | 0.0093 | yes |
| squad | qa_reading | -0.0722 | -0.0352 | +0.0370 | -0.0196 | 0.0254 |  |

Macro: predicted Δ -0.0744, observed Δ -0.0041; sign agreement 42/51; residual RMS 0.0897.

#### Cap 7: predicted vs observed OLMix-to-WSPU component deltas (BPB; negative = WSPU better)

| component | family | predicted Δ | observed Δ (matched seed) | residual | observed Δ (original seed) | panel repeat SD | systematic regressor |
|---|---|---:|---:|---:|---:|---:|:---:|
| arc_challenge | arc | -0.1075 | +0.0519 | +0.1594 | +0.0521 | 0.0155 | yes |
| arc_easy | arc | +0.0005 | +0.0489 | +0.0484 | +0.0504 | 0.0132 | yes |
| basic_skills_arithmetic | basic_skills | -0.1135 | -0.0906 | +0.0230 | -0.0852 | 0.0274 |  |
| basic_skills_coding | basic_skills | -0.1918 | -0.0928 | +0.0990 | -0.0774 | 0.0270 |  |
| basic_skills_common_knowledge | basic_skills | -0.0259 | +0.0748 | +0.1007 | +0.0359 | 0.0364 | yes |
| basic_skills_logical_reasoning | basic_skills | -0.0064 | +0.0275 | +0.0340 | +0.0492 | 0.0187 | yes |
| basic_skills_pattern | basic_skills | -0.0136 | -0.0149 | -0.0014 | +0.0406 | 0.0531 |  |
| basic_skills_string_operations | basic_skills | -0.1072 | -0.1140 | -0.0068 | +0.0120 | 0.0714 |  |
| codex_humaneval | code | -0.0613 | -0.0444 | +0.0169 | -0.0361 | 0.0185 |  |
| mbpp | code | -0.1040 | -0.0318 | +0.0722 | -0.0299 | 0.0089 |  |
| mt_mbpp_bash | code | -0.2608 | -0.0795 | +0.1813 | -0.0515 | 0.0283 |  |
| mt_mbpp_c | code | -0.1305 | -0.0240 | +0.1065 | -0.0275 | 0.0088 |  |
| mt_mbpp_cpp | code | -0.1140 | -0.0252 | +0.0887 | -0.0277 | 0.0069 |  |
| mt_mbpp_csharp | code | -0.0923 | -0.0149 | +0.0775 | -0.0228 | 0.0048 |  |
| mt_mbpp_go | code | -0.1526 | -0.0486 | +0.1040 | -0.0308 | 0.0098 |  |
| mt_mbpp_haskell | code | -0.1191 | -0.0089 | +0.1102 | -0.0485 | 0.0233 |  |
| mt_mbpp_java | code | -0.0896 | -0.0227 | +0.0670 | -0.0283 | 0.0061 |  |
| mt_mbpp_javascript | code | -0.1559 | -0.0400 | +0.1159 | -0.0365 | 0.0071 |  |
| mt_mbpp_matlab | code | -0.0738 | -0.0033 | +0.0705 | -0.0107 | 0.0150 |  |
| mt_mbpp_php | code | -0.1161 | -0.0397 | +0.0764 | -0.0296 | 0.0123 |  |
| mt_mbpp_python | code | -0.1073 | -0.0330 | +0.0743 | -0.0292 | 0.0088 |  |
| mt_mbpp_r | code | -0.1451 | -0.0341 | +0.1110 | -0.0227 | 0.0099 |  |
| mt_mbpp_ruby | code | -0.2278 | -0.0369 | +0.1909 | -0.0541 | 0.0181 |  |
| mt_mbpp_rust | code | -0.2753 | -0.0499 | +0.2253 | -0.0470 | 0.0328 |  |
| mt_mbpp_scala | code | -0.1208 | -0.0031 | +0.1177 | -0.0473 | 0.0206 |  |
| mt_mbpp_swift | code | -0.1876 | -0.0448 | +0.1427 | -0.0699 | 0.0216 |  |
| mt_mbpp_typescript | code | -0.1401 | -0.0399 | +0.1002 | -0.0321 | 0.0063 |  |
| csqa | commonsense | +0.0002 | -0.0665 | -0.0668 | -0.0292 | 0.0286 |  |
| hellaswag | commonsense | +0.0028 | +0.0009 | -0.0019 | +0.0029 | 0.0016 |  |
| piqa | commonsense | -0.0588 | -0.0123 | +0.0465 | -0.0109 | 0.0076 |  |
| socialiqa | commonsense | +0.0160 | +0.0249 | +0.0089 | +0.0195 | 0.0148 | yes |
| winogrande | commonsense | -0.0307 | +0.0274 | +0.0581 | +0.0261 | 0.0088 |  |
| minerva_math_algebra | math | -0.0508 | -0.0269 | +0.0239 | -0.0183 | 0.0080 |  |
| minerva_math_counting_and_probability | math | -0.0368 | -0.0217 | +0.0150 | -0.0175 | 0.0062 |  |
| minerva_math_geometry | math | -0.0578 | -0.0153 | +0.0426 | +0.0007 | 0.0046 |  |
| minerva_math_intermediate_algebra | math | -0.0451 | -0.0178 | +0.0272 | -0.0015 | 0.0097 |  |
| minerva_math_number_theory | math | -0.0417 | -0.0155 | +0.0262 | -0.0145 | 0.0055 |  |
| minerva_math_prealgebra | math | -0.0456 | -0.0191 | +0.0265 | -0.0151 | 0.0062 |  |
| minerva_math_precalculus | math | -0.0441 | -0.0212 | +0.0230 | -0.0027 | 0.0074 |  |
| mmlu_humanities | mmlu | +0.0124 | +0.0313 | +0.0189 | +0.0335 | 0.0163 | yes |
| mmlu_other | mmlu | +0.0235 | +0.0584 | +0.0349 | +0.0547 | 0.0064 | yes |
| mmlu_social_sciences | mmlu | -0.0697 | +0.0427 | +0.1124 | +0.0439 | 0.0123 | yes |
| mmlu_stem | mmlu | +0.1080 | +0.0142 | -0.0937 | +0.0344 | 0.0106 | yes |
| coqa | qa_reading | -0.1444 | -0.0323 | +0.1121 | -0.0589 | 0.0253 |  |
| drop | qa_reading | -0.1233 | +0.0748 | +0.1981 | +0.0371 | 0.0359 | yes |
| jeopardy | qa_reading | +0.0690 | +0.0413 | -0.0277 | +0.0455 | 0.0249 | yes |
| lambada | qa_reading | +0.0281 | +0.0261 | -0.0020 | +0.0223 | 0.0123 | yes |
| medmcqa | qa_reading | +0.0171 | +0.0757 | +0.0587 | +0.0831 | 0.0161 | yes |
| naturalqs | qa_reading | -0.1604 | +0.0455 | +0.2058 | +0.0680 | 0.0222 | yes |
| sciq | qa_reading | -0.0362 | +0.0879 | +0.1241 | +0.0772 | 0.0093 | yes |
| squad | qa_reading | -0.0647 | -0.0308 | +0.0339 | +0.0256 | 0.0254 |  |

Macro: predicted Δ -0.0779, observed Δ -0.0091; sign agreement 42/51; residual RMS 0.0957.

#### Cap 8: predicted vs observed OLMix-to-WSPU component deltas (BPB; negative = WSPU better)

| component | family | predicted Δ | observed Δ (matched seed) | residual | observed Δ (original seed) | panel repeat SD | systematic regressor |
|---|---|---:|---:|---:|---:|---:|:---:|
| arc_challenge | arc | -0.1021 | +0.0768 | +0.1790 | +0.0608 | 0.0155 | yes |
| arc_easy | arc | +0.0078 | +0.0766 | +0.0688 | +0.0683 | 0.0132 | yes |
| basic_skills_arithmetic | basic_skills | -0.1209 | -0.0704 | +0.0505 | -0.0842 | 0.0274 |  |
| basic_skills_coding | basic_skills | -0.2020 | -0.0798 | +0.1221 | -0.0937 | 0.0270 |  |
| basic_skills_common_knowledge | basic_skills | -0.0219 | +0.1048 | +0.1267 | +0.0441 | 0.0364 | yes |
| basic_skills_logical_reasoning | basic_skills | -0.0083 | +0.0307 | +0.0390 | +0.0609 | 0.0187 | yes |
| basic_skills_pattern | basic_skills | -0.0120 | +0.0130 | +0.0251 | +0.0001 | 0.0531 |  |
| basic_skills_string_operations | basic_skills | -0.1154 | +0.0244 | +0.1398 | -0.0276 | 0.0714 |  |
| codex_humaneval | code | -0.0604 | -0.0446 | +0.0158 | -0.0602 | 0.0185 |  |
| mbpp | code | -0.1044 | -0.0283 | +0.0761 | -0.0368 | 0.0089 |  |
| mt_mbpp_bash | code | -0.2825 | -0.0612 | +0.2213 | -0.0632 | 0.0283 |  |
| mt_mbpp_c | code | -0.1398 | -0.0264 | +0.1134 | -0.0255 | 0.0088 |  |
| mt_mbpp_cpp | code | -0.1174 | -0.0256 | +0.0918 | -0.0399 | 0.0069 |  |
| mt_mbpp_csharp | code | -0.0952 | -0.0167 | +0.0785 | -0.0141 | 0.0048 |  |
| mt_mbpp_go | code | -0.1559 | -0.0496 | +0.1063 | -0.0242 | 0.0098 |  |
| mt_mbpp_haskell | code | -0.1277 | -0.0551 | +0.0725 | -0.0277 | 0.0233 |  |
| mt_mbpp_java | code | -0.0922 | -0.0322 | +0.0600 | -0.0281 | 0.0061 |  |
| mt_mbpp_javascript | code | -0.1667 | -0.0342 | +0.1324 | -0.0363 | 0.0071 |  |
| mt_mbpp_matlab | code | -0.0760 | +0.0091 | +0.0851 | +0.0018 | 0.0150 |  |
| mt_mbpp_php | code | -0.1261 | -0.0349 | +0.0912 | -0.0294 | 0.0123 |  |
| mt_mbpp_python | code | -0.1076 | -0.0321 | +0.0755 | -0.0366 | 0.0088 |  |
| mt_mbpp_r | code | -0.1534 | -0.0304 | +0.1230 | -0.0255 | 0.0099 |  |
| mt_mbpp_ruby | code | -0.2500 | -0.0482 | +0.2017 | -0.0553 | 0.0181 |  |
| mt_mbpp_rust | code | -0.2938 | -0.0719 | +0.2219 | -0.0542 | 0.0328 |  |
| mt_mbpp_scala | code | -0.1198 | -0.0129 | +0.1069 | -0.0442 | 0.0206 |  |
| mt_mbpp_swift | code | -0.2018 | -0.0495 | +0.1523 | -0.0612 | 0.0216 |  |
| mt_mbpp_typescript | code | -0.1524 | -0.0385 | +0.1140 | -0.0389 | 0.0063 |  |
| csqa | commonsense | +0.0037 | -0.0537 | -0.0573 | -0.0333 | 0.0286 |  |
| hellaswag | commonsense | +0.0079 | +0.0103 | +0.0023 | +0.0104 | 0.0016 |  |
| piqa | commonsense | -0.0548 | +0.0140 | +0.0688 | -0.0229 | 0.0076 |  |
| socialiqa | commonsense | +0.0181 | +0.0348 | +0.0167 | +0.0135 | 0.0148 | yes |
| winogrande | commonsense | -0.0277 | +0.0085 | +0.0362 | +0.0437 | 0.0088 |  |
| minerva_math_algebra | math | -0.0515 | -0.0295 | +0.0220 | -0.0125 | 0.0080 |  |
| minerva_math_counting_and_probability | math | -0.0367 | -0.0161 | +0.0206 | -0.0090 | 0.0062 |  |
| minerva_math_geometry | math | -0.0580 | -0.0166 | +0.0413 | -0.0029 | 0.0046 |  |
| minerva_math_intermediate_algebra | math | -0.0449 | -0.0173 | +0.0276 | -0.0007 | 0.0097 |  |
| minerva_math_number_theory | math | -0.0414 | -0.0241 | +0.0173 | -0.0113 | 0.0055 |  |
| minerva_math_prealgebra | math | -0.0461 | -0.0214 | +0.0247 | -0.0092 | 0.0062 |  |
| minerva_math_precalculus | math | -0.0440 | -0.0196 | +0.0244 | -0.0005 | 0.0074 |  |
| mmlu_humanities | mmlu | +0.0163 | +0.0464 | +0.0301 | +0.0135 | 0.0163 | yes |
| mmlu_other | mmlu | +0.0341 | +0.0757 | +0.0416 | +0.0686 | 0.0064 | yes |
| mmlu_social_sciences | mmlu | -0.0662 | +0.0613 | +0.1275 | +0.0545 | 0.0123 | yes |
| mmlu_stem | mmlu | +0.1135 | +0.0578 | -0.0557 | +0.0176 | 0.0106 | yes |
| coqa | qa_reading | -0.1385 | -0.0255 | +0.1130 | -0.0463 | 0.0253 |  |
| drop | qa_reading | -0.1183 | +0.0852 | +0.2035 | +0.0245 | 0.0359 | yes |
| jeopardy | qa_reading | +0.0909 | +0.1525 | +0.0617 | +0.0880 | 0.0249 | yes |
| lambada | qa_reading | +0.0415 | +0.0455 | +0.0041 | +0.0369 | 0.0123 | yes |
| medmcqa | qa_reading | +0.0334 | +0.1060 | +0.0726 | +0.0970 | 0.0161 | yes |
| naturalqs | qa_reading | -0.1544 | +0.0680 | +0.2224 | +0.0824 | 0.0222 | yes |
| sciq | qa_reading | -0.0270 | +0.0837 | +0.1107 | +0.0895 | 0.0093 | yes |
| squad | qa_reading | -0.0592 | +0.0167 | +0.0760 | +0.0140 | 0.0254 |  |

Macro: predicted Δ -0.0786, observed Δ +0.0027; sign agreement 37/51; residual RMS 0.1042.

## Artifacts

- Outputs: `reference_outputs/single_phase_observatory_benchmark_20260902/olmix_gap_round5/` (`component_deltas.csv`,
  `component_predictions.csv`, `bucket_decomposition.csv`, `extrapolation.csv`, `dose_curves_weibull_softplus_unscaled.csv`,
  `remedies_*.csv`, `candidates_*.csv`, `summary.json`).
- Fieldbook: experiment `exp_01m1msmaf8p4wenrgyfxw2cm54`, note `note_01m1q2a10tmjvvnjk84mv5adzw` (input).
