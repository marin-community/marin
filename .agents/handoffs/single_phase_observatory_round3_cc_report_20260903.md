# Single-phase Observatory, round 3: Delphi 3e18 optima (Claude Code report, 2026-09-03)

Fieldbook experiment `exp_01m1ge7ye6hz2epd0mjkbkrvt8`; continues rounds 1-2 (commits 99bea291d7, 954a881eea). Plan:
`.agents/projects/single_phase_round3_plan_20260903.md`. Status: closed 2026-09-03 02:30 except the Table-9 refresh rerun.

## 0. Answers

1. **Can the surrogate be improved, with the extra Delphi data as development evidence, so that its Delphi 3e18
   optimum beats the frontier?** Not with this model family and a panel-only final fit. Every mechanism tried in
   round 3 that keeps the final fit on the 280-run panel is null or negative under an out-of-sample check by source:
   a bank-selected shared shape (regret 0 in sample; split-half +0.0037 / +0.0085 worse than inner CV), coarse grid
   rules (+0.0002 / +0.0025), fixed ensembles (no interval excludes zero), and adding the dose rows to training
   (two cells worse, two better, four ties; no paired interval excludes zero). No tested selector improves on the
   per-component inner CV on the panel.
2. **What limits Uncheatable?** Coverage. Fitted on the panel every model is optimistic by 0.02-0.03 at L1 >= 0.5
   and picks the successor's own cap-4-to-6 sweep runs (regret 0.0023, about three run SDs, all models within
   0.0012 of each other). Fitted with the archive neighbours (leave-one-source-out) the bounded log link is
   calibrated and reaches best-of-5 regret 0 with the frontier 5th. The frontier family (shared-shape DSP epoch caps
   4 to 10) is still improving at cap 10 and is right-censored: extending it to caps 12 and 16 is the cheapest next
   validation and needs no surrogate change.
3. **What limits Table 9?** Not coverage, for the models tested. The frontier region (HPR-280 tied controls and
   neighbours, 1.058-1.069: 12-16 epochs on synth-math, finemath, wikipedia and stem-heavy crawl, 8 % CC-HQ, 8 %
   CC-low) is ranked 10th-35th by the panel fits and 31st-48th once the archive neighbours are in training; the
   four tested additive per-bucket models learn to rank inside a family but not which family holds the frontier.
   Half of the successor's 0.030 error there is over-valued mid-quality CC at 1-2
   epochs, half is harm charged to the high-epoch special buckets. The pending 181 Table-9 dose-response payloads
   will show directly whether those buckets' Table-9 harm onset is later than the panel implies.
4. **Registry state.** 57 dose-response runs were exported with stale W&B summaries (Uncheatable +0.12 to +0.21);
   round-3 tables use a corrected view built from the exact step-3006 files; the materializer fix and the canonical
   regeneration belong to its owner. The epoch-cap sources store Table-9 components under short task names, which
   silently drops them from componentwise scoring in the harness.
5. **No successor is named.** The round-2 candidates stay where they were: the bounded link is the best-calibrated
   model and the best selector once coverage exists, but on a panel-only fit it does not beat the successor's bank
   selection (paired bootstrap intervals include zero on both targets).

## 1. Registry audit: 57 dose-response runs carried stale Uncheatable values

The refreshed registry (`reference_outputs/single_phase_heldout_benchmark_20260902/`, 2026-09-03 00:14) adds 237
fit-panel-disjoint conditional epoch-dose runs (277 raw; the proportional anchor and 39 zero-dose controls overlap the
panel). Auditing the 40 overlaps against the panel's own measurements:

| Raw runs | W&B state | Uncheatable provenance | dose minus panel at the same coordinate |
|---|---|---|---|
| 28 overlaps (193 raw) | finished | W&B summary | +0.0015 +- 0.0014 |
| 5 overlaps (27 raw) | crashed | `eval_metrics.jsonl` at step 3006 | +0.0013 +- 0.0023 |
| 7 overlaps (57 raw) | crashed | W&B summary | **+0.17 +- 0.04 (0.12 to 0.21)** |

The 84 crashed runs were preempted and finished from checkpoints (all 277 have step-3006 exports). The materializer
(`materialize_delphi_bucket_epoch_dose_heldout_20260903.py`, `_uncheatable_candidate`) accepts any complete W&B
summary before looking at the persisted file, so a crashed run whose summary froze at an earlier eval step passes
validation with a stale value. The same region-local files at step 3006 recover all 84 (the 27 already recovered
reproduce exactly; 56 change; 49 of those are eligible heldout rows). Across the whole grid the stale rows sit
+0.12 median above the anchor at every multiplier, where finished runs sit within 0.001.

- Fix: prefer the persisted final-step payload whenever the training run is not `finished` (or whenever it
  exists), then regenerate the canonical registry with the two refresh commands and rerun the heldout stage.
- Update 2026-09-03: the registry owner repaired the materializer (all 84 non-finished runs now require exact
  step-3006 metrics) and canonicalized the Table-9 names; the rebuilt canonical registry (manifest SHA-256
  `6cce727e50cc...`) matches the corrected view on all 779 Uncheatable rows exactly and adds 8 dose-response
  Table-9 rows (food-and-dining-low at multipliers 4, 8, 32; games-high at 0.25 to 8), so Delphi Table-9 coverage
  is 247 coordinates with complete components (34 repaired epoch-cap coordinates, 8 new). The heldout stage and the
  Table-9 scoring were rerun on the canonical registry (section 2a); the corrected view is superseded.
- Before that repair every round-3 table used a corrected view,
  `reference_outputs/single_phase_heldout_round3_corrected_20260903/` (same rows and order, 49 Uncheatable values and
  their 343 component cells replaced), built by `scratchpad/r3_corrected_registry.py` from
  `scratchpad/dose_uncheatable_final_step.csv`. Table 9 is unaffected (separate eval runs on the exported checkpoints).
- Effect on the frozen scores: on the dose stratum the as-exported Uncheatable RMSE was 0.054 to 0.064 for every
  model (the two degenerate references, linear_weight and fold_mean, included) and the Spearman 0.54 to 0.69 for the
  tabular models; corrected, 0.009 to 0.020 and 0.70 to 0.93 (linear_weight 0.026 / 0.53, fold_mean 0.031 / n.a.).
  Selection metrics on the archive stratum are unchanged.

## 2. Frozen heldout refresh (Codex request), corrected view

Heldout stage rerun 2026-09-03 00:30-00:40 for 22 models (16 parents, 2 references, both successors, the two
round-2 link candidates; 3828 fits, all ok). Scoring script:
`single_phase_round3_heldout_selection_20260903.py` (tables under
`reference_outputs/single_phase_observatory_benchmark_20260902/heldout_round3_corrected/`; the as-exported scores
under `heldout_round3_asexported/`).

Coverage: Uncheatable 408 coordinates / 471 runs (237 dose-response, 171 archive); Table 9 239 coordinates / 302
runs (81 dose-response, 158 archive). Frontier: Uncheatable 0.9811 (shared-shape DSP epoch cap 10, L1 0.95 from
the panel); Table 9 1.0579 (HPR-280 tied control, L1 0.68).

| Model | U regret@1 | U best-of-5 | U rank of pick | U frontier predicted rank | T9 regret@1 | T9 best-of-5 | T9 rank of pick | T9 frontier predicted rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| weibull_softplus_unscaled | 0.0023 | 0.0012 | 5 | 6 | 0.0157 | 0.0132 | 14 | 10 |
| @log_deficit_bounded_link | 0.0030 | 0.0016 | 6 | 8 | 0.0143 | 0.0143 | 10 | 35 |
| @link_by_cv | 0.0035 | 0.0012 | 7 | 7 | 0.0143 | 0.0143 | 10 | 22 |
| weibull_softplus_shared | 0.0030 | 0.0012 | 6 | 7 | 0.0143 | 0.0143 | 10 | 24 |
| dsp_total_exposure | 0.0035 | 0.0023 | 7 | 9 | 0.0151 | 0.0132 | 12 | 16 |
| dsp_total_exposure_concentration | 0.0030 | 0.0012 | 6 | 8 | 0.0143 | 0.0132 | 10 | 20 |
| bucket_family_power_grp | 0.0023 | 0.0023 | 5 | 7 | 0.0157 | 0.0132 | 14 | 10 |
| weibull_family_grp_shared_onset | 0.0030 | 0.0023 | 6 | 11 | 0.0151 | 0.0143 | 12 | 17 |
| olmix_loglinear_taskwise | 0.0086 | 0.0012 | 22 | 9 | 0.0190 | 0.0190 | 24 | 38 |
| grp_pair_power | 0.2092 | 0.0103 | 408 | 45 | 0.1067 | 0.0132 | 141 | 6 |
| random ranking (expectation) | 0.0505 | 0.0191 | | | 0.0889 | 0.0276 | | |

- Paired bootstrap over coordinates (2000 draws): on Uncheatable no model's regret@1 is better than the successor's
  (canonical DSP +0.0007 [0.0000, 0.0033] and GRP pairs are worse); on Table 9 DSP-concentration is -0.0007
  [-0.0014, 0.0000] and the CV link -0.0007 [-0.0014, +0.0023] (both intervals include zero). Best-of-5 regret: no
  interval excludes zero in either direction; the models that come closest (canonical DSP, the family models on
  Uncheatable) are worse, not better. The coordinate bootstrap resamples the bank itself; the fixed-bank measurement
  bootstrap (run noise added to every coordinate, `selection_measurement_bootstrap.csv`, archive stratum) agrees:
  against the successor, Uncheatable regret@1 differences are +0.0007 [-0.0017, +0.0029] (bounded link), +0.0012
  [-0.0012, +0.0035] (CV link, canonical DSP) and +0.0063 [+0.0040, +0.0086] (OLMix, worse); Table 9 -0.0015
  [-0.0099, +0.0067] for the bounded link, the CV link and DSP-concentration. Nothing excludes zero in the
  successor's disfavour.
- Bias at L1 > 0.75 from the panel: successor -0.033 (U) / -0.030 (T9); bounded link -0.005 / +0.010; CV link
  -0.017 / -0.001; canonical DSP -0.027 / -0.054; OLMix -0.011 / +0.009. Calibration far from the panel does not
  translate into selection: the bounded link ranks the Table-9 frontier 35th.
- The historical Table-9 best of 1.05753 (`origstyle_sep_t9_1p_kl0p075_3e18`, also registered as
  `dphase_t9b075_tied_3e18`, one training run) is in the bank and eligible, but it is one seed of the "frontier
  centre" coordinate that the phase-fiber, random-phase and aggressive-asymmetry studies reran 26 times with
  different seeds: mean 1.06385, SD 0.0041 (SE 0.0008), best seed 1.0553, worst 1.0717. At coordinate level that
  makes it the second-best Table-9 coordinate after the single-run HPR-280 control (1.05789, n = 1), which is
  itself one seed draw. Only 7 of the 239 Table-9 coordinates are replicated. Against the replicated frontier the
  successor's pick (1.0736, n = 1) has regret 0.0098 and the link candidates 0.0083, not 0.0143-0.0157; the models
  rank the 26-seed centre 14th-30th (successor 18th, predicted 1.039 against 1.004 for its own pick, a relative
  error of 0.045 against the measured +0.010 in the centre's favour). Seed noise at this coordinate (0.0041) is
  larger than the 0.003 Table-9 repeat SD used elsewhere in the harness.
- Dose-response stratum (near the anchor): every model except GRP pairs selects the measured best Uncheatable
  coordinate (regret 0); Table 9 regret 0.0015 (rank 4) for all.
- The pick is the same neighbourhood for every model: the successor's own epoch-cap runs (caps 4-6) on Uncheatable
  and caps 6-8 on Table 9; the true frontier (shared-shape DSP caps 4-10 for Uncheatable, the HPR-280 tied controls
  for Table 9) is ranked 6th-35th.

## 2a. Rerun on the repaired canonical registry

Heldout stage (22 models, 3828 fits) and scoring rerun on the canonical registry (manifest SHA-256 `6cce727e50cc...`),
tables under `heldout_round3_canonical/`. Every Uncheatable number in section 2 is reproduced exactly (408 / 171 /
237 coordinates). Table 9 now has 247 coordinates (158 archive, 89 dose-response); the archive stratum is
unchanged to the last digit, the pooled stratum keeps every pick and regret (successor 0.0157, rank 14, frontier
10th; bounded and CV links and DSP-concentration 0.0143, rank 10), and the dose stratum keeps regret 0.0015 (rank
4) for every model with RMSE 0.016-0.032 over 89 coordinates.

Componentwise scoring on the coordinates with complete components (harness stratum `component_complete_subset`,
mean over components of the per-component RMSE / Spearman):

| Model | Table 9 (247 coordinates, 51 components) | Uncheatable (403 coordinates, 7 components) |
|---|---|---|
| @log_deficit_bounded_link | 0.0368 / 0.79 | 0.0165 / 0.93 |
| @link_by_cv | 0.0391 / 0.79 | 0.0191 / 0.93 |
| olmix_loglinear_taskwise | 0.0478 / 0.75 | 0.0271 / 0.79 |
| weibull_softplus_unscaled | 0.0530 / 0.78 | 0.0237 / 0.92 |
| bucket_family_power_grp | 0.0671 / 0.76 | 0.0211 / 0.90 |
| dsp_total_exposure_concentration | 0.0731 / 0.74 | 0.0264 / 0.86 |
| dsp_total_exposure | 0.0755 / 0.72 | 0.0310 / 0.82 |

The two link candidates are the best componentwise predictors on both targets (the bounded link's Table-9
component RMSE is 30 % below the successor's), which restates the round-2 finding: calibration and selection are
different properties, and on this bank the better-calibrated links do not select better.

## 3. Why the frontier is misranked

Per-bucket decomposition of the successor (panel fit) at the Table-9 frontier against its own cap-8 pick: predicted
1.034 against 1.004 (measured 1.0579 against 1.0736). Total benefit -0.563 against -0.578, total harm +0.056 against
+0.041. Half of the 0.030 error is benefit: the pick loads literature-high (7 %), history-high (3.4 %), food-low,
industrial-low and 17 % cc-low in total at 1-2 epochs, which the model values above the frontier's common-crawl HQ
(8 %), finance/health/entertainment-high (3 % each). Half is harm: synth-math (11 epochs), finemath (6.7),
synth-instruction (4.8), synth-thinking (7.7) are charged 0.015 more than at the pick. Uncheatable shows the same
pattern: the DSP cap-10 frontier holds 12 % common-crawl HQ and 16 % science-math-high; the successor's cap-6 pick
holds 4 % and 12 % with more olmocr and stack-edu at 5 epochs; predicted 0.9556 against 0.9471, measured 0.9811
against 0.9834.

## 4. Dose-response anatomy (corrected values; `single_phase_round3_dose_anatomy_20260903.py`)

The successor and canonical DSP were refitted on the panel per component (heldout inner folds) and evaluated on the
237 dose-response runs that are not panel coordinates; predictions are decomposed into the benefit and harm blocks
relative to the anchor. The anchor and the 39 zero-dose controls are panel coordinates, so their residuals are
in-sample fit (successor -0.0015 Uncheatable, -0.0052 Table 9; `dose_anatomy_in_sample_rows.csv`) and are kept out
of every accuracy table below.

- Near the anchor the successor is accurate: Uncheatable residual -0.001 to -0.002 (RMSE 0.002-0.003) for
  multipliers 0.25 to 8; Table 9 (83 of its 96 runs are at multipliers 0.25 to 8) -0.005 to +0.000 (RMSE 0.003-0.009).
- At multipliers 16 and 32 (a single bucket at 15-29 epochs and up to 50 % weight) both models are optimistic:
  successor residual -0.008 / -0.025 on Uncheatable (RMSE 0.014 / 0.038), -0.022 / -0.073 on Table 9; DSP
  -0.008 / -0.020 and -0.013 / -0.062. Calibration regression of the measured change on the predicted benefit and
  harm changes: Uncheatable 1.51 x benefit + 1.25 x harm, Table 9 1.27 x benefit + 3.0 x harm (1.0 = calibrated).
- The optimism is concentrated in the Common Crawl buckets: at their largest dose the measured damage is 2-3 times
  the prediction (industrial-high +0.124 measured against +0.047 predicted, food-low +0.117 / +0.052, games-low
  +0.127 / +0.062, olmocr +0.058 / +0.020), because the panel never exposes them beyond 1.5-22 epochs so their harm
  amplitudes are unidentified. The special buckets that the panel does expose to 20-50 epochs (arxiv, finemath,
  wikipedia, stem-heavy crawl, synth code/instruction/math/thinking) are calibrated or slightly pessimistic at 29
  epochs (residual -0.001 to +0.006).
- Deletion direction (multiplier 0): these 39 controls are panel rows, so the comparison is between the measured
  deletion effect and the successor's in-sample fit at the same coordinate, not a forecast. Even in sample the fit
  misplaces bucket values: on Uncheatable it over-values stack-edu (+0.024 fitted loss on deletion against +0.019
  measured), stack-edu-fim (+0.022 / +0.019), synth-code (+0.010 / +0.007) and finemath (+0.003 / +0.000), and
  under-values arxiv (+0.002 / +0.004) and science-math-high (+0.002 / +0.003); deleting common-crawl HQ improves
  Uncheatable by 0.0035 (fitted 0.0065). On Table 9 the fitted deletion values are off by 0.005 in both directions
  (crime-low +0.006 measured / +0.002 fitted, art-low +0.007 / -0.000, electronics-low -0.001 / +0.005). The
  out-of-sample counterpart, multiplier 0.25, has residuals of -0.002 (Uncheatable) and +0.000 (Table 9).
- Relevance to the frontier: the frontier mixtures do not push any bucket outside the epoch range the panel covers
  for it; their distance from the panel is a distance in composition (L1 0.7-0.95), which an additive model
  cannot see. The next section tests whether an additive fit that has seen the frontier's neighbourhood ranks it.

## 5. Development regimes: fitting on the bank (`single_phase_round3_union_loso_20260903.py`)

Five regimes scored on the archive stratum (Uncheatable 171 coordinates in 16 source groups, Table 9 158 in 15;
coordinates without complete components are test-only rows; the epoch-cap sources' short Table-9 component names
are remapped; a coordinate is held out with every source it belongs to). Two views: pooled (all out-of-fold
predictions ranked together, which is what an optimum search sees) and within source (selection scored inside each
held-out source, then averaged, with a paired bootstrap over sources against panel_only).

| Regime | training rows added | Uncheatable pooled regret@1 / best-of-5 / frontier rank (successor; bounded link) | Table 9 pooled |
|---|---|---|---|
| panel_only | none | 0.0023 / 0.0012 / 6; 0.0030 / 0.0016 / 8 | 0.0157 / 0.0132 / 10; 0.0143 / 0.0143 / 35 |
| panel_dose | 237 dose-response coordinates | 0.0030 / 0.0012 / 7; 0.0068 / 0.0012 / 8 | 0.0143 / 0.0143 / 25; 0.0143 / 0.0143 / 37 |
| loso | dose + every archive source but the held-out one | 0.0030 / 0.0023 / 11; **0.0016 / 0.0000 / 5** | 0.0168 / 0.0143 / 45; 0.0168 / 0.0132 / 46 |
| dose_only | dose runs alone, no panel | 0.0219 / 0.0174 / 137; 0.0092 / 0.0068 / 73 | 0.0793 / 0.0517 / 111; 0.0793 / 0.0394 / 97 |
| dose_holdout | archive; test on the dose stratum | 0.0172 / 0 / 2; 0.0000 / 0 / 1 | 0.0003 / 0.0003 / 12; 0.0003 / 0.0003 / 12 |

Within-source means (regret@1 / frontier rank inside its source) and the paired difference against panel_only
over sources (2000 draws):

| Regime | Uncheatable successor | Uncheatable bounded link | Table 9 successor | Table 9 bounded link |
|---|---|---|---|---|
| panel_only | 0.0008 / 1.9 | 0.0011 / 2.1 | 0.0033 / 2.3 | 0.0077 / 2.6 |
| panel_dose | 0.0006 / 1.6; -0.0002 [-0.0004, 0.0000] | 0.0014 / 2.3; +0.0003 [-0.0004, +0.0010] | 0.0046 / 2.3; +0.0013 [-0.0012, +0.0038] | 0.0072 / 2.5; -0.0006 [-0.0016, 0.0000] |
| loso | 0.0012 / 2.1; +0.0004 [-0.0003, +0.0011] | 0.0006 / 1.75; **-0.0005 [-0.0011, -0.0000]** | 0.0058 / 2.5; +0.0026 [-0.0011, +0.0063] | 0.0052 / 2.3; -0.0026 [-0.0057, +0.0002] |
| dose_only | 0.0100 / 5.3; +0.0092 [+0.0053, +0.0124] | 0.0066 / 3.7; +0.0055 [+0.0028, +0.0079] | 0.0389 / 7.4; +0.0357 [+0.0257, +0.0450] | 0.0370 / 6.9; +0.0292 [+0.0209, +0.0373] |

- Uncheatable: with the frontier's neighbours in training the bounded log link is calibrated far from the panel
  (bias +0.0000 at L1 >= 0.5 against -0.0265 for the panel-fitted successor), reaches pooled best-of-5 regret 0 with
  the frontier 5th, and is the only regime-model pair whose within-source improvement excludes zero (-0.0005
  [-0.0011, -0.0000], better in 4 sources, worse in 1). The successor and DSP do not improve with the same data
  (pooled frontier rank 11). Coverage limits Uncheatable selection for the bounded link; the identity-link models do
  not use the extra coverage.
- Adding the dose-response coordinates to the panel is a wash: pooled regret@1 changes in two of eight cells for the
  worse (successor and bounded link on Uncheatable), two for the better (CV link and DSP on Uncheatable, successor on
  Table 9 by 0.0014), the rest tie; no within-source paired interval excludes zero. The frontier rank never improves.
- Table 9: with the neighbours in training the pooled frontier rank of every tested model gets worse (31st-48th
  against 10th-35th), while within-source selection is a wash for the successor (+0.0026 [-0.0011, +0.0063]) and
  improves for the bounded link (-0.0026 [-0.0057, +0.0002]). The bank data teach the tested models to rank inside
  a family, not which family holds the frontier; the frontier region (section 3) stays misranked. For the four
  models tested this is not a coverage limit; whether a different additive form would transfer is untested.
- Rerun on the repaired canonical registry (Table 9 only; 89 dose-response coordinates instead of 81): every
  pooled archive number above is reproduced, and the paired-over-sources differences move within their intervals
  (successor loso +0.0020 [-0.0020, +0.0055], bounded link loso -0.0026 [-0.0056, +0.0002], successor panel_dose
  +0.0026 [-0.0005, +0.0056]); tables under `heldout_round3_canonical/`.
- The dose runs alone (no panel) predict nothing in the archive region (pooled frontier ranks 73-149, within-source
  regret 0.007-0.039): single-bucket curves around the anchor plus additivity do not carry to the frontier
  composition, though extrapolation and additivity are confounded in this check.

## 6. Development-data selection of the successor's hyperparameters

### 6a. Fixed shared shape chosen on the bank (`single_phase_round3_shape_scan_20260903.py`)

Every (shape, ridge, link) of the successor's grid (168 x 5 x 2 = 1680 rows) with amplitudes fitted on the panel
and one shared shape across components, scored on the archive stratum of the corrected bank:

- Uncheatable: the frozen inner-CV model has regret 0.0023 (frontier ranked 6th); the fixed shape rate 0.1, power
  1.0, threshold 1.0 (ridge 0 or 1e-3, identity link) has regret 0 and ranks the frontier first; the twelve best
  rows all have rate 0.1-0.25, power 1.0, threshold 1-3. Only 7 % of the 1680 rows match or beat the frozen regret.
- Table 9: frozen regret 0.0157 (frontier 10th); the best fixed rows (rate 0.05, power 0.3, threshold 5-6, ridge
  1.0) reach regret 0.0082 with the frontier 2nd, at the price of a +0.06 bias and RMSE 0.07; 31 % of rows match or
  beat the frozen regret.
- **Out of sample the selection does not survive** (`single_phase_round3_shape_selection_check_20260903.py`, 200
  random halves of the archive sources, coordinates whose sources straddle the halves dropped; the row chosen on
  one half is scored on the other): Uncheatable regret 0.0047 for the chosen shape against 0.0010 for the frozen
  model (difference +0.0037 [-0.0003, +0.0253], chosen better in 14 % of splits), frontier rank 10.9 against 3.1;
  Table 9 0.0177 against 0.0092 (+0.0085 [-0.0157, +0.0910], better in 16 %), frontier rank 10.5 against 6.5. A
  regret-0 row among 1680 is a selection artifact, not a better model.

### 6b. Coarse grid rules (`single_phase_round3_grid_rules_20260903.py`)

Fifteen low-dimensional rules (restrict the harm threshold, rate, power or ridge range, or choose the link) with the
per-component inner CV still selecting inside the rule; the frozen rule reproduces the successor's bank predictions
to 3e-8. In sample no rule beats the frozen model on Uncheatable (threshold >= 4 or power <= 0.5 are much worse:
regret 0.0078, frontier 23rd-52nd); on Table 9 the link rules and threshold <= 2 reach 0.0143 against 0.0157 with
intervals touching zero. Out of sample (rule chosen on half the sources, scored on the other half): Uncheatable
+0.0002 [+0.0000, +0.0012] against the frozen rule (the frozen rule is chosen in 160 of 200 splits); Table 9
+0.0025 [-0.0014, +0.0161], the chosen rule better in 9 % of splits. No tested selector (one shared-shape search,
fifteen coarse rules) shows a transferable improvement over the full-grid per-component inner CV on the panel; the
split-half intervals include both signs, so the evidence is that none of these choices helps, not that none can.

### 6c. Fixed ensembles (`single_phase_round3_ensembles_20260903.py`)

Mean, mean-z-score and mean-rank ensembles of {successor, both links}, the five best tabular models, and
{successor, DSP, OLMix}, scored on the archive stratum: none beats the successor on Uncheatable (all 0.0023-0.0035);
on Table 9 the rank ensemble of successor + DSP + OLMix reaches 0.0132 against 0.0157 (paired bootstrap
-0.0016 [-0.0025, 0.0000], better in 86 % of draws) but ranks the frontier 16th against 10th. Among all 22 frozen
models the best Table-9 frontier rank is 6 (GRP pairs, whose own pick is 0.107 worse) and 9 (family-onset GRP).

## 7. Proposals (`single_phase_round3_proposals_20260903.py`, `single_phase_round3_proposal_predictions_20260903.py`)

The successor's identity-link aggregate is separable by bucket, so its constrained optimum on the exact 1/2048
mixture grid is solved by min-plus dynamic programming with per-bucket count bounds: an epoch cap and, optionally,
a box of half-width 0.02 or 0.05 in weight around the measured-best bank coordinate (the frontier). Checks: with cap
6 (Uncheatable) and cap 8 (Table 9) and no box the programme returns exactly the successor's epoch-cap sweep points
that are in the bank (nearest bank L1 = 0), so the search reproduces the user's sweep.

- Uncheatable: the unconstrained optimum saturates at cap 6 (predicted 0.9471; measured 0.9834 in the sweep); the
  box proposals around the DSP cap-10 frontier (L1 0.24-0.32 from it) are predicted 0.9475-0.9484.
- Table 9: the optimum saturates at cap 8 (predicted 1.0039; measured 1.0736); box proposals around the HPR-280
  frontier (L1 0.33-0.43) are predicted 1.0044-1.0078. A 0.02 box is infeasible below cap 8 because the frontier
  itself needs 16 epochs on some buckets.
- Five panel-fitted models (successor, both links, DSP-concentration, OLMix) were refitted and asked for the gain
  of each proposal over the frontier. All five agree that every Table-9 proposal beats the frontier by 0.003-0.030
  and that the Uncheatable cap-4 proposals beat it by 0.0005-0.006; OLMix dissents on the Uncheatable cap >= 6
  proposals. Agreement is worthless here: the cap-8 Table-9 "proposal" is a measured bank point that is 0.016 worse
  than the frontier, and the cap-6 Uncheatable one is 0.0023 worse, so the shared optimism of every model at its own
  proposal region (section 2: +0.03 to +0.05 absolute) swamps the predicted gains.
- What the bank does say: the Uncheatable frontier family (shared-shape DSP epoch caps 4, 6, 8, 10 = 0.9827, 0.98226,
  0.98232, 0.9811) is flat from cap 4 to 8 and improves at cap 10, its last tested cap, so it is right-censored;
  the successor's family peaks at cap 6 (caps 3-8: 0.9879, 0.9841, 0.9846, 0.9834, then worse). Extending the DSP family to caps 12 and 16
  is a search-space question that needs no new surrogate. The Table-9 frontier region (HPR-280 tied controls and
  their neighbours, 1.058-1.069) is not reachable by any epoch-capped optimum of the frozen models: their picks in
  that region are ranked 10th-41st and the region needs 12-16 epochs on synth-math, finemath, wikipedia and
  stem-heavy crawl, which every model charges 0.01-0.02 of harm for.

## 8. Reviews

Codex (gpt-5.6-sol, max reasoning) and DeepSeek (deepseek-v4-pro, max reasoning) reviewed the nine round-3 scripts
and this report independently (briefs and raw outputs under the session scratchpad `reviews/`). Every finding was
checked against the code and the output tables before it was acted on.

| Finding | Verified | Disposition |
|---|---|---|
| Codex P1: the dose anatomy scores the anchor and the 39 zero-dose controls, which are panel coordinates, as if out of sample; the deletion-direction analysis is in-sample | yes | anatomy script flags `in_panel` rows, excludes them from the multiplier, calibration and high-dose tables, and reports the zero-dose fit separately as an in-sample residual; section 4 rewritten |
| Codex P1: LOSO pools predictions from different fits into one global ranking, so source-specific calibration shifts can decide the argmin and frontier rank; no paired uncertainty for regime comparisons (also P2) | yes | union harness now scores each held-out source separately and reports the within-source mean plus a paired bootstrap over sources against panel_only; the pooled ranking is kept as a second view; section 5 rewritten |
| Codex P1: semicolon-delimited multi-source coordinates were treated as one label, so a coordinate could stay in training while one of its sources was held out; the split-half checks had the same flaw | yes (5 coordinates, including the second-best Table-9 one) | memberships parsed; a coordinate is held out with every source it belongs to and pooled under its primary source; the split-half checks drop coordinates whose sources straddle the halves; test added |
| Codex P1: defaults pointed at the stale canonical registry while writing into a `corrected` directory, and the correction lived in the scratchpad | yes | `--registry-dir`, `--output-dir` and `--recovery` are required; each output records the registry path and manifest hash; the recovery and the corrected-view builder are versioned as `single_phase_round3_registry_correction_20260903.py` and the recovery table lives in the view |
| Codex P2: the coordinate bootstrap resamples the bank, so it measures sensitivity to a resampled candidate set rather than uncertainty for the fixed proposal set | yes | a fixed-bank measurement bootstrap (run noise added to every coordinate) is reported alongside; both are labelled |
| Codex P2: the structural claim for Table 9 is scoped to the four tested models; the dose-fitted additivity check was planned but not run | yes | a `dose_only` regime (dose runs alone predicting the archive) added; wording changed from "cannot represent" to "the tested additive models do not transfer" |
| Codex P2: "best selector available" over-claims | yes | reworded to "no tested selector shows a transferable improvement" |
| Codex P3: the DSP cap family is not monotone (cap 8 is 6e-5 worse than cap 6) | yes | reworded: flat from 4 to 8, improving at 10, right-censored |
| DeepSeek B1: "panel_dose is worse for every model" is false (2 of 8 cells worse, 4 ties, 2 better) | yes | section 5 and answer 1 rewritten from the rerun tables |
| DeepSeek B2: the CV link's Table-9 interval was misquoted | yes | corrected to -0.0007 [-0.0014, +0.0023] |
| DeepSeek B3: best-of-5 "no interval excludes zero except ..." | yes | corrected: no interval excludes zero in either direction |
| DeepSeek B4: dose-stratum ranges claimed "every model" but excluded the two degenerate references | yes | ranges restated with the references named |
| DeepSeek B5: four of the five component-less Uncheatable coordinates are the frontier family, not three | yes | corrected |
| DeepSeek N1: the dose_holdout row was not reproducible from the current artifacts | yes | regenerated in the rerun |
| DeepSeek N2/N3: Table-9 box L1 range; "96 runs" phrasing | yes | corrected |
| DeepSeek N4/N5: overlap audit, frontier decomposition and the "first pass" sentence were not reproducible from the tree | yes | decomposition versioned (`single_phase_round3_frontier_decomposition_20260903.py`), overlap audit lives in the registry-correction script's output, sentence dropped; "+0.12 to +0.21" now stated as the overlap subset, with 36 of the 49 replaced rows shifting by more than 0.01 |
| DeepSeek N6: 12 weight cells differ at 1e-25 after the CSV round trip | yes | benign; noted in the view's manifest |
| DeepSeek N7/N8: silent drop of unmapped component names; dead constant | yes | the harness now raises on an unmapped name; constant removed |
| DeepSeek N9: extreme-order-statistic bootstraps are fragile | yes | caveat in section 9; measurement bootstrap added |

## 9. Limitations and open items

- The corrected registry view is superseded by the repaired canonical registry (identical Uncheatable values,
  plus 8 Table-9 rows). 173 Table-9 dose-response payloads are still pending; the Table-9 dose anatomy (section 4)
  covers 96 runs and 15 buckets.
- Every bank number in sections 5-7 is development evidence: the archive coordinates were used to select, score and
  refute candidates. The only prospective evidence of the round is the frozen-model refresh in section 2 restricted
  to the 237 corrected dose-response coordinates, which no model had seen before.
- The split-half checks resample sources, not coordinates, so their intervals are wide (12-17 archive sources per
  target); they establish that the bank-selected choices do not transfer, not a precise loss.
- Regret at 1 and best-of-5 are extreme order statistics; the coordinate bootstrap changes the candidate set on
  every draw and can count a coordinate twice in the top five, and the fixed-bank measurement bootstrap only moves
  the measured values by run noise. Neither is a full account of fit uncertainty; read intervals that touch zero as
  ties.
- The union regimes use mixture-blocked inner folds over the union rows and hold out whole sources; a coordinate that
  appears in two sources counts under its first source. Coordinates without complete components are test-only.
- The dynamic programme needs an identity link, so the bounded-link proposals were not searched; a local search under
  the non-separable aggregate is the missing piece if the bounded link is ever used to propose.
- No proposal was validated. The right next validation runs are the shared-shape DSP family at epoch caps 12 and 16
  (Uncheatable, right-censored family) and, for Table 9, perturbations of the HPR-280 tied control rather than any
  surrogate optimum.
