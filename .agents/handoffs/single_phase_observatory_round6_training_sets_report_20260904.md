# Round 6: a different 280-row training set from sampled and intervention runs (2026-09-04)

Question: within the 280-run budget, can a training set drawn from every Delphi 3e18 run that was sampled or designed
as an intervention (no surrogate- or OLMix-proposed optimum) produce a surrogate that finds better Table-9 and
Uncheatable optima than the panel does? Nothing was launched; all evidence is refits on existing runs, scored on the
archive of model-proposed coordinates that no design may train on. This version supersedes the first draft of the
same day: the Codex review showed that the adversarial stress panel is surrogate-selected, so it left the training pool
and every run was repeated (section 6).

## 0. TLDR

- **No.** Within the 280-row budget, no training set drawn from the eligible runs improves the optimum the successor
  selects, on either target; no other model improves consistently either. Table-9 regret@1 moves 0.0157 → 0.0143–0.0151
  (one neighbour inside the WSPU sweep, paired interval covering zero in every design) and the frontier's predicted
  rank worsens (10th → 13th–23rd); Uncheatable regret moves 0.0023 → 0.0023–0.0078 with the frontier 7th–22nd.
- **What the eligible rows give**: better calibration where they sit. Held-out-intervention RMSE falls (Table 9 0.025
  → 0.020–0.024; Uncheatable 0.009 → 0.007–0.011) and the successor's optimism on the optima shrinks (Uncheatable
  bias −0.022 → −0.010 to −0.021; Table 9 −0.025 → −0.017 to −0.027). The pools over-expose 10 (Table 9) to 17
  (Uncheatable) of the 26 Common Crawl buckets beyond the panel maximum and reach 14 and 12 effective buckets.
- **What they do not give**: selection. Within the 157 Table-9 optima the refit still ranks the OLMix coordinate
  88th–105th (measured 24th) and its own sweep 1st–3rd. The over-budget fit with the panel plus every pool row (326 /
  400 rows) selects the same coordinates, so for this eligible pool the budget is not what binds; the pool is.
- **Why**: every run near the Table-9 frontier is a surrogate, OLMix, or validation optimum and stays evaluation-only
  under the rule; the dose ladders sit around a proportional-like panel row (Table-9 macro 1.19–1.38, TV 0.44 from the
  frontier, 0.30–0.38 from the replicated centre). Refitted on them, the successor's cap-7 optimum stays where the
  panel put it (Table 9: stack 0.26–0.30, CC 0.40–0.45, 20–24 effective buckets, predicted 1.000–1.017 against 1.005).
- **What would change the answer**: eligible-by-rule runs near the frontier, i.e. designed perturbations (dose
  ladders, deletions, a TV-0.1 ring) around the replicated Table-9 centre and around OLMix. These are interventions, not
  optima, and none exist. The pending registry refresh adds 148 dose rows with Table 9; the armed watcher reruns this
  script on it, and since the anchor region is unchanged no change in the answer is expected.

## 1. What is eligible

Registry sources (canonical registry, manifest `6cce727e…`, Delphi 3e18, 39 buckets), classified by provenance. The
registry's `proposal_model` field marks the dose runs as a design (`designed_conditional_epoch_dose`) and the
surrogate sweeps and panels as proposals; sources with an empty field were classified from their generator or name.

| source | coordinates (U / T9) | provenance | role |
|---|---|---|---|
| 280-run panel (`singleavg_fit_*`) | 280 / 280 | proportional perturbations and deletions | eligible (training) |
| conditional_epoch_dose_response | 237 / 89 (237 after the pending refresh) | designed single-bucket dose ladders around the panel's proportional anchor row | eligible |
| delphi_baseline_mixtures_issue6607_20260623 | 1 / 1 | baseline mixture | eligible |
| delphi_3e18_adversarial_stress_panel_20260716 | 12 / 12 | looks sampled (empty `proposal_model`), but its generator keeps only coordinates that a frozen surrogate predicted at or beyond the frontier (`materialize_delphi_3e18_adversarial_heldout_panel.py`) | evaluation only |
| weibull_softplus_unscaled / shared_shape_dsp / full_canonical_dsp / aggregate_v epoch-cap sweeps | 12+11+16+8 | surrogate optima | evaluation only |
| hpr_300m_to_3e18, hpr_3e18_to_3e18 optimum validation panels | 10+10 | `one_phase_aggregate_path` optima | evaluation only |
| delphi-corrective-hpr-280-tied-controls, hybrid_phase_ordering, aggressive_phase_asymmetry, frontier_phase_fiber, frontier_random_phase_population | 6+8+2+2+2 | tied-control, anchor and centre-control optima (the replicated Table-9 centre) | evaluation only |
| delphi_one_phase_olmix_kl_sweep, olmix_scaling | 16+1 | OLMix outputs | evaluation only |
| symmetric_sepheads_geometry_frontier, original_style_matched_sepheads_ablation, sep_frontier_tied, uncheatable_optimized, uncheatable_validation, table9 dsp kl sweep, gamma_capped_bowl, decoupled_phase_information, best_phase_model, the 2026-07-05 table9 validations | 18+12+2+9+2+5+3+3+8+7 | surrogate optima or their validations | evaluation only |

Coordinates can carry several sources (the replicated centre carries five), so the per-source counts exceed the 170 /
157 optima coordinates. Eligible non-panel rows: 238 for Uncheatable, 90 for Table 9 today. What they add:

- Common Crawl over-exposure: across the full eligible set every one of the 26 CC buckets exceeds the panel's maximum
  epochs (median 2.4×, up to 29 epochs); stack reaches 14.5 epochs (panel 7), curated 29; concentrated rows reach 9.4
  effective buckets (panel minimum 21.9). The seeded halves that form the training pools keep 17 (Uncheatable) and 10
  (Table 9) CC buckets beyond the panel maximum and 11.7 / 14.2 effective buckets at the minimum.
- What they do not add: rows near the Table-9 frontier. The dose anchor is the panel's proportional row (`p000`,
  in the panel, 0.9 epochs per bucket at multiplier 1); the dose rows lie within TV 0.01–0.43 of it, at Table-9 macro
  1.19–1.38 (Uncheatable 1.02–1.17), TV 0.44 from the Table-9 frontier and 0.30–0.38 from the replicated centre.

## 2. Protocol

- Evaluation set, fixed per target: the "optima" stratum (all model-proposed archive coordinates: 170 U / 157 T9,
  carrying the frontier, the OLMix coordinate and the WSPU sweep) plus a seeded random half of the eligible registry
  rows ("interventions held out": 118 U / 44 T9). Training designs draw from the panel and the other half of the
  eligible rows (pool: 120 U / 46 T9).
- Designs, all ≤ 280 rows: `panel_280` (reference); `swap_pruned_coverage_k` for k = 40, 80, 120, capped at the pool
  size and deduplicated (Table 9: 40 and 46), removing the k panel rows closest to another training row and adding k
  pool rows by farthest-point sampling in TV; `swap_random_{k}_seed{0,1,2}` (random removals and additions, k = 80 U /
  46 T9, for the design's own uncertainty); `coverage_280` (farthest-point sampling over panel ∪ pool from the most
  spread row); `pool_first_280` (every pool row plus coverage-selected panel rows); and an over-budget reference,
  the panel plus every pool row (400 U / 326 T9), to show what the budget itself costs.
- Models refitted per component on each design: `weibull_softplus_unscaled`, its bounded log-deficit link, the DSP
  concentration model and the OLMix log-linear taskwise baseline (inner folds by weight blocks over the training rows
  only, as in the round-3 union harness). Scores: regret@1, best-of-5 regret, frontier predicted rank, Spearman, bias,
  RMSE on the optima stratum; RMSE, bias, Spearman on the held-out interventions; paired coordinate bootstrap (1000
  draws) of each design against `panel_280` on identical rows. For the successor, the cap-7 min-plus optimum of each
  design (aggregated with the target's component weights) with its nearest measured neighbours.
- Script: `single_phase_round6_training_sets_20260904.py`; outputs in
  `reference_outputs/single_phase_observatory_benchmark_20260902/training_sets_round6{,_reference,_table9,_table9_reference}/`,
  each with a `provenance.json` (registry manifest hash, script hash, eligible sources, seed; written after the fact
  for these four runs, by the script from now on).

## 3. Uncheatable

Optima stratum = 170 model-proposed coordinates (Uncheatable frontier 0.9811, the `shared_shape_dsp` cap-10 run);
interventions held out = 118; pool = 120. Paired bootstrap against `panel_280`, 1000 draws, negative = better.

| design | rows: panel / dose / other | CC buckets beyond panel max; min eff. buckets | WSPU regret@1 / best-of-5 / frontier rank / bias / RMSE | WSPU held-out interventions RMSE | WSPU Δregret vs panel [95% CI], P(better) | DSP-conc regret / frontier rank | bounded link regret / rank | OLMix log-linear regret / rank |
|---|---|---|---|---|---|---|---|---|
| panel_280 | 280 / 0 / 0 | 0; 21.9 | 0.0023 / 0.0012 / 6 / -0.022 / 0.029 | 0.009 | reference | 0.0030 / 8 | 0.0030 / 8 | 0.0086 / 9 |
| swap_pruned_coverage_40 | 240 / 40 / 0 | 16; 11.7 | 0.0030 / 0.0023 / 20 / -0.011 / 0.019 | 0.007 | +0.0005 [-0.0005, +0.0069], 0.15 | 0.0030 / 11 | 0.0068 / 8 | 0.0078 / 13 |
| swap_pruned_coverage_80 | 200 / 79 / 1 | 17; 11.7 | 0.0030 / 0.0023 / 17 / -0.014 / 0.020 | 0.007 | +0.0007 [-0.0005, +0.0055], 0.15 | 0.0030 / 6 | 0.0068 / 17 | 0.0078 / 15 |
| swap_pruned_coverage_120 | 160 / 119 / 1 | 17; 11.7 | 0.0078 / 0.0030 / 22 / -0.013 / 0.019 | 0.009 | +0.0041 [-0.0005, +0.0066], 0.05 | 0.0035 / 14 | 0.0068 / 15 | 0.0144 / 15 |
| swap_random_80_seed0 | 200 / 80 / 0 | 16; 11.7 | 0.0030 / 0.0012 / 7 / -0.012 / 0.025 | 0.011 | +0.0000 [-0.0023, +0.0033], 0.43 | 0.0023 / 13 | 0.0068 / 6 | 0.0566 / 18 |
| swap_random_80_seed1 | 200 / 80 / 0 | 14; 11.7 | 0.0023 / 0.0023 / 10 / -0.021 / 0.025 | 0.007 | +0.0003 [+0.0000, +0.0066], 0.00 | 0.0035 / 9 | 0.0068 / 6 | 0.0144 / 16 |
| swap_random_80_seed2 | 200 / 79 / 1 | 11; 11.7 | 0.0023 / 0.0023 / 8 / -0.014 / 0.020 | 0.007 | +0.0002 [+0.0000, +0.0055], 0.00 | 0.0030 / 10 | 0.0068 / 25 | 0.0566 / 17 |
| coverage_280 | 243 / 37 / 0 | 16; 11.7 | 0.0030 / 0.0023 / 15 / -0.010 / 0.019 | 0.008 | +0.0005 [-0.0005, +0.0077], 0.15 | 0.0078 / 11 | 0.0068 / 10 | 0.0078 / 16 |
| pool_first_280 | 160 / 119 / 1 | 17; 11.7 | 0.0030 / 0.0023 / 9 / -0.015 / 0.020 | 0.007 | +0.0006 [-0.0005, +0.0055], 0.15 | 0.0030 / 10 | 0.0068 / 31 | 0.0144 / 15 |
| panel_plus_pool_400_over_budget | 280 / 119 / 1 | 17; 11.7 | 0.0030 / 0.0023 / 15 / -0.012 / 0.018 | 0.008 | +0.0005 [-0.0005, +0.0055], 0.15 | 0.0035 / 9 | 0.0068 / 10 | 0.0030 / 13 |

- The successor's pick stays inside the WSPU cap sweep (0.9834 or 0.9841) under every design except the 120-row
  pruned swap, which picks a 0.9889 coordinate (regret 0.0078). Its optimism halves and its ordering does not improve:
  frontier rank 6th → 7th–22nd, paired regret differences +0.0000 to +0.0041 with every interval covering zero or
  positive.
- No other model gains. The DSP concentration model's regret is 0.0023–0.0078 across designs (0.0030 under the panel;
  the 0.0012 seen in the first draft came from adversarial-panel rows in the pool and is gone). The OLMix log-linear
  baseline improves only under the over-budget reference (0.0086 → 0.0030, P(better) 0.74, interval [−0.0070,
  +0.0014]).
- Successor cap-7 optimum per design (nearest measured neighbours as id: TV: BPB):

| design | predicted (weighted) | synth QA | OLMOCR | stack | CC | eff. buckets | max epochs | TV to nearest panel row | nearest measured (id: TV: BPB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| panel_280 | 0.9471 | 0.073 | 0.158 | 0.204 | 0.440 | 16.7 | 5.92 | 0.463 | c77261f6:0.000:0.9834; 736347a1:0.002:0.9846; dbb99b4e:0.072:0.9841 |
| swap_pruned_coverage_40 | 0.9530 | 0.063 | 0.085 | 0.178 | 0.542 | 23.8 | 5.89 | 0.369 | 62c14c95:0.118:0.9889; 25c23e68:0.120:0.9897; 19850a58:0.123:0.9882 |
| swap_pruned_coverage_80 | 0.9488 | 0.101 | 0.086 | 0.175 | 0.526 | 22.5 | 6.73 | 0.365 | 62c14c95:0.143:0.9889; 25c23e68:0.144:0.9897; ff9fa514:0.148:0.9878 |
| swap_pruned_coverage_120 | 0.9412 | 0.083 | 0.074 | 0.172 | 0.582 | 22.9 | 6.73 | 0.370 | 25c23e68:0.131:0.9897; 62c14c95:0.131:0.9889; ff9fa514:0.137:0.9878 |
| swap_random_80_seed0 | 0.9398 | 0.058 | 0.065 | 0.167 | 0.524 | 21.7 | 6.52 | 0.368 | 25c23e68:0.138:0.9897; 62c14c95:0.148:0.9889; ff9fa514:0.154:0.9878 |
| swap_random_80_seed1 | 0.9430 | 0.104 | 0.104 | 0.195 | 0.504 | 18.3 | 5.92 | 0.426 | c77261f6:0.149:0.9834; 736347a1:0.150:0.9846; dbb99b4e:0.154:0.9841 |
| swap_random_80_seed2 | 0.9542 | 0.080 | 0.139 | 0.188 | 0.475 | 20.1 | 6.52 | 0.414 | c77261f6:0.111:0.9834; 736347a1:0.112:0.9846; dbb99b4e:0.126:0.9841 |
| coverage_280 | 0.9518 | 0.063 | 0.083 | 0.168 | 0.525 | 23.7 | 6.73 | 0.355 | 25c23e68:0.110:0.9897; 62c14c95:0.110:0.9889; ff9fa514:0.116:0.9878 |
| pool_first_280 | 0.9468 | 0.101 | 0.095 | 0.175 | 0.538 | 21.1 | 6.73 | 0.391 | dbb99b4e:0.161:0.9841; ad3ef173:0.167:0.9879; 25c23e68:0.173:0.9897 |
| panel_plus_pool_400_over_budget | 0.9536 | 0.090 | 0.085 | 0.171 | 0.528 | 22.3 | 5.33 | 0.366 | 25c23e68:0.111:0.9897; 62c14c95:0.114:0.9889; ff9fa514:0.120:0.9878 |

  Under the panel the optimum is the measured 0.9834 coordinate. Under the augmented designs it moves to CC 0.47–0.58
  and 18–24 effective buckets, TV 0.11–0.17 from any measured coordinate, with nearest measured values 0.983–0.990:
  more spread, further from data, no evidence it is better.

## 4. Table 9

Optima stratum = 157 model-proposed coordinates (frontier 1.0579, the HPR-280 tied control; the replicated centre
1.0639; the OLMix KL=0.005 cap-4 coordinate 1.0769; the WSPU sweep 1.0722–1.0747); interventions held out = 44; pool =
46 (the 89 dose rows with Table 9 today, halved, plus the baseline row).

| design | rows: panel / dose / other | CC buckets beyond panel max; min eff. buckets | WSPU regret@1 / best-of-5 / frontier rank / bias / RMSE | WSPU held-out interventions RMSE | WSPU Δregret vs panel [95% CI], P(better) | DSP-conc regret / frontier rank | bounded link regret / rank | OLMix log-linear regret / rank |
|---|---|---|---|---|---|---|---|---|
| panel_280 | 280 / 0 / 0 | 0; 21.9 | 0.0157 / 0.0132 / 10 / -0.025 / 0.038 | 0.025 | reference | 0.0143 / 20 | 0.0143 / 35 | 0.0190 / 38 |
| swap_pruned_coverage_40 | 240 / 39 / 1 | 10; 14.2 | 0.0143 / 0.0143 / 22 / -0.020 / 0.033 | 0.021 | -0.0007 [-0.0014, +0.0023], 0.71 | 0.0143 / 24 | 0.0143 / 38 | 0.0643 / 54 |
| swap_pruned_coverage_46 | 234 / 45 / 1 | 10; 14.2 | 0.0151 / 0.0132 / 13 / -0.024 / 0.037 | 0.023 | -0.0002 [-0.0007, +0.0000], 0.43 | 0.0143 / 25 | 0.0143 / 39 | 0.0643 / 54 |
| swap_random_46_seed0 | 234 / 45 / 1 | 10; 14.2 | 0.0151 / 0.0143 / 16 / -0.024 / 0.035 | 0.024 | -0.0004 [-0.0014, +0.0023], 0.57 | 0.0143 / 21 | 0.0143 / 24 | 0.0272 / 42 |
| swap_random_46_seed1 | 234 / 45 / 1 | 10; 14.2 | 0.0151 / 0.0143 / 16 / -0.017 / 0.032 | 0.020 | -0.0003 [-0.0014, +0.0023], 0.57 | 0.0151 / 5 | 0.0143 / 26 | 0.0190 / 32 |
| swap_random_46_seed2 | 234 / 45 / 1 | 10; 14.2 | 0.0151 / 0.0132 / 15 / -0.027 / 0.038 | 0.022 | -0.0003 [-0.0007, +0.0000], 0.43 | 0.0157 / 11 | 0.0143 / 36 | 0.0643 / 51 |
| coverage_280 | 256 / 23 / 1 | 10; 14.2 | 0.0151 / 0.0132 / 17 / -0.019 / 0.032 | 0.023 | -0.0002 [-0.0007, +0.0000], 0.43 | 0.0418 / 20 | 0.0143 / 35 | 0.0190 / 45 |
| pool_first_280 | 234 / 45 / 1 | 10; 14.2 | 0.0151 / 0.0132 / 17 / -0.022 / 0.035 | 0.022 | -0.0004 [-0.0014, +0.0000], 0.57 | 0.0151 / 13 | 0.0143 / 39 | 0.0190 / 49 |
| panel_plus_pool_326_over_budget | 280 / 45 / 1 | 10; 14.2 | 0.0143 / 0.0132 / 23 / -0.024 / 0.033 | 0.022 | -0.0007 [-0.0014, +0.0000], 0.71 | 0.0143 / 20 | 0.0143 / 38 | 0.0190 / 49 |

- Every augmented design moves the successor's pick from the WSPU cap-8 coordinate (1.0736) to the cap-6 (1.0722) or
  cap-7 (1.0730) coordinate: one neighbour, at most 0.0014, with every paired interval covering zero. The frontier's
  predicted rank worsens from 10th to 13th–23rd. The DSP concentration model keeps its pick (1.0722) under most designs,
  collapses under `coverage_280` (regret 0.042, picking a 1.0997 cap-sweep coordinate), and under one random swap
  ranks the frontier 5th while still picking 1.0730.
- Predicted rank (measured rank) within the 157 optima, successor:

| design | OLMix KL0.005 cap4 | WSPU cap 6 | WSPU cap 7 | WSPU cap 8 | T9 frontier (HPR-280) | replicated centre |
|---|---|---|---|---|---|---|
| panel_280 | 87 (24) | 3 (10) | 2 (12) | 1 (14) | 10 (1) | 18 (2) |
| swap_pruned_coverage_40 | 100 (24) | 1 (10) | 2 (12) | 3 (14) | 22 (1) | 13 (2) |
| swap_pruned_coverage_46 | 100 (24) | 3 (10) | 1 (12) | 2 (14) | 13 (1) | 18 (2) |
| swap_random_46_seed0 | 99 (24) | 2 (10) | 1 (12) | 3 (14) | 16 (1) | 10 (2) |
| swap_random_46_seed1 | 105 (24) | 2 (10) | 1 (12) | 3 (14) | 16 (1) | 14 (2) |
| swap_random_46_seed2 | 88 (24) | 3 (10) | 1 (12) | 2 (14) | 15 (1) | 19 (2) |
| coverage_280 | 105 (24) | 3 (10) | 1 (12) | 2 (14) | 17 (1) | 18 (2) |
| pool_first_280 | 99 (24) | 2 (10) | 1 (12) | 3 (14) | 17 (1) | 16 (2) |
| panel_plus_pool_326_over_budget | 88 (24) | 1 (10) | 2 (12) | 3 (14) | 23 (1) | 14 (2) |

  The augmented fits still prefer their own sweep to OLMix by a wide margin (OLMix 88th–105th, measured 24th) and
  still cannot place the frontier (13th–23rd) or the centre (10th–19th).
- Successor cap-7 optimum per design:

| design | predicted (weighted) | synth QA | OLMOCR | stack | CC | eff. buckets | max epochs | TV to nearest panel row | nearest measured (id: TV: BPB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| panel_280 | 1.0046 | 0.124 | 0.038 | 0.296 | 0.398 | 20.1 | 6.99 | 0.394 | e86fef97:0.000:1.0730; e429a20f:0.033:1.0736; b1f11612:0.048:1.0722 |
| swap_pruned_coverage_40 | 1.0102 | 0.130 | 0.036 | 0.260 | 0.436 | 21.4 | 6.94 | 0.376 | b1f11612:0.038:1.0722; e86fef97:0.054:1.0730; e43b733d:0.073:1.0747 |
| swap_pruned_coverage_46 | 1.0023 | 0.127 | 0.035 | 0.296 | 0.403 | 20.2 | 6.99 | 0.389 | e86fef97:0.029:1.0730; e429a20f:0.049:1.0736; b1f11612:0.061:1.0722 |
| swap_random_46_seed0 | 1.0096 | 0.115 | 0.041 | 0.274 | 0.419 | 20.8 | 6.94 | 0.397 | e86fef97:0.071:1.0730; b1f11612:0.072:1.0722; e429a20f:0.093:1.0736 |
| swap_random_46_seed1 | 1.0165 | 0.096 | 0.046 | 0.267 | 0.450 | 23.5 | 6.94 | 0.344 | b1f11612:0.083:1.0722; e86fef97:0.088:1.0730; e429a20f:0.105:1.0736 |
| swap_random_46_seed2 | 0.9998 | 0.131 | 0.038 | 0.296 | 0.397 | 19.6 | 6.99 | 0.399 | e86fef97:0.062:1.0730; e429a20f:0.085:1.0736; b1f11612:0.089:1.0722 |
| coverage_280 | 1.0119 | 0.117 | 0.034 | 0.288 | 0.424 | 20.5 | 6.94 | 0.383 | e86fef97:0.047:1.0730; e429a20f:0.061:1.0736; b1f11612:0.072:1.0722 |
| pool_first_280 | 1.0067 | 0.125 | 0.035 | 0.280 | 0.421 | 20.8 | 6.94 | 0.379 | e86fef97:0.038:1.0730; b1f11612:0.053:1.0722; e429a20f:0.062:1.0736 |
| panel_plus_pool_326_over_budget | 1.0106 | 0.144 | 0.040 | 0.259 | 0.422 | 20.2 | 6.94 | 0.390 | b1f11612:0.057:1.0722; e86fef97:0.074:1.0730; e43b733d:0.082:1.0747 |

  With only dose rows added, the optimum stays where the panel put it: synthetic QA 0.10–0.14, stack 0.26–0.30, CC
  0.40–0.45, 20–24 effective buckets, predicted 1.000–1.017 (panel 1.005), within TV 0.03–0.09 of the WSPU sweep's
  measured coordinates (1.072–1.075). The measured best region (stack 0.18–0.30, curated 0.13–0.17, CC-high 0.23–0.32,
  17–23 effective buckets) is not where it goes.

## 5. Answer and caveats

- Under the rule "sampled coordinates or interventions only", the available data cannot move the selected optimum:
  the eligible rows are all far from the frontier (dose anchor TV 0.44 from it), and the additive heads keep crediting
  spread after seeing them. They improve calibration where they sit (CC over-exposure) and nowhere else; the sixth
  demonstration on this bank that calibration and selection are different properties.
- For this eligible pool the budget is not what binds: the over-budget fit with every pool row selects the same
  coordinates on both targets. This is one over-budget fit, not an upper bound over all subsets; the DSP model shows the
  refits are not monotone in the training set.
- Data that would be eligible and would matter: designed perturbations around the replicated Table-9 centre and
  around the OLMix mixture (single-bucket ladders in stack, synthetic QA, OLMOCR, curated; deletions; a TV-0.1 ring).
  These are interventions by construction and would give the heads support where the optimum lives. That is the
  round-4 "ring around the centre" request, unchanged.
- Caveats: Table 9 sees 89 of the 237 dose rows today (46 in the pool); the armed watcher reruns the Table-9 designs
  when the refresh lands (`training_sets_round6_table9_refreshed`). The evaluation strata are model-optimum
  coordinates, which is the population the question is about, but they are also the rows the panel-fitted models were
  selected on in rounds 1–4; a design that only re-ranks within the WSPU sweep cannot show up as a gain here. Design
  uncertainty was probed with three random swaps and a pruned/random pair; no seed changes the conclusion.

## 6. Reviews

- Codex (P1): the adversarial stress panel is surrogate-selected; removed from eligibility, all four runs repeated;
  the first draft's Uncheatable DSP gain (0.0012 under two coverage designs) disappeared with it. (P1) The headline had
  claimed "no model improves" while DSP had; the current numbers support the claim and the text now says what each
  model does. (P2) Capped swap sizes deduplicated; the Uncheatable proposal value now uses the component weights; the
  coverage claims now describe the actual pools; the Table-9 seed exception (cap-7 pick), the proposal ranges and the
  budget statement are qualified; best-of-5 value corrected.
- DeepSeek (B1): artifacts were regenerated during its review and carried no provenance; each output directory now
  has `provenance.json` with the registry manifest hash and script hash. Its numeric notes on the first draft (P
  range, "three coincide", "every augmented design", top-5 value, proposal ranges, anchor distances, the omitted
  uncheatable-validation source, duplicate designs, missing `predictions.csv`) are all resolved by the rerun and this
  rewrite; the bootstrap `Bank` run counts remain a placeholder the coordinate bootstrap does not read. A test now
  pins design composition (pool-only rows, budget, deduplicated names, over-budget reference = panel ∪ pool).

Artifacts: `single_phase_round6_training_sets_20260904.py`; tests `tests/test_single_phase_round6_training_sets_20260904.py`;
outputs as listed in section 2.
