# Comparator proposals at 3e18 (materialized 2026-09-09)

Trained-proposal validation of the complexity ladder (`../complexity_ladder_20260909/`): the mixture each
alternative surrogate proposes when fitted on the frozen Qwen3 3e18 swarm and optimized without a cap or penalty,
to be trained on the seed-matched validation panel (Uncheatable at data seed 666200, suite at 662009).
Script: `materialize_delphi_comparator_proposals_20260909.py`; launcher:
`experiments/domain_phase_mix/launch_delphi_comparator_proposals_3e18.py` (thirteen runs, ids 7,440,000+ and
7,440,100+). The natural spline joined on 2026-09-09 after its knots were bound to the swarm. Fieldbook experiment `exp_01m21f631n71w9xe9n6krpr2kw`.

## Protocol (prespecified in Fieldbook before the proposals were computed)

- Fits: every model on all 280 runs with the harness's heldout-stage protocol (certify inner folds from k-means on
  sqrt weights at seed 20,350,902, proportional run pinned to training, the frozen floor anchors and noise margins).
- Smooth models (quadratic and spline under the floor and log link, Hellinger kernel ridge, power-one MARINER): the reference
  implementation's optimizer (`mixture-selection/mixture_selection.py`): SLSQP from the proportional mixture and
  four seeded swarm rows, polished, rounded to the 1/2048 grid with the exchange search.
- Trees (`lightgbm_regmix`): RegMix's recipe, one million candidates from the swarm's sampling law (uniform on
  the simplex, seed 20,260,909), the 128 best predictions averaged, rounded to the grid.
- Power-one MARINER: the reference implementation with `SHAPES` restricted to power 1. Guard: the unrestricted
  reference fit reproduces `lwspu_u_snc_cap06` and `lwspu_t9_snc_cap08` exactly (TV 0) before any proposal.
- The candidate ids carry a nominal cap (the launcher requires one); every proposal lies strictly inside it.

## Proposals

| candidate | max epochs | active buckets | effective buckets | TV to proportional | TV to MARINER | own prediction | MARINER's prediction |
|---|---|---|---|---|---|---|---|
| cmp_u_quad_cap08 | 7.36 | 21 | 10.5 | 0.547 | 0.138 | 0.9796 | 0.9843 |
| cmp_u_spline_cap08 | 7.24 | 20 | 10.3 | 0.561 | 0.184 | 0.9825 | 0.9861 |
| cmp_u_lgbm_cap08 | 6.78 | 39 | 34.9 | 0.387 | 0.515 | 1.0138 | 0.9982 |
| cmp_u_krr_cap12 | 9.62 | 37 | 20.1 | 0.548 | 0.439 | 0.9509 | 1.0001 |
| cmp_u_mk1_cap06 | 5.25 | 24 | 12.2 | 0.515 | 0.000 | 0.9810 | 0.9810 |
| cmp_t9_quad_cap12 | 11.19 | 20 | 10.5 | 0.678 | 0.218 | 1.0205 | 1.0867 |
| cmp_t9_spline_cap12 | 10.87 | 21 | 10.8 | 0.660 | 0.247 | 1.0310 | 1.0900 |
| cmp_t9_lgbm_cap12 | 9.26 | 39 | 34.3 | 0.383 | 0.314 | 1.1220 | 1.0861 |
| cmp_t9_krr_cap16 | 15.15 | 38 | 25.0 | 0.485 | 0.212 | 1.0250 | 1.0819 |
| cmp_t9_mk1_cap08 | 7.64 | 35 | 18.3 | 0.527 | 0.049 | 1.0647 | 1.0642 |

MARINER's own proposals predict 0.9810 (Uncheatable; measured 0.9814, repeats mean 0.982) and 1.0631 (suite;
measured 1.068 mean of three). Full cross-prediction and distance tables: `cross_predictions_*.csv`,
`tv_between_proposals_*.csv`; per-task hyperparameters: `fits.json`; optimizer records: `summary.json`.

## Readings before training

- Every comparator predicts its own proposal below MARINER's, and MARINER predicts every comparator's proposal
  above its own (suite: quadratic 1.087, trees 1.086, kernel ridge 1.082 against 1.063). The runs decide which
  side is calibrated; the quadratic's suite proposal is the sharpest test (own 1.020 against MARINER's 1.087).
- The spline's proposals lie near the quadratic's (TV 0.12 and 0.23) and are a little less optimistic about themselves
  (0.9825 and 1.031); MARINER predicts 0.986 and 1.090 at them.
- The kernel ridge is the most optimistic about itself (0.951 on Uncheatable against 1.003 at MARINER's mixture)
  and drives one bucket to 15 epochs on the suite.
- RegMix's recipe does not return the trees' own minimizer: the trees predict 1.102 at MARINER's suite mixture
  and 1.122 at their averaged proposal (best single candidate 1.119). The proposal is diffuse (effective
  buckets 34 of 39), which is what averaging uniform-simplex candidates yields. An exchange-refined tree proposal
  is possible as an extra run but was not prespecified.
- The Uncheatable power-one proposal is the frozen mixture itself (every Uncheatable task already selects power 1);
  the suite one differs by TV 0.049 and predicts 1.0647 for itself. Only the suite twin trains.

## Run plan (13 runs, approved 2026-09-09)

Uncheatable: quadratic at trainer seeds 0, 1, 2; spline, trees and kernel ridge at seed 0. Suite: quadratic at seeds
0, 1, 2; spline, trees, kernel ridge and power-one MARINER at seed 0. `submission/launch_command.sh` (parent 4GB because
8GB parents have pended on scheduler memory since 2026-09-08), east5 safety check in `submission/`.

## Measured runs (all thirteen by 2026-09-11; Iris parent succeeded)

`collect_delphi_3e18_validation_results_20260906.py --launch comparator_proposals` writes `measured_results.csv`;
`summarize_delphi_comparator_proposals_20260909.py` pairs every run with MARINER's run at the same data and trainer
seed (frozen-procedure validation seed 0, fairness repeats seeds 1 and 2) and writes `paired_results.csv` and the
appendix rows `paired_rows.tex`; `plot_comparator_proposal_diagnostics_20260909.py` (run with `--with lightgbm`) draws the support and
extrapolation-path figures into `diagnostics/` (the paper calls these models baselines; the calibration figure was
dropped on 2026-09-11 because the paired table carries its numbers).

| proposal | measured | paired vs MARINER | own prediction | MARINER's prediction |
|---|---|---|---|---|
| MARINER, Uncheatable | 0.9825 +- 0.0010 (3 seeds) | -- | 0.981 | 0.981 |
| olmixq_u_kl0p05_cap04 (matched Olmix, cap 4, KL 0.05) | 1.0022 +- 0.0014 (3 seeds) | +0.020 +- 0.001 | 0.994 | 1.012 |
| cmp_u_quad_cap08 | 0.9862 +- 0.0003 (3 seeds) | +0.004 +- 0.001 | 0.980 | 0.984 |
| cmp_u_spline_cap08 | 0.9855 | +0.004 | 0.982 | 0.986 |
| cmp_u_lgbm_cap08 | 1.0003 | +0.019 | 1.014 | 0.998 |
| cmp_u_krr_cap12 | 1.0050 | +0.024 | 0.951 | 1.000 |
| cmp_u_mk1_cap06 (MARINER's mixture; MARINER's runs) | 0.9825 +- 0.0010 (3 seeds) | 0 | 0.981 | 0.981 |
| MARINER, suite | 1.0678 +- 0.0052 (3 seeds) | -- | 1.063 | 1.063 |
| olmixq_t9_kl0p005_cap04 (matched Olmix, cap 4, KL 0.005) | 1.0922 +- 0.0040 (3 seeds) | +0.024 +- 0.002 | 1.095 | 1.123 |
| cmp_t9_quad_cap12 | 1.0905 +- 0.0026 (3 seeds) | +0.023 +- 0.003 | 1.020 | 1.087 |
| cmp_t9_spline_cap12 | 1.0834 | +0.015 | 1.031 | 1.090 |
| cmp_t9_lgbm_cap12 | 1.0868 | +0.019 | 1.122 | 1.086 |
| cmp_t9_krr_cap16 | 1.0810 | +0.013 | 1.025 | 1.082 |
| cmp_t9_mk1_cap08 | 1.0693 | +0.001 | 1.065 | 1.064 |

Every comparator's mixture loses to MARINER's at the same seeds. MARINER's prediction of each comparator mixture is
within 0.007 BPB of the measurement; the comparators' own predictions miss by 0.035 to 0.070 on the suite (smooth
models optimistic, trees pessimistic) and by up to 0.054 on Uncheatable. MARINER is pessimistic about the capped,
diffuse matched-Olmix mixtures (predicts 1.012 / 1.123 against measured 1.002 / 1.092). The power-one twin reproduces MARINER
within 0.001. Paper: `app:nonlinear-comparators` (results paragraph, `tab:a-comparator-proposals`, `fig:a-comparator-paths`)
and the Section 3.4 sentence, pushed to Overleaf on 2026-09-11.

## Materialization notes

Two earlier attempts segfaulted inside libomp: LightGBM's OpenMP runtime and scikit-learn's (the inner folds'
k-means) are separate copies, and two active runtimes crash in one process or in spawned workers
(`materialize_attempt*.log`). The script now pins `OMP_NUM_THREADS=1` and `OMP_THREAD_LIMIT=1` before any import;
single-threaded LightGBM fits take about three seconds instead of thirty.
