# Three-component optimum of the frozen procedure (Delphi 3e18, 2026-09-08)

MARINER's Uncheatable optimum (`lwspu_u_snc_cap06`, three-seed means) lowers GitHub C++ (0.907 → 0.747), GitHub
Python (0.880 → 0.729), arXiv physics (1.092 → 1.029) and arXiv computer science (1.003 → 0.971) against the
proportional mixture (eleven runs), leaves AO3 unchanged (1.2331 → 1.2333, repeat SD 0.001) and worsens BBC News
(1.042 → 1.085) and Wikipedia (1.101 → 1.116). This package re-targets the frozen procedure at the byte-weighted
aggregate of those three components and trains its unconstrained optimum.

- `objectives.csv`, `anchors.csv`: the `uncheatable_worsened` objective (AO3 0.4486, BBC News 0.3137,
  Wikipedia 0.2378, the Uncheatable byte shares renormalized) and the three components' proportional anchors,
  copied from the standalone `mixture-selection/data/`.
- `fit_uncheatable_worsened.json`: `mixture_selection.py fit --objective uncheatable_worsened` (the frozen
  procedure; all three tasks take the flat-profile multiplier 1.5). `fit_uncheatable.json`: the bundled
  Uncheatable objective refitted for the cross predictions.
- `policy_uncheatable_worsened.csv`: `mixture_selection.py optimize` with no cap and no KL: 21 nonzero buckets,
  most repeated bucket 5.63 epochs (literature high), both code sources at zero, total variation 0.51 to the full
  Uncheatable optimum.
- `candidate_weights.csv` (sha256 `6bee06fe…`): the runtime table for
  `launch_delphi_three_component_optimum_3e18.py`, candidate `lwspu_w3_snc_cap06` (the cap suffix is the nominal
  bound above 5.63 epochs); data seed 666200, trainer seeds 0/1/2, v6e-8, run ids 7,406,000 + 10 t.
- `mixtures_for_prediction.csv`, `predictions_uncheatable.csv`, `predictions_uncheatable_worsened.csv`: the two
  objectives predicted at proportional, the full Uncheatable optimum and the three-component optimum:

| mixture | predicted Uncheatable | predicted three-component aggregate |
|---|---|---|
| proportional | 1.0376 | 1.1416 |
| full Uncheatable optimum | 0.9810 | 1.1473 |
| three-component optimum | 1.1082 | 1.1360 |

Measured values arrive through `collect_delphi_3e18_validation_results_20260906.py --launch three_component_optimum`.

## Measured (2026-09-08, Iris `/calvinxu/dm-delphi-3e18-three-component-optimum-v6e8-20260908`, three trainer seeds, no preemptions)

| mixture | AO3 | BBC News | Wikipedia | three-component aggregate (predicted) | GitHub C++ | GitHub Python | full Uncheatable (predicted) | OBE Easy mean |
|---|---|---|---|---|---|---|---|---|
| proportional (11 runs) | 1.233 | 1.042 | 1.101 | 1.142 (1.142) | 0.907 | 0.880 | 1.038 (1.038) | 1.198 |
| full Uncheatable optimum (3 seeds) | 1.233 | 1.085 | 1.116 | 1.159 (1.147) | 0.747 | 0.728 | 0.982 (0.981) | 1.091 |
| three-component optimum (3 seeds) | 1.178 | 1.032 | 1.084 | 1.110 ± 0.001 (1.136) | 1.523 | 1.433 | 1.208 ± 0.001 (1.108) | 1.439 ± 0.004 |

Per seed: three-component aggregate 1.1105 / 1.1092 / 1.1107; full aggregate 1.2071 / 1.2077 / 1.2093; OBE Easy 1.4431 / 1.4355 / 1.4384.
The surrogate was conservative on its own objective (measured 0.026 better than predicted) and optimistic by 0.10 on the full aggregate, where both code sources sit at zero weight, outside the sampled range.
