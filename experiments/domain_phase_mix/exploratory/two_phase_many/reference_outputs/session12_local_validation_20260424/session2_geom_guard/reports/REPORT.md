# v31 continuation: S2 geometry plus directional reason guard

## Primary result

The best continuation candidate in this archive is `s2_geom_reason_guard_67`.

It keeps the S2-tweak scale skeleton,

`A(w) + a_N u_N + B(w) u_D + a_ND u_ND`,

and adds a positive geometry term:

`lambda_family * G_family(w) * (D/D0)^(-eta_family) + lambda_support * max(0, TV(w, S_train) - rho)^2 + lambda_reason * max(0, p0_reason - r)^2 / r^2`.

The free count is 67 parameters: the 61-parameter S2 tweak plus six guard scalars. The conservative serialized count is 79 if the six family floors and six family ceilings are counted as constants.

## Key metrics

| model                   | split               | params_free | params_serialized_conservative | rows | rmse     | bias      | slope    | std_ratio |
| ----------------------- | ------------------- | ----------- | ------------------------------ | ---- | -------- | --------- | -------- | --------- |
| s2_base_rebuilt_61      | holdout             | 61          | 61                             | 61   | 0.011284 | 0.002010  | 1.031373 | 1.043408  |
| s2_base_rebuilt_61      | fixed_340m          | 61          | 61                             | 27   | 0.007176 | 0.001851  | 0.968659 | 1.045710  |
| s2_base_rebuilt_61      | random_supplement   | 61          | 61                             | 34   | 0.013694 | 0.002137  | 1.105537 | 1.133667  |
| s2_e018_b030_tweak_61   | holdout             | 61          | 61                             | 61   | 0.011051 | 0.001524  | 1.033160 | 1.044768  |
| s2_e018_b030_tweak_61   | fixed_340m          | 61          | 61                             | 27   | 0.007091 | 0.001024  | 0.927322 | 1.007262  |
| s2_e018_b030_tweak_61   | random_supplement   | 61          | 61                             | 34   | 0.013385 | 0.001921  | 1.103253 | 1.130310  |
| s2_geom_reason_guard_67 | holdout             | 67          | 79                             | 61   | 0.011083 | 0.001890  | 1.030285 | 1.041960  |
| s2_geom_reason_guard_67 | fixed_340m          | 67          | 79                             | 27   | 0.006821 | 0.001390  | 0.928130 | 1.000564  |
| s2_geom_reason_guard_67 | random_supplement   | 67          | 79                             | 34   | 0.013544 | 0.002287  | 1.093248 | 1.121917  |
| s2_base_rebuilt_61      | validation_900m_all | 61          | 61                             | 4    | 0.012217 | -0.001832 | 0.484227 | 0.634067  |
| s2_e018_b030_tweak_61   | validation_900m_all | 61          | 61                             | 4    | 0.012155 | -0.002499 | 0.498750 | 0.646514  |
| s2_geom_reason_guard_67 | validation_900m_all | 67          | 79                             | 4    | 0.011529 | -0.001530 | 0.485305 | 0.598898  |

The primary guard improves fixed-340M and all-900M RMSE versus the rebuilt S2 and the S2 exponent tweak, while remaining close on seed-7 holdout. It does not beat the delayed power-beta references predictively; its improvement is structural: monotone scaling plus non-collapsed raw-simplex optima.

## Raw-simplex optimum diagnostics

| model                   | target     | raw_source                                                                    | predicted_loss | nearest_train_tv | p0_broad | p0_tech  | p0_reason | p1_broad | p1_tech  | p1_reason | phase0_eff_support | phase1_eff_support | pathology_reason                        |
| ----------------------- | ---------- | ----------------------------------------------------------------------------- | -------------- | ---------------- | -------- | -------- | --------- | -------- | -------- | --------- | ------------------ | ------------------ | --------------------------------------- |
| s2_e018_b030_tweak_61   | 60m_1p2b   | dirichlet_alpha=0.03:895                                                      | 0.970491       | 0.718451         | 0.608105 | 0.391895 | 0.000000  | 0.999663 | 0.000000 | 0.000337  | 2.786222           | 2.657149           | family-collapse-or-scarcity             |
| s2_e018_b030_tweak_61   | 100m_6b    | dirichlet_alpha=0.03:977                                                      | 0.899830       | 0.793957         | 0.889406 | 0.110594 | 0.000000  | 1.000000 | 0.000000 | 0.000000  | 2.460327           | 1.830644           | family-collapse-or-scarcity;hard-corner |
| s2_e018_b030_tweak_61   | 340m_10p4b | dirichlet_alpha=0.03:655                                                      | 0.806134       | 0.772527         | 0.806995 | 0.003889 | 0.189117  | 0.996761 | 0.000000 | 0.003239  | 3.436719           | 2.534725           | family-collapse-or-scarcity             |
| s2_e018_b030_tweak_61   | 900m_24b   | dirichlet_alpha=0.03:655                                                      | 0.700873       | 0.772527         | 0.806995 | 0.003889 | 0.189117  | 0.996761 | 0.000000 | 0.003239  | 3.436719           | 2.534725           | family-collapse-or-scarcity             |
| s2_geom_reason_guard_67 | 60m_1p2b   | observed:baseline_genericfamily_power_observed_only_trustblend_top8actual_cap | 1.036636       | 0.000000         | 0.703323 | 0.260900 | 0.035777  | 0.719069 | 0.268056 | 0.012876  | 25.174266          | 21.127289          | ok                                      |
| s2_geom_reason_guard_67 | 100m_6b    | observed:baseline_genericfamily_power_observed_only_trustblend_top8actual_cap | 0.959834       | 0.000000         | 0.703323 | 0.260900 | 0.035777  | 0.719069 | 0.268056 | 0.012876  | 25.174266          | 21.127289          | ok                                      |
| s2_geom_reason_guard_67 | 340m_10p4b | observed:baseline_genericfamily_power_observed_only_trustblend_top8actual_cap | 0.869982       | 0.000000         | 0.703323 | 0.260900 | 0.035777  | 0.719069 | 0.268056 | 0.012876  | 25.174266          | 21.127289          | ok                                      |
| s2_geom_reason_guard_67 | 900m_24b   | observed:baseline_genericfamily_observed_only_trustblend_top8actual_cap       | 0.781040       | 0.000000         | 0.797694 | 0.174994 | 0.027312  | 0.734240 | 0.254821 | 0.010939  | 28.229240          | 22.945008          | ok                                      |

The S2 exponent tweak still raw-optimizes into phase-1 broad/tech collapse. The primary guard's raw optima are observed-support mixtures with balanced broad/tech/reason family shares and high effective support.

## Fixed 340M target-budget drops

| model                   | drop   | n  | actual_mean | predicted_mean | ratio_mean | rmse     |
| ----------------------- | ------ | -- | ----------- | -------------- | ---------- | -------- |
| s2_e018_b030_tweak_61   | 0.5->1 | 12 | 0.023636    | 0.021675       | 0.917029   | 0.004481 |
| s2_e018_b030_tweak_61   | 0.5->2 | 3  | 0.048517    | 0.040750       | 0.839910   | 0.007843 |
| s2_e018_b030_tweak_61   | 1->2   | 3  | 0.021206    | 0.018264       | 0.861251   | 0.003023 |
| s2_geom_reason_guard_67 | 0.5->1 | 12 | 0.023636    | 0.021675       | 0.917029   | 0.004481 |
| s2_geom_reason_guard_67 | 0.5->2 | 3  | 0.048517    | 0.040750       | 0.839910   | 0.007843 |
| s2_geom_reason_guard_67 | 1->2   | 3  | 0.021206    | 0.018264       | 0.861251   | 0.003023 |

The primary guard leaves same-mixture fixed-340M target-budget drops exactly unchanged relative to S2 tweak, because the added support/reason terms are constant in realized `D` for a fixed mixture and the family guard is zero on these rows.

## Structural checks

See `csv/structural_sanity_summary.csv`. The generated candidates have zero numerical monotonicity violations on the check grid. Analytically, all added guard terms are nonnegative, independent of `N`, and either constant or decreasing in `D`, so the S2 nonnegative scale-head monotonicity is preserved.

## Caveats and next steps

The delayed 98- and 102-parameter power-beta references remain stronger predictive models. The continuation candidate is a compact structural alternative aimed at satisfying the raw-optimum and monotonicity requirements. The useful next direction is to refit the predictive scale head jointly with a smooth version of this guard without allowing the fit to cancel the raw-optimum barrier.
