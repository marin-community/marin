# Delphi a-priori pilot predictive value

Negative deltas favor the pilot-augmented WSPU fit. Cross-block intervals resample the 16 support conditions; external intervals resample coordinates.

| evaluation | stratum | clusters | rmse_delta | rmse_delta_ci_low | rmse_delta_ci_high | rmse_share_better | mae_delta | mae_delta_ci_low | mae_delta_ci_high |
|---|---|---|---|---|---|---|---|---|---|
| cross_block_pilot | pooled | 16 | -0.010833 | -0.019678 | -0.003088 | 1.000000 | -0.008117 | -0.014392 | -0.002916 |
| external_pre_pilot_bank | pooled | 247 | 0.005976 | -0.002067 | 0.014432 | 0.080000 | 0.001430 | -0.002064 | 0.005253 |
| external_pre_pilot_bank | eligible_intervention | 90 | -0.003220 | -0.005833 | 0.000693 | 0.955000 | -0.000124 | -0.001887 | 0.001605 |
| external_pre_pilot_bank | model_optimum_archive | 157 | 0.008796 | -0.002347 | 0.019578 | 0.056000 | 0.002321 | -0.003314 | 0.008315 |

## External model-optimum selection

| design | bank_size | regret_at_1 | top5_regret | selected_rank | frontier_predicted_rank | rmse | spearman |
|---|---|---|---|---|---|---|---|
| panel_280 | 157 | 0.015735 | 0.013245 | 14.000000 | 10.000000 | 0.037832 | 0.893534 |
| panel_plus_all_37_pilot | 157 | 0.026141 | 0.016787 | 39.000000 | 58.000000 | 0.046628 | 0.562897 |

The 317-row external fit measures incremental data value and is not a matched-budget replacement for the 280-row panel.
