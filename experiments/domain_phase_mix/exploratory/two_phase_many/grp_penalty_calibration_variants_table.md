# GRP Penalty / CES Variant Metrics

Lower is better for all metric columns except parameter counts.

| variant | source | params | nonlinear | train_rmse | cv_rmse | fold_regret1 | tail_opt | cv_depopt8 | cv_raw_tv | anchor_mae | raw_opt_bpb | deploy_bpb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_grp_log | prior benchmark | 37 | 6 | 0.00793 | 0.00889 | 0.00281 | 0.00232 | 0.06274 | 0.63077 | 0.02751 | 1.00841 | 1.04036 |
| power_family | prior benchmark | 39 | 8 | 0.00762 | 0.00864 | 0.00000 | 0.00147 | 0.06604 | 0.67490 | 0.02070 | 1.00343 | 1.04372 |
| power_boxcox_family | prior benchmark | 40 | 9 | 0.00775 | 0.00883 | 0.00000 | 0.00277 | 0.06066 | 0.60414 | 0.02084 | 1.00965 | 1.04096 |
| power_family_penalty | full retune | 43 | 10 | 0.00830 | 0.00952 | 0.00281 | 0.00256 | 0.04571 | 0.50550 | 0.00927 | 1.02480 | 1.04535 |
| power_family_penalty_global_ftotal | full retune | 48 | 11 | 0.00821 | 0.00939 | 0.00281 | 0.00104 | 0.03613 | 0.57156 | 0.00580 | 1.03093 | 1.05128 |
| power_family_penalty_global_ftotal_pairces | full retune | 49 | 12 | 0.00810 | 0.00924 | 0.00000 | 0.00135 | 0.04061 | 0.52073 | 0.00820 | 1.02720 | 1.05216 |
| power_boxcox_family_penalty | full retune | 44 | 11 | 0.00796 | 0.00917 | 0.00281 | 0.00125 | 0.04181 | 0.61189 | 0.01184 | 1.02621 | 1.05031 |
| power_boxcox_family_penalty_global_ftotal | full retune | 49 | 12 | 0.00797 | 0.00902 | 0.00000 | 0.00212 | 0.05231 | 0.59654 | 0.02062 | 1.01575 | 1.04731 |
| power_boxcox_family_penalty_global_ftotal_pairces | full retune | 50 | 13 | 0.00786 | 0.00897 | 0.00000 | 0.00187 | 0.04839 | 0.54203 | 0.01545 | 1.01964 | 1.04947 |

Notes:
- `original_grp_log`, `power_family`, and `power_boxcox_family` are carried over from the earlier benchmark on the same calibration metric set.
- The six penalty / richer-penalty rows are now full Powell retunes from deterministic seeded starts.
