# Repetition-Aware Data-Filtering Law on 300M Swarm

## Data

- Source signal rows: `242`
- Complete target rows: `{'issue5416_loss': 93, 'uncheatable_bpb': 100}`
- Source matrix: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv`
- Targets: `eval/uncheatable_eval/bpb` and `-issue5416_aggregate`.
- Caveat: each target is fitted only on rows complete for that target.

## Form

This fixed-scale screen reuses the RAML adaptations from the ND script. Domain exposure is

\(h_i = 0.8w_{0i}+0.2w_{1i}\), \(r_i=w_{0i}c_{0i}+w_{1i}c_{1i}\),

and repeated exposure is discounted by

\(r_{i,\mathrm{eff}}=r_i\) for \(r_i\le1\), and \(r_{i,\mathrm{eff}}=1+r_1(1-\exp(-(r_i-1)/r_1))\) for \(r_i>1\).

The scalar law uses \(D_{\mathrm{eff}}=\sum_i h_iD\,r_{i,\mathrm{eff}}/r_i\), optionally with positive per-domain value weights and a signed linear mixture head.

## 5-Fold OOF Results

| target          | model                                      |   parameter_count |     rmse |      mae |   spearman |   pearson |   regret_at_1 |   top8_overlap | optimizer_success   |
|:----------------|:-------------------------------------------|------------------:|---------:|---------:|-----------:|----------:|--------------:|---------------:|:--------------------|
| issue5416_loss  | scale_only_chinchilla_size                 |                 1 | 0.448819 | 0.349323 |  -0.082002 | -0.132090 |      0.204078 |       0.250000 | True                |
| issue5416_loss  | raml_uniform_domain_value_shared_r1        |                 7 | 0.449192 | 0.354187 |  -0.030005 |  0.061575 |      1.137471 |       0.250000 | False               |
| issue5416_loss  | raml_domain_value_shared_r1                |                46 | 0.463667 | 0.389219 |   0.252753 |  0.385701 |      0.574919 |       0.500000 | False               |
| issue5416_loss  | raml_domain_value_linear_mix_per_domain_r1 |               123 | 0.788401 | 0.532383 |   0.494032 |  0.412516 |      1.361513 |       0.250000 | False               |
| issue5416_loss  | raml_domain_value_linear_mix_shared_r1     |                85 | 0.803540 | 0.644050 |   0.222748 |  0.278701 |      0.574919 |       0.250000 | False               |
| uncheatable_bpb | raml_domain_value_shared_r1                |                46 | 0.051973 | 0.035401 |   0.367417 |  0.346376 |      0.023493 |       0.250000 | True                |
| uncheatable_bpb | raml_uniform_domain_value_shared_r1        |                 7 | 0.054776 | 0.037545 |   0.173165 |  0.142857 |      0.065379 |       0.250000 | True                |
| uncheatable_bpb | scale_only_chinchilla_size                 |                 1 | 0.055459 | 0.037467 |  -0.033070 | -0.124690 |      0.025077 |       0.250000 | True                |
| uncheatable_bpb | raml_domain_value_linear_mix_per_domain_r1 |               123 | 0.073251 | 0.061235 |   0.143810 |  0.154247 |      0.020708 |       0.250000 | False               |
| uncheatable_bpb | raml_domain_value_linear_mix_shared_r1     |                85 | 0.123193 | 0.092062 |   0.070999 |  0.072428 |      0.065379 |       0.250000 | True                |

## Interpretation

- The fixed-scale form is tested on every currently target-complete row: 100 for uncheatable BPB and 93 for issue #5416 aggregate.
- Domain value weights are required for useful rank fit; uniform-domain repetition is too weak as a mixture model.
- The more flexible signed-head/per-domain-r1 forms should be read as high-capacity comparators because optimizer convergence is not always clean.
