# Domain-Aware Synergy Joint-Law Sprint

## Goal

Test whether the domain-aware scaling-law features from `Domain-Aware Scaling Laws Uncover Data Synergy` improve our joint mixture/scale law on the canonical ND panel.

The paper proposes two relevant ideas:

- First-order domain-benchmark synergy: domain-specific deviations from the global data scaling exponent.
- Second-order pretraining synergy: pairwise domain co-occurrence terms based on a soft-min of domain token exposures.

## Headline Result

The reliable canonical-residual diagnostic is negative: the unchanged canonical MCT-LRQ baseline remains best with score `0.04431`. The best paper-style residual correction is `canon_resid_paper_first_order_domain_log` with score `0.04519`.
The full local anchored ablation is more encouraging but less decisive: `paper_full_current_source_power_tau0.1` improves the local selection score from `0.15191` to `0.13193`. This should be interpreted as evidence that current-source/entropy scale features are useful, not as a promotion candidate.

In our anchored single-target setting, the first-order feature reduces to:

```latex
\Delta z_k(w,D) = u_k(w)\log(D/D_0), \quad \sum_k \gamma_k = 0.
```

The sum-to-zero constraint is implemented by centering the share vector before multiplying by the scale term.

The source-group second-order feature is:

```latex
\Delta q_{ab}(w,D)=\operatorname{softmin}_\tau(\log(1+u_aD),\log(1+u_bD))-\operatorname{softmin}_\tau(\log(1+u_aD_0),\log(1+u_bD_0)).
```

All scale features are zero at the corrected `100M/6B` anchor, preserving the anchor mixture regression.

## Variants Tested

| model                                          | group          | transform   | first_order   | entropy   | second_order   | description                                                                               |
|:-----------------------------------------------|:---------------|:------------|:--------------|:----------|:---------------|:------------------------------------------------------------------------------------------|
| smooth_mct_lrq_like                            | domain/all     | power       | False         | False     | False          | Existing anchored LRQ mixture body plus global N, canonical family D, and ND cross terms. |
| paper_first_order_domain_power                 | domain/all     | power       | True          | False     | False          | Paper first-order domain synergy adapted as centered domain-specific D-scaling heads.     |
| paper_first_order_domain_log                   | domain/all     | log         | True          | False     | False          | Same as first-order domain, using log(D/D0) exactly from z_k(D)-z_k(D0).                  |
| paper_first_order_current_source_power         | current_source | power       | True          | False     | False          | First-order synergy on source-aware groups instead of all 39 domains.                     |
| paper_first_order_current_source_log           | current_source | log         | True          | False     | False          | Source-aware first-order synergy using log(D/D0).                                         |
| paper_first_order_canonical_power              | canonical      | power       | True          | False     | False          | First-order synergy on the canonical MCT-LRQ family partition.                            |
| paper_entropy_current_source_power             | current_source | power       | False         | True      | False          | Paper log-D decomposition entropy term on source-aware shares.                            |
| paper_second_order_current_source_tau0.1       | current_source | power       | False         | False     | True           | Paper second-order soft-min co-occurrence terms on source-aware groups, tau=0.1.          |
| paper_second_order_current_source_tau1         | current_source | power       | False         | False     | True           | Paper second-order soft-min co-occurrence terms on source-aware groups, tau=1.            |
| paper_first_second_current_source_power_tau0.1 | current_source | power       | True          | False     | True           | Source-aware first-order plus second-order soft-min synergy.                              |
| paper_full_current_source_power_tau0.1         | current_source | power       | True          | True      | True           | Source-aware first-order, entropy, and second-order synergy terms.                        |

## Canonical MCT-LRQ Residual Diagnostic

This is the highest-signal diagnostic: start from current canonical `mct_lrq69_drop_no_barrier` row predictions, then fit only paper-style residual corrections.

| model                                                      |   feature_count | ridge      |   seed7_holdout_rmse |   fixed340_holdout_rmse |   random_supplement_rmse |   all900_holdout_rmse |   score |   seed7_holdout_spearman |
|:-----------------------------------------------------------|----------------:|:-----------|---------------------:|------------------------:|-------------------------:|----------------------:|--------:|-------------------------:|
| canonical_mct_lrq69_drop                                   |               0 |            |              0.01007 |                 0.00614 |                  0.01233 |               0.01575 | 0.04431 |                  0.9707  |
| canon_resid_paper_first_order_domain_log                   |              48 | 1000.00000 |              0.01067 |                 0.00694 |                  0.01289 |               0.01469 | 0.04519 |                  0.97017 |
| canon_resid_paper_first_order_domain_power                 |              48 | 1000.00000 |              0.01069 |                 0.00696 |                  0.0129  |               0.01469 | 0.04525 |                  0.97017 |
| canon_resid_paper_entropy_current_source_power             |              10 | 100.00000  |              0.01224 |                 0.00985 |                  0.01385 |               0.01461 | 0.05056 |                  0.96827 |
| canon_resid_paper_second_order_current_source_tau0.1       |              30 | 0.00000    |              0.01217 |                 0.01011 |                  0.01359 |               0.01498 | 0.05084 |                  0.9623  |
| canon_resid_paper_second_order_current_source_tau1         |              30 | 0.00000    |              0.01217 |                 0.01011 |                  0.01359 |               0.01498 | 0.05085 |                  0.96166 |
| canon_resid_paper_first_order_canonical_power              |              16 | 0.00000    |              0.01222 |                 0.0102  |                  0.01361 |               0.01533 | 0.05136 |                  0.96081 |
| canon_resid_paper_first_second_current_source_power_tau0.1 |              37 | 0.10000    |              0.01221 |                 0.00845 |                  0.01451 |               0.01845 | 0.05362 |                  0.9651  |
| canon_resid_paper_first_order_current_source_power         |              16 | 0.00000    |              0.01235 |                 0.00852 |                  0.0147  |               0.01853 | 0.0541  |                  0.96457 |
| canon_resid_paper_full_current_source_power_tau0.1         |              38 | 100.00000  |              0.01249 |                 0.00836 |                  0.01497 |               0.01855 | 0.05437 |                  0.96504 |
| canon_resid_paper_first_order_current_source_log           |              16 | 0.01000    |              0.01273 |                 0.00961 |                  0.01475 |               0.01864 | 0.05574 |                  0.95627 |

## Full Local Anchored Model Fit

These models refit the LRQ anchor plus each scale-feature family. They are ablations, not exact replacements for canonical MCT-LRQ.

| model                                          |   seed7_train_rmse |   seed7_holdout_rmse |   fixed340_holdout_rmse |   random_supplement_rmse |   all900_holdout_rmse |   external_60m_rmse |   external_900m_rmse |   seed7_holdout_spearman |   drop_0p5_to_1_ratio |   drop_0p5_to_2_ratio |   drop_1_to_2_ratio |   selection_score |
|:-----------------------------------------------|-------------------:|---------------------:|------------------------:|-------------------------:|----------------------:|--------------------:|---------------------:|-------------------------:|----------------------:|----------------------:|--------------------:|------------------:|
| paper_full_current_source_power_tau0.1         |            0.04667 |              0.03378 |                 0.01241 |                  0.04388 |               0.04186 |             0.0526  |              0.04186 |                  0.96742 |               0.60061 |               0.68505 |             0.81043 |           0.13193 |
| paper_entropy_current_source_power             |            0.04668 |              0.03417 |                 0.01357 |                  0.04414 |               0.04156 |             0.05258 |              0.04156 |                  0.96515 |               0.54884 |               0.62602 |             0.74063 |           0.13344 |
| paper_first_order_domain_log                   |            0.04593 |              0.03655 |                 0.02002 |                  0.04559 |               0.03462 |             0.05077 |              0.03462 |                  0.91407 |               0.13763 |               0.15802 |             0.1883  |           0.13678 |
| paper_first_order_domain_power                 |            0.04594 |              0.03661 |                 0.02017 |                  0.04562 |               0.03463 |             0.05076 |              0.03463 |                  0.91322 |               0.13071 |               0.14926 |             0.17681 |           0.13702 |
| paper_first_order_current_source_log           |            0.04776 |              0.03832 |                 0.0185  |                  0.04861 |               0.04339 |             0.05342 |              0.04339 |                  0.91713 |               0.44164 |               0.51644 |             0.62751 |           0.14882 |
| paper_first_order_current_source_power         |            0.0478  |              0.03865 |                 0.01957 |                  0.04875 |               0.04343 |             0.05342 |              0.04343 |                  0.91322 |               0.35841 |               0.4089  |             0.48387 |           0.15039 |
| paper_first_order_canonical_power              |            0.04813 |              0.03839 |                 0.0179  |                  0.04888 |               0.04618 |             0.05384 |              0.04618 |                  0.91983 |               0.41014 |               0.46789 |             0.55364 |           0.15135 |
| smooth_mct_lrq_like                            |            0.04814 |              0.03853 |                 0.01838 |                  0.04894 |               0.04607 |             0.05384 |              0.04607 |                  0.91893 |               0.36271 |               0.41381 |             0.48968 |           0.15191 |
| paper_first_second_current_source_power_tau0.1 |            0.04776 |              0.03916 |                 0.01952 |                  0.04949 |               0.04378 |             0.05343 |              0.04378 |                  0.90777 |               0.35707 |               0.40736 |             0.48204 |           0.15195 |
| paper_second_order_current_source_tau0.1       |            0.04811 |              0.03899 |                 0.01831 |                  0.04961 |               0.04647 |             0.05386 |              0.04647 |                  0.90999 |               0.36249 |               0.41355 |             0.48937 |           0.15338 |
| paper_second_order_current_source_tau1         |            0.04811 |              0.03899 |                 0.01831 |                  0.04961 |               0.04647 |             0.05386 |              0.04647 |                  0.90999 |               0.36249 |               0.41355 |             0.48937 |           0.15338 |

## Interpretation

- The paper's first-order term is a real structural direction for our setting, but after anchoring it mostly becomes domain- or group-specific data-scaling heads.
- On top of canonical MCT-LRQ, raw domain first-order heads slightly improve the tiny 900M diagnostic and fixed-340M-all RMSE, but they worsen seed7 holdout, fixed-340M holdout, and random supplement RMSE.
- Source-group/entropy terms help the weaker full local anchored model, which suggests useful missing structure in the compact local ablation.
- Second-order soft-min terms are interpretable, but with only one primary BPB target they are weakly identified unless restricted to coarse source groups.
- Recommendation: do not replace the current law with these terms yet. If we pursue this direction, use a constrained source-group entropy/first-order correction inside the exact canonical implementation and validate raw optima.

## Artifacts

- `csv/model_summary.csv`: split metrics for full local anchored variants.
- `csv/row_predictions.csv`: row-level predictions for full local variants.
- `csv/canonical_residual_synergy_summary.csv`: residual corrections on top of canonical MCT-LRQ.
- `csv/canonical_residual_synergy_predictions.csv`: row-level canonical residual predictions.
- `csv/fixed340_drop_summary.csv`: fixed-340M target-budget drop ratios.
- `plots/domain_aware_synergy_pred_actual.html`: full local prediction scatter.
- `plots/canonical_residual_domain_aware_synergy_pred_actual.html`: canonical residual prediction scatter.
