# Effective-N Sweep for 20M Bias

Date: 2026-04-26

## Question

The refreshed MCT-LRQ74 model overpredicts the corrected `20M/2.6B` rows by
about `+0.019` BPB. One hypothesis was that using pure non-embedding parameter
count makes the smallest tied-embedding model look too small. This sweep tests:

```text
N_eff = non_embedding_params + rho * input_embedding_params
```

where `rho=0` is the current canonical convention and `rho=1` is tied-total
parameter count.

This is an isolated counterfactual. It does not mutate the canonical analysis
dataset.

## Naive Fixed-Exponent Sweep

First pass: keep the current MCT-LRQ74 exponents fixed and only replace `N`.

| rho | seed7 RMSE | fixed340 RMSE | all900 RMSE | all-row RMSE | 20M bias | 20M RMSE | 20M 0.5x bias | 20M 1x bias | 20M 2x bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.010034 | 0.006082 | 0.010397 | 0.025518 | 0.019085 | 0.022269 | 0.030913 | 0.019709 | 0.006633 |
| 0.25 | 0.011623 | 0.010738 | 0.023184 | 0.025634 | 0.014942 | 0.019206 | 0.027882 | 0.015051 | 0.001891 |
| 0.50 | 0.013497 | 0.014319 | 0.034273 | 0.025849 | 0.011920 | 0.017550 | 0.026211 | 0.011571 | -0.002023 |
| 0.75 | 0.014507 | 0.015834 | 0.042189 | 0.026025 | 0.009669 | 0.016733 | 0.025191 | 0.008947 | -0.005132 |
| 1.00 | 0.014926 | 0.016207 | 0.048057 | 0.026153 | 0.007944 | 0.016399 | 0.024532 | 0.006921 | -0.007621 |

Adding embedding capacity does reduce 20M bias, but the cost is large:
fixed-340M and 900M extrapolation degrade rapidly. Tied-total params are not a
viable replacement for non-embedding params under the current structural law.

## Exponent-Retuned Check

Because the MCT exponents were selected under `rho=0`, I reran the local
exponent-refinement diagnostic for `rho in {0, 0.25, 0.5, 1.0}`.

| rho | selected alpha | beta | gamma | delta | seed7 RMSE | fixed340 RMSE | all900 RMSE | 0.5x->2x ratio | 1x->2x ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.154791 | 0.146425 | 0.014295 | 1.063376 | 0.010034 | 0.006082 | 0.010397 | 0.936468 | 0.999807 |
| 0.25 | 0.290911 | 0.207383 | 0.053442 | 1.296238 | 0.010006 | 0.006067 | 0.010262 | 0.911840 | 0.957382 |
| 0.50 | 0.290911 | 0.207383 | 0.053442 | 1.296238 | 0.010536 | 0.008208 | 0.016871 | 0.891842 | 0.931224 |
| 1.00 | 0.290911 | 0.207383 | 0.053442 | 1.296238 | 0.011964 | 0.011360 | 0.030433 | 0.876519 | 0.907467 |

Retuning makes `rho=0.25` competitive on RMSE, but it does not preserve the
fixed-340M continuation ratios as well as `rho=0`.

The row-level retuned `rho=0.25` model still has essentially the same 20M
problem:

| 20M multiplier | n | bias | RMSE |
|---:|---:|---:|---:|
| 0.5x | 13 | 0.032355 | 0.032819 |
| 1.0x | 13 | 0.019315 | 0.019846 |
| 2.0x | 13 | 0.005738 | 0.009092 |

So the apparent 20M improvement in the fixed-exponent sweep is mostly a
tradeoff against scale extrapolation. Once exponents are retuned to recover
cross-scale performance, the 20M bias returns.

## Conclusion

The 20M bias is not cleanly fixed by counting embedding parameters globally.

The evidence points away from replacing canonical `N = non_embedding_params`
with tied-total params. A small partial embedding count (`rho=0.25`) is
plausible numerically, but it does not solve the 20M bias once the structural law
is retuned.

The next likely causes are:

- the `20M/2.6B` proxy has a different model-family/configuration regime than
  the larger scales;
- the joint law needs an explicit small-model correction or embedding-heavy
  regime term, not a global `N` replacement;
- the low-`N`, low-`D` corner needs direct fitting pressure, because the current
  objective is dominated by the 60M and 100M panels.

Recommendation: keep non-embedding params as the canonical `N` for now. If we
want to address 20M, test a targeted small-model or embedding-fraction residual
with a guard that fixed-340M and 900M metrics must not degrade.

## Artifacts

- `n_eff_sweep_summary.csv`: fixed-exponent sweep metrics.
- `n_eff_refined_top_summary.csv`: best exponent-refined rows by score.
- `rho_0p25_refined_eval/csv/`: row-level metrics for the best retuned
  `rho=0.25` law.
- `n_eff_sweep_tradeoffs.png`: fixed-exponent tradeoff plot.
- `n_eff_refined_tradeoff.png`: refined exponent tradeoff plot.
