# Figure 3 fitted comparison and 60% audit — 13 September 2026

The MARINER companion is available for visual comparison. The current measured Figure 3 remains preferable: its target upturn and minimum are represented more faithfully by the observations than by this fitted curve. No manuscript figure, caption, outline, or main PDF was replaced.

## Fitted companion

The companion uses the same existing MARINER implementation and tuning procedure as the selected Figure 4. Each curve is fitted independently to all twelve complete, consistently normalized mixture means. The three matched subsets remain separate; the secondary pooled matched fit is recorded but not drawn as an extra curve. The fitting coordinates use realized allocations and physical pool sizes; the x axis maps these back to the requested StarCoder fractions using all 21 audited allocation coordinates.

The Weibull-benefit and squared-softplus-harm shapes are shared across the two physical pools within a curve. Shape and ridge are selected using leave-one-mixture-out tuning at the provisional floor, followed by floor-multiplier tuning with those choices fixed and a full-data refit. The existing fold-specific median anchor and zero external repeat-noise margin are retained. These are descriptive fits; inner cross-validation is tuning evidence. No settings were changed to improve the picture after inspecting the result.

| Curve | Observed selection | Fitted minimum | Fitted epochs | Fit RMSE (BPB) |
|---|---:|---:|---:|---:|
| target | 70% | 63.00% | 9.990 | 0.00984 |
| unmatched | 100% | 100.00% | 0.872 | 0.00612 |
| matched_20260912 | 50% | 50.53% | 8.023 | 0.01543 |
| matched_20260913 | 50% | 50.99% | 8.095 | 0.02058 |
| matched_20260914 | 50% | 52.91% | 8.401 | 0.01588 |
| matched_pooled | 50% | 52.31% | 8.306 | 0.01466 |

The target fitted minimum is 0.772947 BPB, whereas the observed grid minimum is 0.765565 at 70%. At 100%, the fit predicts 0.780012 versus the measured 0.800619 BPB, an error of −0.020607 BPB. Its predicted penalty between 100% and its own minimum is only 0.007065 BPB, versus the measured-grid penalty of 0.035054 BPB. The continuous minimum near 63% is therefore model-dependent and does not supersede the observed selection.

The plot retains measured points within the original displayed axis ranges, measured-grid stars, and target losses at the measured proxy selections. Its bars still report 1.32% and 4.58% excess relative to the observed target grid minimum, and 71.2% less excess loss with simulated epoching. The top axis, FLOP labels, colors, and subset curves are preserved. Fitted lines are explicitly distinguished from measured selections. The left axis retains the original p≥0.1 display, and the right retains p≥0.3; all twelve points, including p=0, are used in every fit.

## The 60% point

Independent reconstruction verifies all 102 archived raw artifacts and all 60 plotted curve means exactly. Target losses at 55%, 60%, 65%, and 70% are 0.7701349691, 0.7694386281, 0.7665562588, and 0.7655645236 BPB. Thus 60% is a flattening, not an increase. It lies 0.0010930142 BPB above the chord between 55% and 65%; accounting for actual fractions changes this to 0.0011017899.

The feature exists in saved token-loss history before the 60% checkpoint resumed. All three local endpoints use the same original schema, and the common scored-byte normalization preserves the feature. The audit found no remaining metric, coordinate, evaluation-population, or checked configuration error. One target seed and archived runtime evidence do not establish the cause or rule out every possible runtime/resume effect. No fresh checkpoint rescoring was performed. Keep the measurement; do not label it definitively as training noise.

The unmatched proxy also flattens at 60%, with a two-seed mean chord residual of 0.0058828512 BPB. Both trainer seeds contribute. Shared data support and order mean the coincident feature is not independent evidence of noise.

See [the detailed audit](audit/REPORT.md) for configurations, checkpoint history, allocation checks, limitations, and reproduction evidence.

## Validation and reproduction

An independent reviewer reconstructed every grid and dense prediction from the saved MARINER coefficients exactly. A separate 100,001-point search and basin refinement reproduced all six minima within 1e-6 in fraction and 1e-12 BPB. All seven source hashes, six input identities, canonical design digest, and standalone/summary outputs agree. The original PDF and PNG still match the consistent-BPB correction receipts.

PDF rendering and a cold visual review found no cropped or overlapping labels. The reviewer independently identified the poor target fit. The companion is intentionally retained as a comparison, not promoted into the manuscript. No training or evaluation jobs were launched, and no Overleaf push was made.

From the Marin checkout:

```bash
PYTHONPATH=. uv run python .agents/projects/starcoder_tpp10/live/figure3_mariner_20260913/fit_curves.py
uv run .agents/projects/starcoder_tpp10/live/figure3_mariner_20260913/plot_comparison.py
uv run .agents/projects/starcoder_tpp10/live/figure3_mariner_20260913/audit/audit_p060.py
```

Per-curve fit files are reused only when their source/input hashes match. The original measured figure and builder are preserved in `sources/`.
