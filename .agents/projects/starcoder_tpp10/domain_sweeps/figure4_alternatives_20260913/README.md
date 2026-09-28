# Figure 4 alternatives

Three versions share a MATH-500 right panel and differ only in the left-panel benchmarks:

- `figure4_left_gsm8k.pdf`: Wikipedia on AO3 English; FineMath on GSM8K.
- `figure4_left_math500.pdf`: Wikipedia on AO3 English; FineMath on MATH-500.
- `figure4_left_wikipedia_english_math500.pdf`: Wikipedia on Wikipedia English; FineMath on MATH-500.

All panels use absolute evaluation loss in bits per byte, with identical linear limits of 0.70–2.02. The math scores are already logged schema-v2 ratios of total scored reference-solution loss bits to total scored token bytes. They use the same answer-only mask as the perplexities, and have the same minimizing checkpoints. These three observed versions join measured points, without smoothing or normalization. Different evaluation corpora have different inherent difficulty; the common unit supports comparing proxy and target for each benchmark, without equating performance across benchmarks.

The legend labels per-run training compute: proxy 2.49e16 FLOPs; target 6.66e18 FLOPs, a factor of 267.53. Both retain total-parameter TPP approximately 10. FLOPs follow the frozen design's three times analytic forward FLOPs per token times training tokens, including attention and vocabulary projection.

The preferred Wikipedia-English version now marks all observed minima with stars, including FineMath's MATH-500 target endpoint. That endpoint is 100% focal-domain training data and the largest feasible epoch count at this horizon and pool size. The explanatory footers, boundary-point text, and off-scale annotation have been removed. Wikipedia's proxy MATH-500 endpoint (2.9336 BPB) is still in the data and is clipped only for display. Every exported curve retains all eight points. One trainer seed and one matched subset are available. The two older AO3 versions retain their earlier marker conventions until regenerated.

Wikipedia-English has observed minima at 7.94 proxy and 11.10 target epochs. AO3 instead has minima at 3.16 and 1.58 epochs and measures transfer to English fiction. FineMath on GSM8K has both minima at 11.10 epochs; FineMath on MATH-500 has a proxy minimum at 11.10 and its target minimum at the 15.83-epoch upper boundary.

The source and output hashes are in `provenance.json`. Independent checks verified the metric definitions, compute, original two versions' numeric values, and visual encodings. An additional exact-value check covers all 192 plotted values across the three versions and confirms all right-panel records are identical. Minimum labels were moved away from crossing curves after visual review. Ruff and Black passed through the repository lint entry point.

Reproduce with `uv run plot_alternatives.py` from this directory. The fitted Wikipedia-English/MATH-500 version was selected and integrated into the paper on 13 September; the original alternatives remain available for comparison. No training or evaluation jobs were launched.

## MARINER companion and CC handoff, 13 September

`figure4_left_wikipedia_english_math500_mariner.pdf` is a separate fitted companion to the preferred version. It retains the observed circles and squares, overlays MARINER fits, and marks their constrained minima with stars. Domain colors, proxy/target line styles, absolute BPB limits, and compute labels are shared with the observed version. The simplified figures are 4.05 inches tall, reduced from 4.45 inches.

| Domain and evaluation | Observed proxy epochs | Fitted proxy epochs | Observed target epochs | Fitted target epochs |
| --- | ---: | ---: | ---: | ---: |
| Wikipedia / Wikipedia English | 7.94 | 7.61 | 11.10 | 10.77 |
| Wikipedia / MATH-500 | 4.77 | 5.34 | 7.94 | 5.99 |
| FineMath-3+ / MATH-500 | 11.10 | 13.28 | 15.83 | 14.52 |

The companion reuses `fit_starcoder_tpp10_mariner_20260911.py` and the registry entry `weibull_softplus_unscaled@kappa_floor_link_flat15_nocap`. Its two features are the fixed web blend and the varied domain. Each curve uses its actual materialized epochs and physical pool sizes, including all eight measurements and the off-scale point. Leave-one-mixture-out validation selects shared shapes, ridge, and the floor multiplier, followed by a full refit. These two-bucket fits use fold-specific median floor anchors and zero external noise margin; they do not import the 39-domain task anchors. The minimizer checks the feasible boundaries and refines all basins found by a dense scan. The FineMath fits are reused identically in both panels.

Fit RMSE ranges from 0.00039 to 0.00583 BPB. An independent audit reconstructed every prediction and verified each constrained minimum with a separate 100,001-point scan. These are descriptive fits to eight points, with sparse observations around several minima. FineMath's target fit predicts a small rebound after 14.5 epochs even though the measured losses decrease through the 15.83-epoch endpoint. Several fitted columns are poorly conditioned, so individual amplitudes should not receive a mechanistic interpretation. The plot reports epochs to one decimal and makes no confidence-interval claim.

Reproduce the fitted pair from the repository root:

```bash
PYTHONPATH=. uv run python .agents/projects/starcoder_tpp10/domain_sweeps/figure4_alternatives_20260913/fit_mariner.py
uv run .agents/projects/starcoder_tpp10/domain_sweeps/figure4_alternatives_20260913/plot_alternatives.py --preferred
```

Per-curve fits, coefficients, CV diagnostics, and source hashes are in `mariner_fits/`; `preferred_provenance.json` records the new outputs. The earlier `provenance.json` describes the previous three-version snapshot and is superseded for the preferred figure. CC changes to carry forward: replace its boundary marker with a star; remove explanatory plot footers; consider the fitted companion with fitted-minimum wording in the caption. The author selected this companion; a compact manuscript version and the synchronized outline are recorded in the paper directory at revision_notes/20260913_domain_repetition_figure/CC_CHANGES.md.
