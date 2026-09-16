# Completed Wikipedia and FineMath sweeps

At the 12 September 2026, 06:02 UTC snapshot, Wikipedia's epoch-matched proxy and target both have their lowest observed Uncheatable loss at 30% Wikipedia, or 4.77 materialized epochs. FineMath-3+'s proxy also selects 30%. The experiment asks whether domains prefer substantially different repetition levels under the same evaluation objective, which would motivate alternatives to one shared epoch cap. These measurements do not show that separation. FineMath's target curve is incomplete.

Each run mixes the named domain with the same six-part web background: `hq_actual`, `hq_synth`, `medium_high`, `medium`, `medium_low` and `low_actual`, in the proportions defined by `WEB_COUNTS` in [starcoder_tpp10.py](../../../../../experiments/domain_phase_mix/starcoder_tpp10.py). Both model scales use ten training tokens per total model parameter. The proxy has 16.6M parameters and a 10.49M-token domain pool; the target has 301.2M parameters and a 190.32M-token pool. The pool reduction matches the domain's epochs between scales at each mixture weight. The objective equally averages bits per byte (BPB) across seven Uncheatable components: Wikipedia English, GitHub Python, GitHub C++, BBC News, arXiv physics, arXiv computer science and AO3 English.

| Domain | Scale | Lowest completed mixture weight | Domain epochs | Uncheatable BPB | Coverage |
| --- | --- | ---: | ---: | ---: | --- |
| Wikipedia | Matched proxy | 30% | 4.7684 | 1.530924 | Full grid |
| Wikipedia | Target | 30% | 4.7685 | 1.113183 | Full grid |
| FineMath-3+ | Matched proxy | 30% | 4.7684 | 1.493869 | Full grid |
| FineMath-3+ | Target | 20% | 3.1610 | 1.090937 | 30/50/70/100% unfinished |

The grid is 0, 5, 10, 20, 30, 50, 70 and 100%. Twenty-four of the 28 nonzero-domain training points are complete, plus two shared web-only controls, one per model scale. The CSV repeats those controls across domains for plotting; they are not independent runs.

Wikipedia's target valley is shallow: 20% is only 0.000505 BPB worse than 30%; 50% is 0.003572 worse. The FineMath target continues improving through its last completed point, so 20% does not establish its optimum. One trainer seed and a coarse grid do not resolve the continuous optima or statistical uncertainty. Changes along these sweeps combine repetition with replacement of web data.

The component plot shows different trade-offs despite the coincident aggregate proxy minima. FineMath improves coding losses more, while Wikipedia improves Wikipedia loss more. These component results do not establish different optima for the prespecified aggregate objective.

[Overview and valley detail](completed_sweeps.png) · [Component changes from each scale's web-only control](component_tradeoffs.png) · [Plotted values](completed_points.csv) · [Exact minima and neighboring gaps](summary.json)

All plotted BPBs use total scored loss bits divided by total scored bytes per evaluation component. The previous evaluator averaged batch BPB ratios with token weights, allowing batch boundaries to change the result. Three saved-checkpoint audits verify the correction: shared proxy control, shared target control and Wikipedia proxy at 50%. Corrected BPB agrees across evaluation batching layouts and with reconstruction from saved token losses to within the original 0.00005 BPB tolerance. The controls also reproduce the previous evaluator's discrepancy. `source.json` pins the corrected result snapshot; audit receipts and each mixture's realized epoch count are archived beside it. The scheduler's completion records will be finalized once the remaining training ends; saved checkpoints and metrics for the plotted points have already been verified.

Reproduce from the archived snapshot at the repository root:

```bash
uv run python -m experiments.domain_phase_mix.plot_tpp10_domain_sweeps \
  --repair-plan .agents/projects/starcoder_tpp10/domain_sweeps/repairs_20260911/plan.json \
  --output .agents/projects/starcoder_tpp10/domain_sweeps/plots_20260912
```

Use a new output directory and `--refresh` to archive a later snapshot. The plotter uses measured points and straight joins without fitted curves. Repository lint and focused Pyrefly passed; both rendered figures were inspected for overlap and clipping. The manuscript and experimental settings are unchanged.
