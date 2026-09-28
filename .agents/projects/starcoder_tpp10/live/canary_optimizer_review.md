# TPP10 canary review — 10 September 2026

All three frozen p=1 canaries pass the endpoint and optimizer checks. The recovered matched proxy finished at 05:25:40 PDT; its on-demand parent finished 17 seconds later. Calibration remains complete at eight of eight runs, with proxy batch 32 retained.

| Arm | Final evaluation BPB | Final step | Config fingerprint |
| --- | ---: | ---: | --- |
| Target | 0.8207455873 | 11490 | `0cbcd192` |
| Unmatched proxy | 1.0613559484 | 2531 | `78cb251e` |
| Epoch-matched proxy | 1.3144932985 | 2531 | `347b7fbc` |

The archived-plan collector independently reverified successful artifact states, frozen fingerprints, code/runtime receipts and finite, consistent final-step metrics. The complete rows are in [canary_metrics.csv](canary_metrics.csv). Fieldbook's earlier endpoint audit also verified all three final checkpoint metadata files. No completed run was retrained during recovery.

The [trace summary](canary_trace_summary.json) covers 16,537 training-loss/LR records and 1,655 paired gradient/parameter-norm records. All scanned values are finite. The matched proxy's training loss decreases from 10.350 to 1.440; its final logged gradient norm is 0.162. Transient spikes recover, and both learning rates follow the frozen warmup, plateau and decay schedule. The target and unmatched traces remain healthy. The [combined plot](canary_optimizer_traces.png) was visually inspected, with no overlapping or cropped labels.

W&B exported 11 identical duplicate rows in each target history group and two in each proxy group. The reader retains the raw histories, removes exact duplicates for analysis and rejects conflicting records at the same step. The raw history hashes and per-metric summaries are archived alongside the plot.

The canary checks establish training and data-path feasibility at p=1. These endpoints do not establish the curve shapes or selection benefit of epoch matching; the matched proxy has higher evaluation loss at this coordinate. Retain the frozen design and analyze the common-grid sweep as prespecified. The seven-coordinate pilot and dense stage remain unreleased. This review submitted no jobs and changed no training settings, code pins, manuscript or outline.
