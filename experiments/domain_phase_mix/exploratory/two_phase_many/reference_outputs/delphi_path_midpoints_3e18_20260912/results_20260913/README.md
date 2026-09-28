# Figure 10 midpoint results, 13 September 2026

All ten midpoint runs and their native evaluations completed. Each permanent checkpoint is step 3006; each has seven inline Uncheatable components and 104 native OlmoBaseEval Easy leaf scores, assembled into the prescribed 51-component mean. Durable training and evaluation statuses are SUCCESS. No predictor was refitted and no new job was submitted during this analysis.

`midpoint_scored_results.csv` is the figure input. It uses the frozen predictions at the actual 1/2048-rounded runtime mixture. `raw_midpoint_results_not_for_plot.csv` retains the unadjusted logged aggregates and is deliberately not the figure input. `score_midpoints.py` reproduces the final scores from the archived raw measurements and unchanged frozen objective weights.

| Objective | Baseline | Measured midpoint | MARINER prediction | Baseline prediction | MARINER absolute error | Baseline absolute error |
|---|---|---:|---:|---:|---:|---:|
| Uncheatable | Olmix | 0.985230 | 0.986294 | 0.991217 | 0.001064 | 0.005988 |
| Uncheatable | Quadratic | 0.982232 | 0.981761 | 0.980957 | 0.000471 | 0.001275 |
| Uncheatable | Natural cubic spline | 0.982676 | 0.982094 | 0.983889 | 0.000582 | 0.001213 |
| Uncheatable | RegMix | 0.987572 | 0.985693 | 1.005856 | 0.001878 | 0.018284 |
| Uncheatable | Hellinger kernel ridge | 0.988633 | 0.985441 | 0.967461 | 0.003192 | 0.021172 |
| OlmoBaseEval Easy | Olmix | 1.064023 | 1.072052 | 1.096653 | 0.008029 | 0.032630 |
| OlmoBaseEval Easy | Quadratic | 1.070183 | 1.066321 | 1.027848 | 0.003863 | 0.042336 |
| OlmoBaseEval Easy | Natural cubic spline | 1.073217 | 1.066926 | 1.036847 | 0.006292 | 0.036371 |
| OlmoBaseEval Easy | RegMix | 1.075381 | 1.068969 | 1.104587 | 0.006413 | 0.029206 |
| OlmoBaseEval Easy | Hellinger kernel ridge | 1.066181 | 1.067070 | 1.033188 | 0.000889 | 0.032993 |

MARINER is closer at all ten midpoints. Its mean absolute error is 0.001437 BPB on Uncheatable and 0.005097 on the suite; baseline errors average 0.009586 and 0.034707. Each midpoint has only trainer seed zero. The quadratic and spline differences on Uncheatable are small. These measurements assess interior calibration, not statistical superiority or optimality. The suite Olmix and kernel-ridge blends measure below the seed-zero MARINER endpoint (1.068202 BPB) by 0.004179 and 0.002021 BPB, respectively.

## Metric comparability

The new inline evaluator logs BPB schema 2. Its parent Uncheatable aggregate divides total scored loss bits by total scored bytes. Historical training outcomes and the frozen predictors instead use fixed weights on the seven task BPBs. Using the new parent aggregate directly would introduce an approximately 0.008–0.011 BPB artificial increase. The figure therefore applies the exact serialized `fit_uncheatable.json` task weights to the seven newly measured component BPBs. The weights are not renormalized or estimated from these outcomes; their saved sum is 1.000000020528133. No empirical offset is added.

There remains a small difference in the component estimator: historical tagged evaluations averaged batch BPBs with token weights; schema 2 pools loss bits and scored bytes within each component. Exact historical batch BPB cannot generally be recovered from saved mean losses alone. The independent bridge audit reconstructed pooled component BPBs from the old saved token losses and the fixed new BPB/loss ratios. Across 98 logged evaluations from the 24 historical endpoint runs, the fixed-weight pooled aggregate was 0.0000389–0.0000538 BPB below the historical aggregate; for their 24 final checkpoints the difference was 0.0000389–0.0000460. This is an empirical discrepancy, not a universal error bound. It is below the figure's precision and much smaller than the smallest observed advantage in absolute error (0.000632 BPB). `uncheatable_metric_bridge_audit.json` and `metric_bridge/` retain the derivation and row-level evidence.

`runtime_identity_audit.json` compares all ten new executor configurations with all 24 historical endpoint configurations. Validation cache paths, source formats, split and packing settings, tags, evaluation frequency, architecture, token horizon and batch parameters are identical. All ten executed mixture vectors equal the frozen candidate CSV, including both historical phase fields. Trainer and data seeds match the specification. This confirms the same declared evaluation population; no new evaluation corpus or normalization fitted to outcomes was introduced.

The native OlmoBaseEval Easy evaluator does not use Levanter's tagged BPB accumulator. It computes each instance's continuation BPB using the original UTF-8 byte denominator, averages instances within each task, applies the published MMLU subject weights, then averages 51 components. All ten midpoint sidecars match the request-set path, version, OLMo-Eval source SHA and per-task instance counts of all 13 prior comparator endpoint evaluations. Each has 104 leaf tasks and 51 finite components, with the logged macro reproduced within 1e-14. Native scoring sources have no local change and last changed before these batches. See `native_obe_comparability.json`.

## Reproduction and receipts

Run `uv run collect_midpoints.py` to verify and reuse small archived GCS metadata/results. Run `uv run score_midpoints.py` to recreate the scored CSV without network access or model fitting. The collection caches immutable raw artifacts and checks the saved completion snapshot. `midpoint_scoring_receipt.json` hashes the final CSV and all scoring sources. `collection_receipt.json`, the runtime identity audit, native evaluator audit, and each raw artifact's receipt preserve checkpoint, evaluation and prediction provenance.
