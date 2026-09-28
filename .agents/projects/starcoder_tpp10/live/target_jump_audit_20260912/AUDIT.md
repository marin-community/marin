# StarCoder target-curve audit: mixed BPB definitions

The jump is a metric-comparability bug in the combined figure. The PDF faithfully plots its inputs, but those inputs mix 97 results from the old BPB calculation with five results from the corrected calculation. It is not evidence of a training-loss increase at 70% StarCoder. The reported 65% target minimum and 48.9% regret reduction must be replaced.

## Evidence

All 102 final metric files were read afresh from the original GCS artifacts. The four resumed targets at 55%, 60%, 65% and 80%, and one matched-proxy 80% replicate (trainer 20260911, subset 20260914), contain `eval/bpb_schema_version=2`. The other 97 endpoints have the legacy calculation. Final steps and original run identities match the plans. Target W&B summaries agree with the saved metrics; the problem is their differing definitions.

| Target StarCoder fraction | Plotted BPB | Consistent BPB | Saved token-average loss |
| --- | ---: | ---: | ---: |
| 50% | 0.794241 | 0.775667 | 1.330683 |
| 55% | 0.770135 | 0.770135 | 1.321192 |
| 60% | 0.769439 | 0.769439 | 1.319998 |
| 65% | 0.766556 | 0.766556 | 1.315053 |
| 70% | 0.784388 | 0.765565 | 1.313352 |
| 80% | 0.770405 | 0.770405 | 1.321656 |
| 90% | 0.798773 | 0.779286 | 1.336891 |
| 100% | 0.820746 | 0.800619 | 1.373488 |

The old evaluator averaged per-batch BPB using token weights. The corrected evaluator divides total loss bits by total scored bytes. The saved token-average loss is unaffected by that correction and decreases through 70%, removing the apparent bump independently of any BPB reconstruction.

The original pilot, original refinement and batch-recovery bundles contain the same legacy evaluator. A bounded range read of the actual interactive recovery bundle proves it contains the corrected `eval.py`. Among Levanter, Marin, Haliax and Fray library sources, `eval.py` is the only changed source by ZIP CRC/size comparison; its full extracted hash matches the corrected source. The target W&B configuration comparison finds only the intended mixture weights and run/output identities changed. Model, optimizer, schedule, data caches, data seed and component shuffle keys agree.

The training guard at `experiments/domain_phase_mix/launch_starcoder_tpp10.py:44` omits `lib/levanter/src/levanter/eval.py` from its code pins. Its collector at line 239, and the refinement collector's `collect_endpoint`, check the metric name and final step without checking BPB schema. Consequently, the existing runtime receipts and W&B parity checks passed. The earlier independent arithmetic check also reproduced the mixed inputs without validating their semantics; its interpretation as a fully validated scientific figure was too strong.

## Consistent reconstruction

The pinned PALOMA population has 5,673 examples, 11,612,631 scored tokens and 28,741,166 scored bytes. For every endpoint, corrected BPB is saved token-average loss multiplied by `11612631 / (28741166 * ln(2)) = 0.582909028651567`.

The counts hash matches three previously completed, live checkpoint audits. Their corrected TPU scores agree with reconstruction within 1.23e-6 BPB. All five newly logged schema-2 endpoints agree within 1.86e-7 BPB. Two independent calculations reproduce the corrected selections and regrets. No checkpoint retraining is needed to put these saved losses on the same scale.

- Target measured minimum: **70%**, **0.765564524 BPB**. The 65% point is 0.000991735 BPB higher; one target seed does not establish a population-optimal fraction.
- Each of the three matched subset means still selects **50%**; unmatched mean still selects **100%**.
- Target excess at those selections: **0.010102816 BPB (+1.3197%)** matched and **0.035054281 BPB (+4.5789%)** unmatched.
- Matched selection avoids **71.1795%** of unmatched excess loss. Absolute target-loss advantage is **0.024951465 BPB**.
- One individual matched replicate's minimum changes from 80% to 55%; its subset's two-seed mean still selects 50%.

## Correction required

Rebuild all five plotted curves and all selection/regret statistics from consistently defined BPB, preserving the original raw measurements separately. Update the figure, caption, main paragraph and current outline facts. Make the collector reject mixed metric definitions or explicitly normalize from saved token losses and frozen population counts. Include evaluator semantics/source provenance in future runtime checks; do not rewrite the original frozen training plans.

This audit leaves the manuscript, paper inputs and scientific source code unchanged. `target_metric_schema_audit.png` is a diagnostic comparison, not the published figure. No training or evaluation jobs were submitted. CC should treat the previous complete-grid figure's numerical claims as superseded by this audit until the correction is applied.

## Reproduction and records

Run `uv run .agents/projects/starcoder_tpp10/live/target_jump_audit_20260912/analyze_metric_schemas.py` from the Marin repository. It verifies input hashes and prior checkpoint audits, then writes `consistent_bpb_audit.json`, `consistent_bpb_audit.csv` and the diagnostic PNG. `plot_audit.json` contains an independent check of PDF coordinates, all 102 input mappings and the recomputed results. `config_interactive_bundle_comparison.json` and `config_audit.md` record the cross-bundle and configuration checks. Raw JSONL metrics and the three previous checkpoint audit receipts remain alongside them.

Echo prior-work lookup returned HTTP 403; the standalone incident record could not be published there. Fieldbook and the experiment's CC change list carry the finding.
