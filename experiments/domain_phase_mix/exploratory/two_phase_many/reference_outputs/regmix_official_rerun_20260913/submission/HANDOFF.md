# Official RegMix validation submission — 13 September 2026

The four authorized validations were submitted at 08:55 PDT: the new RegMix endpoint and its 50% MARINER blend for each of Uncheatable and OlmoBaseEval Easy. [Iris job](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fdm-delphi-3e18-regmix-reference-20260913). The coordinator is running and all four training children are queued for east5-b TPU capacity. No failures were reported in the initial check.

Each point uses Qwen3 360M, 1,576,534,016 training tokens, trainer seed 0, and the same data seed as the earlier validation for that objective: 666200 for Uncheatable and 662009 for the suite. The batch costs 1.2e19 nominal training FLOPs plus evaluation. It uses interactive priority, an east5-a CPU coordinator and east5-b v6e-8 training/evaluation jobs, with all four training runs released concurrently.

CC approved the launcher with no blocking issues. Final preflight resolved four training steps and four dependent native evaluations, checked all 180 GCS paths in east5, verified the seven inline Uncheatable components and the configured path to each final step-3006 checkpoint, and checked 455 required source files in the upload. One formatting change after review left the Python AST and all eight output paths unchanged. Frozen mixture counts were copied without further rounding.

All paths in this paragraph are relative to the submission directory given below. Predictions for the four mixtures are frozen in `validation_predictions.csv`. `collection_protocol.json` names the seven Uncheatable component fields and weights: compute their weighted sum at step 3006 from each training output's `checkpoints/eval_metrics.jsonl`. Lower BPB is better. The current native Uncheatable aggregate differs from the paper's objective. The native OlmoBaseEval Easy evaluator runs after each final checkpoint. Existing paper measurements remain associated with their original adapted-RegMix mixtures until these measurements are available.

Fieldbook experiment: `exp_01m21f631n71w9xe9n6krpr2kw`; parent record: `job_01m2dqhmx2kk87t244sgwz6c3j`. Source artifacts, exact launch command, hashes and resolved output paths are in `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/regmix_official_rerun_20260913/submission`. `full_graph_regional_preflight.json` maps candidate IDs to exact training and evaluation output paths; `submission_receipt.json` maps them to live child jobs. `submit.sh` records the resume command. Preserve the source and candidate hashes when resuming so the executor reuses completed outputs. Save final measurements to `measured_validation.csv` in the official rerun output directory, then record them in Fieldbook and update the paper after checking the frozen prediction residuals.

| Candidate ID | Objective | Mixture |
| --- | --- | --- |
| `rgref_u_endpoint_cap64` | Uncheatable | Official RegMix proposal |
| `rgref_u_midpoint_cap64` | Uncheatable | 50% official RegMix + 50% MARINER |
| `rgref_t9_endpoint_cap64` | OlmoBaseEval Easy | Official RegMix proposal |
| `rgref_t9_midpoint_cap64` | OlmoBaseEval Easy | 50% official RegMix + 50% MARINER |

Each blend uses the MARINER proposal optimized for that same objective. “Official RegMix” refers to the pinned notebook regression and proposal replay in the accompanying REPORT.md; “adapted RegMix” refers to the earlier tuned-tree implementation and its different proposal sampler. The `cap64` suffix is inactive loader metadata; all four mixtures stay below ten materialized epochs.
