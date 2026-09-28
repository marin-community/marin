# Figure 10 midpoint validation

Submitted at 2026-09-12 22:17:45 UTC as `/calvinxu/dm-delphi-3e18-path-midpoints-20260912` after CC review passed. The interactive CPU coordinator was confirmed running. The ten planned runs, frozen predictors, review and Iris acknowledgment are recorded in Fieldbook. See `submission/CC_REVIEW.md`, `submission/CC_RESOLUTION.md` and `submission/submitted.log`; the active prediction receipt is `prediction_freeze/summary.json`.

Calvin requested one measured midpoint per baseline column and objective on 12 September 2026, with CC review before submission. This extends the existing registered experiment, `exp_01m21f631n71w9xe9n6krpr2kw`.

The batch has ten new single-phase runs: five paths for Uncheatable and five for OlmoBaseEval Easy (Table-9). Each path joins that objective's MARINER proposal to the baseline proposal already plotted and measured. The baselines, in figure order, are Olmix, quadratic, natural cubic spline, RegMix trees and Hellinger kernel ridge. The exponential benefit simplification has no column and is not included.

## Mixtures and predictions

The exact candidate is the arithmetic midpoint of the two frozen runtime weight vectors. MARINER uses `lwspu_u_snc_cap06` / `lwspu_t9_snc_cap08`. Olmix uses the matched-Qwen policies `olmixq_u_kl0p05_cap04` / `olmixq_t9_kl0p005_cap04`. The other endpoints are the corresponding `cmp_u_*` / `cmp_t9_*` runtime rows in `delphi_comparator_proposals_3e18_20260909/solutions.csv`.

Both endpoints use the 1/2048 sampler grid; their exact midpoint may require 1/4096. The batch preserves sampler block size 2048 and rounds with the reference largest-remainder routine, breaking ties by alphabetical bucket order. Rounding uses no surrogate prediction or optimization. The ten TV deviations range from 0.001220703125 to 0.003173828125. Exact and runtime coordinates are retained in `exact_midpoints.csv`; only `candidate_weights.csv` is trained. Candidate IDs end in `cap64` to satisfy the existing loader. This is inactive metadata, not a fitted or imposed regularization cap.

Both MARINER and the relevant baseline had predictions frozen at the exact and runtime-rounded midpoint before submission. Reconstructed predictors match all 1,220 original figure path predictions within 2.22e-16 BPB. MARINER loads its saved objective fits; Olmix loads its saved laws. The other baseline predictors are reconstructed using only the original 280-run swarm, with parity required against all archived Figure 10 path predictions. All ten paths are retained regardless of prediction gaps or eventual measurements. Small weight changes can cross RegMix tree thresholds, so exact and rounded predictions are reported separately.

## Training and evaluation

Every run uses the original Qwen3 configuration: 358,304,128 total parameters, 128,469,376 nonembedding parameters, width 896, ten layers, sequence length 4096, batch size 128, 3,007 training steps and 1,576,534,016 tokens. The final checkpoint is step 3006. The nominal training budget is 3e18 FLOPs per run, 3e19 for the batch, excluding evaluation.

Trainer seed is 0 throughout. Uncheatable uses data seed 666200; Table-9 uses 662009. These match the seed-0 endpoint runs. The historical subset configuration and 6,325,183,647,689-token exposure reference are preserved. The data builder removes zero-weight buckets before allocating shuffle keys, so equal seeds do not guarantee identical subsets when active buckets differ. Describe the comparison as seed-matched, without claiming exact corpus pairing.

All ten runs receive inline Uncheatable evaluation and the existing native 51-component Table-9 evaluation. Each path's primary analysis uses its own objective's signed and absolute prediction errors at the runtime mixture. The other objective is supplementary. One seed supplies a descriptive interior calibration check, not an uncertainty estimate or a prespecified claim of superiority.

## Release and recovery

Use the new `launch_delphi_path_midpoints_3e18.py` and its frozen candidate hash. Submit a CPU parent in us-east5-a, with interactive v6e-8 children in us-east5-b and all data, checkpoint, executor and evaluation paths under gs://marin-us-east5. Release all ten runs concurrently; Iris schedules capacity. Preserve executor names and outputs so reruns reuse completed training/evaluation and checkpoints.

Before submission: independent scientific audit, launcher dry-run and sampler audit, frozen-prediction parity, CC review of the concrete artifacts, region validation of the exact command, and Fieldbook registration of each run and the submitting parent. Record the CC outcome and actual Iris acknowledgment in the submission handoff. No cluster restart or additional seeds are authorized by this batch.
