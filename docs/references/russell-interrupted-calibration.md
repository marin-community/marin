# SFT evaluation after interrupted calibration

Use `experiments.post_training.russell_rsi.launch_interrupted_calibration_sft` for the separate `evaluate-interrupted` stage. The coordinator runs locally in the foreground. It opens the existing CW02 Iris client and waits for the remote workers. It does not submit a CPU coordinator job.

The stage requires a pinned interruption amendment and the original post-SFT configuration. Only the version and the two interruption pin fields can change. The amendment must record `incomplete_infrastructure`, a null signal gate, no RL authorization, no remaining whole-cohort replacement, and no repeated issued sample. The stage verifies all eight terminal evidence pins. It retains the original qualified model identity, SFT export qualification, source replay, bank, panel, and incumbent evidence barriers.

The graph contains SFT coding evaluation, retention evaluation, and the existing selection stage. It contains no calibration or RL execution. Coding uses the same 32 HumanEval+ and 32 MBPP+ tasks. Retention uses the same three tasks. Selection keeps the incumbent scores of 25/32 and 27/32, retention 1/3, and the existing strict coding gain and retention gates.

Use a new version and the `russell-rsi-interrupted-calibration-sft-only-v1` output namespace. Original calibration artifacts and journals stay unchanged. The selection directory contains `post-sft-selection.json` and `calibration-interruption.json`; the latter retains the interruption pin and the undetermined signal status.

Workers use CW02, batch priority, zero failure and preemption retries, and a six-hour deadline. Retention keeps the existing eight-H100, 32-CPU, 512-GB memory, 2-TB disk limits. The coding stage keeps the existing H100x8 serving and Evalchemy worker limits. Evalchemy uses `max_retries=1`: the pinned lm-eval client interprets this field as the total attempt budget, so it permits one initial request and zero retries. See the [pinned lm-eval request loop](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.12/lm_eval/models/api_models.py).

Retention seals an `EvaluationJournal` before model startup. Completed attempts reconstruct without inference. An incomplete reservation or changed binding refuses replay. Coding reserves its whole batch before server startup and retains the existing per-evaluation records. An incomplete coding batch refuses replay; it does not issue a replacement for partial results.

The coordinator needs an approved source review with `status`, `source_path`, and the full `source_head`. Its checkout must be clean. The source guard checks loaded branch package paths and the actual installed SkyRL f124 commit. Set the branch source roots in `PYTHONPATH` and use the validated f124 interpreter; do not use the current primary runtime by default.

```bash
python -m experiments.post_training.russell_rsi.launch_interrupted_calibration_sft \
  --config-uri PINNED_CONFIG_URI --config-sha256 CONFIG_SHA256 \
  --source-review-uri SOURCE_REVIEW_URI --source-review-sha256 SOURCE_REVIEW_SHA256 \
  --stage evaluate-interrupted --version 2026.10.06.7 --max-concurrent 1
```

This prints the plan. Add `--run` only to execute the reviewed request in the foreground. Do not detach the process. The coordinator uses Iris controller proxy routes for readiness, metrics, and evaluator endpoints; it needs no direct access to a private CW02 serving address.
