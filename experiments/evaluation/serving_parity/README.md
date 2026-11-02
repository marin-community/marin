# H100 / v4 TPU score check

Run Step12 on one H100 node (eight GPUs) and one v4-32 slice, then compare
accuracy on the first 100 GSM8K test items. Both use the existing campaign
YAML and vendored Evalchemy: seed 42, greedy generation, thinking off,
2,048 output tokens, 8,192 context, and 16 concurrent requests.

The serving library uses the repository's GPU release and TPU pins, with no
`layer_types` override. The v4 job alone applies a hash-guarded attention-tile
patch in its private uv cache; it tests that patched runtime, not unmodified v4
support in the pin. H100 uses TP1/DP8 with expert parallelism; v4 uses TP16.

Land order: #10854 (pin), #10535 (multi-host launcher), then this PR.
The v4 arm reuses #10535's Ray gang, allocated ports and `tpu-name`
coscheduling across four hosts. Only rank zero runs the evaluator.

Each arm owns its server and evaluator in one Iris job, with a one-hour queue
bound, one-hour runtime bound and no retries. Use regional copies of the same
pinned Step12 checkpoint and fresh artifact prefixes. No models run locally.

```bash
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware h100 --name score-check-h100 --weights "$H100_WEIGHTS" \
  --output "$H100_OUTPUT" --submit
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware v4 --name score-check-v4 --weights "$V4_WEIGHTS" \
  --output "$V4_OUTPUT" --submit
# Once both jobs succeed:
uv run python -m experiments.evaluation.serving_parity.score_check \
  --h100 "$H100_OUTPUT/result.json" --tpu "$V4_OUTPUT/result.json"
```

Omit `--submit` to print a plan. H100 output uses S3; TPU output uses GCS.
CoreWeave credentials come from the environment. Each output retains
`eval.log`, the FineStore archive, and `result.json` with scores, item hashes
and worker wall time (startup through teardown, excluding queue).

The score check exits zero when the absolute accuracy difference is at most
five percentage points (`--tolerance 0.05`); otherwise it exits nonzero and
prints both scores. Missing, partial or different-item results are errors.
This is a coarse sanity check, not a guarantee of identical text or logits.
There is no frozen reference, teacher-forced probe, or enabled cron schedule.

Test: `uv run pytest experiments/evaluation/serving_parity/test_score_check.py`.
