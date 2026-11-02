# H100 vs v4 score check

Checks that Snowball Step12 gives the same GSM8K accuracy on TPU as on GPU.
It runs the first 100 GSM8K test items, greedy, on one H100 node and on one
v4-32 slice. It passes if the two accuracies are within 5 percentage points.

## Run

```bash
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware h100 --name check-h100 --weights "$H100_WEIGHTS" --output "$H100_OUTPUT" --submit
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware v4 --name check-v4 --weights "$V4_WEIGHTS" --output "$V4_OUTPUT" --submit

# after both jobs succeed
uv run python -m experiments.evaluation.serving_parity.score_check \
  --h100 "$H100_OUTPUT/result.json" --tpu "$V4_OUTPUT/result.json"
```

Without `--submit`, `launch` prints the job plan. Use a new output prefix for
every run. Each run writes `result.json` (scores and timings) and `eval.log`.

## Notes

- v4 needs an attention-tile patch that the pinned TPU runtime lacks. The v4
  job applies it inside the job only.
- The v4 job uses the multi-host launcher: four hosts, one model at TP16.
- Missing or partial results fail the check.

Test: `uv run pytest experiments/evaluation/serving_parity/test_score_check.py`
