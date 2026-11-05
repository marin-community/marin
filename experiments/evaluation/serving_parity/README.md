# Snowball serving logprob check

Runs the 64 frozen June Snowball prompts through H100 and v4-32 vLLM, covering
256–32768 token buckets with the exact prompt token IDs.

## Run

```bash
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware h100 --name logprob-h100 --output "$H100_OUTPUT" --submit
uv run python -m experiments.evaluation.serving_parity.launch \
  --hardware v4 --name logprob-v4 --output "$V4_OUTPUT" --submit
```

Without `--submit`, prints the plan. Use fresh output prefixes. Each successful
arm writes `result.json` with per-case errors, the bound, and wall time.
Iris logs retain backend startup and per-case probability diagnostics.

## Pass

Each arm independently checks coverage of the frozen top-25 tokens, maximum
absolute probability error within `VLLM_MAX_PROBABILITY_ERROR`, and a greedy
winner supported by that error. The long-prompt sentinel must repeat exactly
on each data-parallel rank. Partial runs never publish a passing result.

## Notes

The exact June d819 export is fixed; TPU uses its verified central2 mirror.
H100 uses eight DP/EP ranks; v4 uses four hosts at TP16 with one DP rank.
Each job has a one-hour runtime, a bounded queue, and no automatic retries.
