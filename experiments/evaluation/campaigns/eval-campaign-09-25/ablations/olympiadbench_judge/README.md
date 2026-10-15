# OlympiadBench judge ablation

Run `regrade.py` with the pinned Evalchemy project from the Marin checkout. It reads the sealed samples in `sources.yaml`, starts the local 131,072-token MiniMax judge on one H100x8 worker, and writes per-model results beneath a new object-store prefix. CoreWeave object-store credentials must be available in the environment.

```bash
CAMPAIGN=experiments/evaluation/campaigns/eval-campaign-09-25
OUTPUT_PREFIX=s3://your-bucket/olympiadbench-judge-ablation
uv run --project config/external/evalchemy-olympiadbench-judge-ablation \
  --with-editable lib/iris --with-editable lib/marin \
  python -m experiments.evaluation.campaigns.eval-campaign-09-25.ablations.olympiadbench_judge.regrade \
  --sources "$CAMPAIGN/ablations/olympiadbench_judge/sources.yaml" \
  --judge-config "$CAMPAIGN/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml" \
  --output-prefix "$OUTPUT_PREFIX"
```

Add `--validate-only` to check the source archives without starting the judge.
