# Tokenizer FLOP-equivalent bake-off

Measure how much real, **compute-fair** uplift grug-moe gets from an alternative tokenizer.
Quality is bits-per-byte (BPB, tokenizer-agnostic); cost is priced at the deployment target
under a configurable serving model; the two are combined into a FLOP-equivalent BPB (feBPB).
Design and rationale: [`.agents/projects/20260703_tokenizer_flop_equivalent_bakeoff.md`](../../.agents/projects/20260703_tokenizer_flop_equivalent_bakeoff.md).

The pipeline is deliberately split so results can be **re-scored under new assumptions without
retraining**: training logs raw `(training-FLOPs, BPB)` points and fertility logs raw
token/byte counts; the cost model is applied only at analysis time.

## Modules

| Module | Role |
|---|---|
| `bakeoff_tokenizers.py` | Registry of tokenizer arms (`name`, HF ref, vocab size, design axis). |
| `flop_equivalent.py` | The scoring core: `ServingCostModel` (target shape, context, attention sparsity, speed), BPB-curve fit, and `febpb`. Run it directly for a self-check. |
| `fertility_report.py` | Phase-1 pre-filter: measures tokens/byte per arm over a held-out corpus and writes raw counts to `fertility_raw.json`. No training. |
| `launch_bakeoff_ladder.py` | Submits the isoFLOP ladder (one grug-moe run per arm × compute point) to a cluster. Prints the plan by default; `--run` submits. |
| `collect_metrics.py` | Reduces one finished run's json_logger stream to its `(train_flops, BPB)` point; the shared parsing used by `collect_ladder.py`. |
| `collect_ladder.py` | Failure-safe ladder collection: queries job state first (`iris job list --json`), pulls logs only for succeeded jobs (per-job timeout, skip+report on failures), and writes `ladder.json`. Discovery (`--prefix`) or explicit (`--point arm=job`) mode. |
| `bakeoff_analysis.py` | Re-scores stored fertility + ladder results under a `ServingCostModel` chosen on the CLI (context, sparsity, target size, speed, serving ratio, domain mix). |
| `../grug/moe/launch_tokenizer_bakeoff.py` | One grug-moe proxy run for a single arm: `BAKEOFF_ARM` sets both data tokenization and model `vocab_size`; held-out validation attached; `compute_bpb` on. |

## Run it

```bash
# 0. sanity-check the scoring model (no cluster, no data)
uv run python -m experiments.tokenize.flop_equivalent

# 1. fertility pre-filter — ranks arms by serving cost before any GPU time
uv run python -m experiments.tokenize.fertility_report --max-mb 4 --out fertility_raw.json

# 2. isoFLOP ladder — one grug-moe proxy per (arm, compute point) on cw-rno2a
#    prints the plan first; add --run to submit
uv run python -m experiments.tokenize.launch_bakeoff_ladder --arms marin-128k,gpt-oss-200k,qwen3-152k --run

# 3. collect each finished run's (train_flops, BPB) point into a ladder file
uv run iris --cluster=cw-rno2a job logs <run-id> \
  | uv run python -m experiments.tokenize.collect_metrics run --arm marin-128k --metrics -
uv run python -m experiments.tokenize.collect_metrics ladder \
  --point marin-128k=marin.log --point gpt-oss-200k=gptoss.log --out ladder.json

# 4. score — re-runnable under any deployment assumption
uv run python -m experiments.tokenize.bakeoff_analysis --fertility fertility_raw.json --bpb ladder.json
uv run python -m experiments.tokenize.bakeoff_analysis --fertility fertility_raw.json --bpb ladder.json \
  --context-len 65536            # replay at 64k serving context
uv run python -m experiments.tokenize.bakeoff_analysis --fertility fertility_raw.json --bpb ladder.json \
  --target-hidden 8192 --target-layers 80   # replay for a larger deployment model
```

Step 4 recomputes the Pareto frontier and feBPB from the stored measurements — change the
cost assumptions and re-run to answer "what's optimal if we serve at 64k / a 400B model / a
code-heavy traffic mix?" without touching the cluster.
