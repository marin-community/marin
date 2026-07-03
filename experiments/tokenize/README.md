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
| `corpus.py` | Track C: builds the ~1.5 GB English/code/math raw-text sample `train_tokenizers.py` learns from, as a lazy `raw_download` `ArtifactStep` (build-opt-in under `--run`, cacheable by name@version). |
| `superbpe_trainer.py` | Track C: trains plain BPE (stock Rust `BpeTrainer`) and SuperBPE (two-stage superword BPE, a from-scratch reimplementation of arXiv:2503.13423's algorithm — see the module docstring for why) from raw text. |
| `train_tokenizers.py` | Track C: the sweep — plain BPE and SuperBPE at a range of (vocab, transition-point) configs — trained on `corpus.py`'s sample; exports each as an HF `tokenizer.json` + `tokenizer_config.json`. |
| `push_trained_tokenizers.py` | Track C: stages each trained tokenizer into the `mirror://tokenizers/...` cache `levanter.load_tokenizer` reads, so cluster workers can load a trained arm by name with no code changes. |

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

## Track C: train our own tokenizers

Trains plain-BPE/SuperBPE tokenizers on the grug-moe data mix instead of borrowing off-the-shelf
ones; see `EXPERIMENT_LOG.md` EXP-008 for the sweep and results.

```bash
# 1. build the ~1.5 GB English/code/math training corpus (a lazy ArtifactStep; run once, then
#    cached by name@version). CPU-only; on a big CPU node, submit as an iris job (see below).
uv run python -m experiments.tokenize.corpus --run

# 2. train the sweep (11 configs in train_tokenizers.TRAIN_SPECS) — --jobs N trains N arms
#    concurrently in separate processes.
uv run python -m experiments.tokenize.train_tokenizers \
  --corpus-dir <output_path printed by step 1> \
  --out-dir experiments/tokenize/results/trained_tokenizers --jobs 11

# 3. stage every trained tokenizer where levanter.load_tokenizer("trained/<name>") finds it
uv run python -m experiments.tokenize.push_trained_tokenizers \
  --tokenizers-dir experiments/tokenize/results/trained_tokenizers

# 4. fertility pre-filter the trained arms against the usual baselines
uv run python -m experiments.tokenize.fertility_report \
  --arms trained-bpe-64k,trained-bpe-96k,trained-bpe-128k,trained-superbpe-96k-t38k,trained-superbpe-96k-t77k,trained-superbpe-128k-t51k,trained-superbpe-128k-t102k,trained-superbpe-160k-t64k,trained-superbpe-160k-t128k,trained-superbpe-64k-t32k,trained-superbpe-80k-t40k,superbpe-128k,gpt-neox-50k,marin-128k \
  --out experiments/tokenize/results/fertility_trained.json

# 5. score with the same English-dominant weighting EXP-002 used
uv run python -m experiments.tokenize.bakeoff_analysis \
  --fertility experiments/tokenize/results/fertility_trained.json \
  --domain-weights english_web=0.8,math=0.2
```

Steps 1-2 are CPU-heavy; on cw-rno2a (a single 192-vCPU/1.5 TB node), submit them as an `iris`
job rather than running locally, e.g.:

```bash
uv run iris --cluster=cw-rno2a job run --cpu 128 --memory 1000GB --extra cpu --enable-extra-resources \
  -e RAYON_NUM_THREADS 12 \
  -- python -m experiments.tokenize.train_tokenizers --corpus-dir s3://... --jobs 11
```

Leave headroom below the node's full capacity — the iris controller for cw-rno2a runs on that
same single node, and a job that claims nearly all of it can starve the controller's own
rescheduling if it restarts for an unrelated reason.
