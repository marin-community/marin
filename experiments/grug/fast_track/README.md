# fast_track — dense vs MoE scaling ladder (H100, 16k BPE)

Self-contained grug variant with 16k vocab size for fast iteration. Dense and MoE baselines shown
below from 9.4e16 to 4.3e19 FLOPs.

## Files

| file | contents |
|------|----------|
| [`launch.py`](launch.py) | ladder rungs, budget resolution (`--match`), Iris/W&B wiring |
| [`data_pipeline.py`](data_pipeline.py) | raw sources, DataKit artifact, and store mixture |
| [`model.py`](model.py) | the transformer: attention, GatedNorm, SConv, QB-routed MoE |
| [`train.py`](train.py) | trainer/eval/loss wiring and runtime (XLA) defaults |
| [`optimizer.py`](optimizer.py) | MuonH optimizer config: LR groups + hyperball step |
| [`grugmuon_stacked.py`](grugmuon_stacked.py) | Newton-Schulz orthogonalization (Muon direction) |
| [`adamh.py`](adamh.py) | AdamH scale transform (the `adamh` LR group) |
| [`heuristic.py`](heuristic.py) | compute-scaling LR / beta2 / epsilon fit |
| [`router_metrics.py`](router_metrics.py) | routing-stats telemetry (logging-only) |

## Results

These recorded runs use the existing training cache (`--source-mode cache`).
They do not measure the default testbed sample introduced here.

| size | variant | TPP | batch | active | total | steps | tokens | FLOPs | MFU | Paloma loss | Paloma bpb | uncheat bpb | runtime |
|------|---------|----:|------:|-------:|------:|------:|-------:|------:|----:|------------:|-----------:|------------:|--------:|
| d512  | dense | 20 | 128 |  18.1M |  36.8M |    690 | 0.36B | 9.4e16 | 16.6% | 3.676 | 1.520 | 1.243 | 3.7m |
| d768  | dense | 20 | 128 |  53.5M |  82.3M |  2,040 | 1.07B | 6.3e17 | 20.2% | 3.283 | 1.361 | 1.063 | 9.2m |
| d1024 | dense | 20 | 256 | 144.7M | 185.3M |  2,760 | 2.89B | 3.9e18 | 26.7% | 3.006 | 1.248 | 0.945 | 34.8m |
| d1280 | dense | 20 | 256 | 261.5M | 313.6M |  4,988 | 5.23B | 1.2e19 | 28.8% | 2.847 | 1.183 | 0.881 | 1.5 hr |
| d512  | moe   | 60 | 128 |  20.8M |   483M |  2,385 | 1.25B | 3.5e17 |  8.2% | 3.156 | 1.308 | 1.008 | 14.0m |
| d768  | moe   | 60 | 128 |  60.6M |  1.42B |  6,930 | 3.63B | 2.3e18 | 10.3% | 2.872 | 1.193 | 0.888 | 54.4m |
| d1024 | moe   | 60 | 256 | 162.0M |  3.75B |  9,270 | 9.72B | 1.4e19 | 13.9% | 2.633 | 1.096 | 0.793 | 3.8 hr |
| d1280 | moe   | 60 | 256 | 291.3M |  6.81B | 16,669 | 17.5B | 4.3e19 | 15.6% | 2.504 | 1.044 | 0.741 | 10.0 hr |

W&B runs (project `marin-community/marin_moe`) —
dense: [d512](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-dense-d512) ·
[d768](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-dense-d768) ·
[d1024](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-dense-d1024) ·
[d1280](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-dense-d1280) —
moe: [d512](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-moe-d512) ·
[d768](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-moe-d768) ·
[d1024](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-moe-d1024) ·
[d1280](https://wandb.ai/marin-community/marin_moe/runs/fasttrack-moe-d1280)

## Tokenizer impact (16k vs 128k)

Same MoE geometry and token budget, swapping the 16k BPE tokenizer for the 128k Marin (llama3-family)
tokenizer. bpb is byte-normalized so it compares fairly across tokenizers; per-token loss does not
(the 128k tokenizer packs more bytes per token, so higher loss at similar bpb).

| size | tokenizer | Paloma loss | Paloma bpb | uncheat bpb |
|------|-----------|------------:|-----------:|------------:|
| d512 | 16k  | 3.156 | 1.308 | 1.008 |
| d512 | 128k | 3.656 | 1.309 | 0.996 |
| d768 | 16k  | 2.872 | 1.193 | 0.888 |
| d768 | 128k | 3.306 | 1.187 | 0.870 |

Both runs train on the same number of tokens, but the 16k tokenizer compresses ~12% worse than the
128k, so at equal token budget the 16k run covers ~12% fewer bytes of text. Even with that data
disadvantage it lands ~neutral on Paloma bpb (+0.001 at d512, −0.006 at d768) and only slightly behind
on uncheatable bpb (−0.012 / −0.018).

## Scaling law

Fitting `L(C) = L∞ + A·C^(−α)` to Paloma macro loss over the four rungs (C = total training FLOPs), with
the irreducible floor pinned at **L∞ = 1.2**:

| variant | fit | R² | α |
|---------|-----|---:|--:|
| MoE   | `L = 1.2 + 58.67·C^(−0.0842)` | 0.99995 | 0.0842 |
| dense | `L = 1.2 + 66.70·C^(−0.0844)` | 0.99870 | 0.0844 |

![Paloma scaling law](scaling_law.png)

**Compute efficiency:** the MoE recipe (60 TPP) reaches the same Paloma loss as the compute-optimal
dense recipe (20 TPP) with **~4.2× less compute**.

## Launch commands

Set `$WANDB_API_KEY` in your shell.

`fast-track` is the single entry point. Run it locally to print the lowered plan for inspection; add
`--submit` to launch it as an 8×H100 Iris job:

```bash
# inspect the plan locally — nothing is submitted
uv run fast-track --run-id dense-d768 --size d768 --dense --version 2026.09.17

# submit it as an Iris H100 job
uv run fast-track --submit --run-id dense-d768 --size d768 --dense --version 2026.09.17
```

`--submit` wraps the launcher in `iris job run … -- python -m …launch … --run` and forwards
`$WANDB_API_KEY` to the job. The default cluster is `cw-us-east-02a`.
Use `--target-cluster` to select another cluster.

Pick a size and variant; the budget defaults to **data-matching** that variant's baseline (dense at
20 TPP, MoE at 60 TPP) at the rung's baseline batch (128 for d512/d768, 256 for d1024/d1280). Steps
are derived automatically. Use `--match compute` to FLOP-match instead, `--batch-size` to change the
batch (steps rescale to hold the match), or `--num-steps` to set the count explicitly.

Dense (data-match baseline):

```bash
uv run fast-track --submit --run-id dense-d768 --size d768 --dense --version 2026.09.17
```

MoE (data-match baseline; bump the batch — steps halve to hold tokens):

```bash
uv run fast-track --submit --run-id moe-d768 --size d768 --batch-size 256 --version 2026.09.17
```

MFU probe (any size, quick — explicit short budget):

```bash
uv run fast-track --submit --run-id probe-d1280 --size d1280 --num-steps 20 --no-eval --version 2026.09.17
```

### Useful flags (all on `launch.py`)

| flag | effect |
|------|--------|
| `--run-id` | **required** run identifier for artifact + W&B names |
| `--size` | **required** `d512` / `d768` / `d1024` / `d1280` |
| `--dense` | dense 3×hidden SwiGLU, no MoE (expert=1, normal eval) |
| `--match` | `data` (default) tokens-match or `compute` FLOP-match the variant baseline |
| `--batch-size` | override the rung's baseline batch (steps rescale to hold the match) |
| `--num-steps N` | set the step budget explicitly (ignores `--match`) |
| `--no-eval` | skip eval (clean MFU probes) |
| `--save-checkpoints` | save a permanent final checkpoint to S3 (off by default) |
| `--submit` | Submit as an Iris H100 job. Omit to print the plan locally. |
| `--target-cluster` | Select the Iris cluster for submission. The default is `cw-us-east-02a`. |
| `--source-mode` | use the existing cache, a normalized sample, or a registry source |
| `--weighting` | use token-proportional or uniform DataKit bucket weights |

Results land in W&B `marin-community/marin_moe`; eval bpb keys are `eval/paloma/macro_bpb`,
`eval/uncheatable_eval/macro_bpb` (MoE dropless eval logs under the normal `eval/` prefix).

## End-to-end data runs

The `fast-track` end-to-end mode runs the production DataKit graph at its small scale. It then reads
the cluster-by-quality store, builds the training mixture, and starts training and evaluation.

Run a short dense experiment on the curated sample:

```bash
uv run fast-track --submit --run-id data-token-weighted --size d512 --dense --source-mode sample \
    --sources cp/arxiv_abstracts,cp/wikiteam,starcoder2/ir_python \
    --num-steps 20 --batch-size 8 --weighting token_proportional --version 2026.09.23
```

Change only the mixture. The second command uses the same DataKit store:

```bash
uv run fast-track --submit --run-id data-uniform --size d512 --dense --source-mode sample \
    --sources cp/arxiv_abstracts,cp/wikiteam,starcoder2/ir_python \
    --num-steps 20 --batch-size 8 --weighting uniform --version 2026.09.23
```

Fast-track defaults to sample mode and all sources in
`s3://marin-us-east-02a/marin/datakit/sample_100b_2026_10_02`.
This sample has a 100B-token target across the current registry.
The default cluster is `cw-us-east-02a`, where the sample resides.
Use `--sources` to select a subset or `--sample-prefix` to select another completed sample.
The sample root must contain the completion record from the materialization command below.

Use `--source-mode registry --sources <name>` to start from a registered raw source.
Use `--source-mode cache` to use the existing training cache.
Use `--run` only in an Iris environment. Without `--run` or `--submit`, the command prints
the artifact plan and does not start work. Set `WANDB_MODE=disabled` to run without a W&B record. The
training mixture omits each bucket that has fewer tokens than one model sequence.
Before training, fast-track compares the usable token count with the run's token budget.
It rejects a DataKit store that is too small.

Sample mode reads an existing normalized sample. It does not create a new
sample. Registry mode includes the source download and normalization recipes.
It processes the selected sources without a token limit.

DataKit uses one fixed CPU worker pool for all Zephyr stages, including source
download, normalization, embedding, quality scoring, deduplication, and store
construction. Source recipes retain their task resource requests. Centroid
training remains a separate CPU job. Model training uses a separate 8×H100 job.
The data pool contains one worker with 120 CPUs, 1 TiB RAM, and 1 TiB disk.
Up to 64 pipeline steps can submit work to this pool at the same time. This lets
more sources supply tasks at once when each source has few shards.
The coordinator allows 68 concurrent pipelines, including capacity for the
centroid sampler's four nested pipelines.
CPU and RAM requests control concurrent task admission. Task disk requests must
fit the worker, but Zephyr does not account for concurrent disk use.

The data artifact records the terminal DataKit store identity. Changes to
upstream source recipes, tokenizer identity, or cluster configuration change
its fingerprint. A changed recipe at a fixed version produces a drift warning
and retains the cached result. Use a new version to build the changed recipe.

To produce a fresh testbed sample from the current registry:

```bash
uv run iris --cluster marin job run --no-wait \
  --job-name fast-track-sample-100b-20261002 --target-cluster cw-us-east-02a \
  --priority batch --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \
  --extra cpu --extra datakit -- \
  python -m experiments.datakit.materialize_zephyr_benchmark_sample \
  --mode regenerate --data-prefix s3://marin-us-east-02a/marin \
  --destination-prefix s3://marin-us-east-02a/marin/datakit/sample_100b_2026_10_02 \
  --target-total-tokens-b 100 --max-concurrent 8
```

The sample builder reuses completed normalized artifacts from the current recipes.
It writes the root completion record only after all source steps succeed.
Use a new destination and job name for a new sample version.

## Shuffled-token comparison

`negative_control.py` reads a completed `FastTrackDataStore` and writes a separate
training store. It shuffles tokens other than special tokens within each document,
using a fixed seed.
Document order, lengths, token counts, special-token positions, and mixture weights
stay the same. Evaluation data, model configuration, and training settings stay the same.

After the baseline finishes training and evaluation, submit the comparison with
its `FastTrackDataStore` artifact directory as `--source-store`:

```bash
uv run iris --cluster marin job run --no-wait \
  --job-name fast-track-shuffled-d512 --target-cluster cw-us-east-02a \
  --priority interactive --cpu 2 --memory 8GB --disk 32GB --enable-extra-resources \
  --extra cpu --extra datakit -e WANDB_API_KEY "$WANDB_API_KEY" -- \
  python -m experiments.grug.fast_track.negative_control \
  --source-store '<completed-baseline-data-artifact>' \
  --run-id shuffled-d512-dense --size d512 --dense --seed 0 --version 2026.10.02 --run
```

For the mixture-of-experts (MoE) comparison, omit `--dense` and select different job
and run names. The two variants reuse the shuffled store when the source, shuffle
seed, and version match. Each run uses its variant's default training budget and
saves its final checkpoint. Compare final Paloma and uncheatable bits per byte
(BPB) with a baseline of the same size, variant, training seed, and token budget.
Higher BPB means worse prediction of the evaluation data.
Use `--stop-after datakit` to build only the shuffled store.
