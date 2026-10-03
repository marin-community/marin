# fast_track — dense vs MoE scaling ladder (H100, 16k BPE)

Self-contained grug variant with 16k vocab size for fast iteration. Dense and MoE baselines shown
below from 9.4e16 to 4.3e19 FLOPs.

## Files

| file | contents |
|------|----------|
| [`launch.py`](launch.py) | ladder rungs, budget resolution (`--match`), Iris/W&B wiring |
| [`data_pipeline.py`](data_pipeline.py) | raw sources, DataKit artifact, and store mixture |
| [`add_dataset.py`](add_dataset.py) | Bounded Hugging Face prefix and simulated exposure |
| [`quality_pipeline.py`](quality_pipeline.py) | Frozen embedding head, pool audit, and selected training cache |
| [`quality_cli.py`](quality_cli.py) | Quality-track command |
| [`model.py`](model.py) | the transformer: attention, GatedNorm, SConv, QB-routed MoE |
| [`train.py`](train.py) | trainer/eval/loss wiring and runtime (XLA) defaults |
| [`optimizer.py`](optimizer.py) | MuonH optimizer config: LR groups + hyperball step |
| [`grugmuon_stacked.py`](grugmuon_stacked.py) | Newton-Schulz orthogonalization (Muon direction) |
| [`adamh.py`](adamh.py) | AdamH scale transform (the `adamh` LR group) |
| [`heuristic.py`](heuristic.py) | compute-scaling LR / beta2 / epsilon fit |
| [`router_metrics.py`](router_metrics.py) | routing-stats telemetry (logging-only) |

## Results

These recorded runs use the existing training cache (`--source-mode cache`).
They do not measure the optional DataKit testbed sample.

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
| `--seed` / `--data-seed` | Model initialization and data-order seeds. `fast-track` derives data order from `--seed` unless specified. |
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
    --sources cp/arxiv_papers,cp/wikiteam,stack-v3 \
    --num-steps 20 --batch-size 8 --weighting token_proportional --version 2026.09.23
```

Change only the mixture. The second command uses the same DataKit store:

```bash
uv run fast-track --submit --run-id data-uniform --size d512 --dense --source-mode sample \
    --sources cp/arxiv_papers,cp/wikiteam,stack-v3 \
    --num-steps 20 --batch-size 8 --weighting uniform --version 2026.09.23
```

Fast-track defaults to the frozen Hero cache. Explicit sample mode reads sources in
`s3://marin-us-east-02a/marin/datakit/sample_25b_2026_10_02`.
This sample has a 25B-token input target across all registered sources.
Each source receives a share proportional to its estimated corpus size.
The sample manifest records the source paths and explicit relative token weights.
These targets use the registry's token estimates. The usable token count depends
on filtering and the training tokenizer. The largest default ladder run requires
17,478,713,344 usable tokens.
The default cluster is `cw-us-east-02a`, where the sample resides.
Use `--sources` to select a subset or `--sample-prefix` to select another completed sample.
The sample root must contain the completion record from the materialization command below.

Use `--source-mode registry --sources <name>` to start from a registered raw source.
Use `--source-mode cache` to read the existing Hero training cache at
`s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_shuf/train`.
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
construction. Source recipes retain their task resource requests. The pipeline
coordinator runs source-recipe, embedding, quality, assignment, and centroid-sampling drivers
with bounded concurrency. These drivers do not create per-source Iris jobs.
Centroid training remains a separate CPU job. Model training uses a separate 8×H100 job.
The pipeline coordinator requests 8 CPUs and 32 GB RAM. The data pool contains
one worker with 120 CPUs, 1900 GiB RAM, and 25 TiB disk, without a GPU reservation.
This profile reserves about 94% of the allocatable CPU, RAM, and disk on an RNO2A H100 node.
Up to 64 pipeline steps can submit work to this pool at the same time. This lets
more sources supply tasks at once when each source has few shards.
The coordinator allows 68 concurrent pipelines, including capacity for the
centroid sampler's four nested pipelines.
CPU and RAM requests control concurrent task admission. Task disk requests must
fit the worker, but Zephyr does not account for concurrent disk use.

The data artifact records the terminal DataKit store identity. Changes to
upstream source recipes, tokenizer identity, or cluster configuration change
its fingerprint. A changed recipe at a fixed version produces a drift warning
and retains the cached result. Set a new `fast-track --version` to build the changed recipe.

The following command defines the DataKit sample from all registered sources:

```bash
uv run iris --cluster marin job run --no-wait \
  --job-name fast-track-sample-25b-20261002 --target-cluster cw-us-east-02a \
  --priority batch --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \
  --extra cpu --extra datakit -- \
  python -m experiments.datakit.materialize_zephyr_benchmark_sample \
  --mode regenerate --data-prefix s3://marin-us-east-02a/marin \
  --destination-prefix s3://marin-us-east-02a/marin/datakit/sample_25b_2026_10_02 \
  --target-total-tokens-b 25 --max-concurrent 4
```

The sample builder reuses completed normalized artifacts from the current recipes.
It writes the root completion record only after all source steps succeed.
The record contains the selected source paths, token target, and mixture weights.
Use `--sources <comma-separated-names>` to select sources with weights proportional
to their estimated corpus sizes. This flag and `--source-mixture` are mutually exclusive.
For a new sample version, change `--destination-prefix` and `--job-name`.
Pass that destination to `fast-track --sample-prefix`.

## Shuffled-token comparison

`negative_control.py` reads a completed `FastTrackDataStore` and writes a separate
training store. It shuffles tokens other than special tokens within each document,
using a fixed seed.
Document order, lengths, token counts, and special-token positions stay the same.
The comparison uses token-proportional mixture weights and default fast-track
training settings. Select a baseline with the same weighting, model settings,
and training budget. Evaluation data stays the same.

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

## Add a dataset

The add-dataset track tokenizes a Hugging Face prefix, limited by the calculated token cap and `--max-rows`.
It combines that prefix with the frozen Hero cache.
It preserves the relative baseline weights. A fraction `p` assigns `p` of the training tokens to the new dataset.
The frozen Hero cache receives `1-p`. Preparation does not run clustering, quality scoring, or the DataKit graph.

The default is the existing 16k-tokenizer reference cache at `hero_tok/v16384_shuf/train`.
The checkout does not link this cache to the current production mixture or phase.
Before a production comparison, verify that link or supply a verified `FrozenBaselineManifest` through the Python API.

The source requires an immutable Hugging Face revision, subset, split, and text field.
It also requires the production token budget `T` and available unique dataset tokens `N` in the same tokenizer.
For a fast-track budget `B`, it uses at most `min(p*B, N*B/T, N)` unique tokens.
Scaling unique data by `B/T` preserves the production exposure of `p*T/N` epochs when the new dataset repeats.
The loader limit rounds down to a whole global batch. A zero-batch limit rejects the experiment.
The prepared cache records the requested cap and actual document and token counts.
Whole-document preparation can exceed the cap. The loader applies the cap once and clears its global simulated-budget fields.

The count `N` must come from a measured count or an explicitly recorded estimate.
A first-row prefix supports a result about that prefix. Sorted data can make the prefix unrepresentative of the full dataset.
The baseline keeps its existing exposure policy. This track simulates production exposure only for the new dataset.

```bash
uv run python -m experiments.grug.fast_track.add_dataset \
  --run-id new-data-d512 --size d512 --dense \
  --repository org/dataset --revision <immutable-commit-hash> \
  --split train --text-field text --fraction 0.1 \
  --target-production-tokens 10000000000000 --available-unique-tokens 100000000000 \
  --max-rows 100000 --seed 0 --data-seed 0 --version 2026.10.03
```

The command prints a plan without network reads. Add `--run` inside an Iris CPU coordinator to prepare the cache and submit training.
The coordinator requires the CPU and DataKit dependencies. The training stage requests eight H100 GPUs.
Increase `--max-rows` only when the selected prefix cannot supply the calculated cap.
Use the same `--prepare-token-cap` for several rungs to reuse one larger prepared prefix.
It must be at least each rung's calculated cap. Each rung still applies its own exposure limit.

Run the control with `fast-track --source-mode cache --dense --size d512 --seed 0 --data-seed 0` and a separate `--run-id`.
Use identical model settings, token budgets, and evaluation data for both runs.

## Improve a quality classifier

The quality track fits a ridge head on frozen embeddings and GLM labels.
It selects whole documents until their token count reaches the requested fraction of a separate frozen pool.
The selected cache supplies all training tokens for the experiment.
The default fraction is 10%. The default downstream model is dense d512.

The label split depends on the duplicate-group ID and a fixed split seed.
SHA-256 of the seed and group ID assigns the entire group to one partition.
Training, development, and audit groups receive approximately 80%, 10%, and 10% of groups.
Candidate changes cannot change this split. Training and development metrics exclude audit labels.
Keep the audit labels for a separate final assessment after candidate selection; this command does not score them.
The pool must exclude all labelled duplicate groups, including duplicates from other sources.

Supply a JSON `QualityBundle` with these fields:

| Field | Meaning |
| --- | --- |
| `tokenizer`, `embedding_revision`, `embedding_scale`, `label_revision` | Pinned feature and label identities. Scale converts stored embedding coordinates to floats. |
| `incumbent_revision` | Fingerprint of the incumbent classifier and its score calibration. |
| `baseline_recipe`, `sampling_method`, `pool_seed`, `split_seed` | Frozen production recipe, `production-weighted-hash` sampling declaration, and fixed seeds. |
| `labels`, `pool` | Lists of `{ "path": "...", "sha256": "..." }` entries for Parquet files. |
| `quality_bin_edges` | Increasing incumbent-score boundaries, with one more boundary than the named quality bins. |
| `requirements` | Declared source token shares, permitted share error, quality bins, minimum bin counts, minimum duplicate groups, and maximum duplicate-group token share. |

Label rows contain `source`, `id`, `duplicate_group`, `embedding`, and `label`.
Use an immutable model commit for `embedding_revision` and a label-artifact fingerprint for `label_revision`.
Pool rows contain `source`, `id`, `duplicate_group`, `embedding`, `token_count`, `content_type`, and `language`.
They also contain `incumbent_score`, `cache_path`, `cache_row`, and `token_sha256`.
Higher labels and scores must mean higher quality.
The token checksum is SHA-256 of the document's little-endian int32 token bytes.
The input join must preserve source/document keys and use duplicate groups shared across sources.

The bundle declares the sampling procedure. The program verifies checksums, source shares, score-bin coverage, duplicate concentration, and label separation.
These checks cannot prove the sampling procedure from metadata alone. Inspect documents and the upstream sample manifest before a production comparison.
Define quality-bin boundaries before candidate evaluation. The program calculates bin membership from the incumbent scores.
Include the low and high ranges present in the production distribution.
Keep an additional diagnostic panel when rare sources or content types require greater coverage.
Do not silently add oversampled diagnostic rows to the production-weighted training pool.

```bash
uv run python -m experiments.grug.fast_track.quality_cli \
  --bundle <frozen-bundle.json> --bundle-sha256 <sha256> \
  --run-id quality-ridge-d512 --size d512 --fraction 0.1 \
  --regularization 0.01 --seed 0 --data-seed 0 --version 2026.10.03
```

Add `--prepare-only --run` for the CPU selection stage. Add `--run` without `--prepare-only` to include training.
The stage verifies each pinned file in local scratch storage before it reads embedding batches.
The largest input file must fit on local disk. Document metadata stays in coordinator memory.
Production-scale memory use and runtime have not been measured.
It writes a selected cache, `selected_ids.jsonl`, and `selection.json`.
Selected documents are shuffled before the cache is written, because the training loader shuffles blocks.
The report contains development errors by source, pool composition, duplicate concentration, cutoff ties, and token overlap with the incumbent selection.
Selection identity includes the bundle, method, fraction, regularization, and tie seed. The same selection can serve several model rungs.

Use `--selection-method incumbent` for the fixed incumbent scores and `--selection-method random` for a random-selection control.
Use the same pool, token fraction, training budget, model seed, and data seed for matched comparisons.
An unchanged selected set supplies no new treatment. Its downstream result can reuse the matched incumbent run.

For training budget `B` and selection fraction `f`, the pool requires at least `B/f` tokens.
This is an arithmetic lower bound. Keep more tokens for document boundaries and other losses.
The training source rejects a capacity-short rung. It does not increase repetition to fill the budget.
A large pool does not prove useful score variation. The declared coverage checks also reject a pool that contains only one quality range.

The CLI runs one rung at a time. Select the next rung only after the matched comparison passes its declared gate.
For comparisons, use final Paloma macro BPB as the primary metric and Uncheatable macro BPB plus domain results as guardrails.
Measure matched-seed noise at d512 before selecting a non-inferiority margin.
Confirm promising results with additional matched seeds, then d768 and d1024.
Record unresolved results as inconclusive. A nonsignificant regression does not prove non-inferiority.
Use a second, independently sampled pool for the final confirmation. Repeated selection on one pool can overfit that pool.
