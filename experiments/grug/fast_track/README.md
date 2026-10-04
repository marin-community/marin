# fast_track — dense vs MoE scaling ladder (H100, 16k BPE)

Self-contained grug variant with 16k vocab size for fast iteration. Dense and MoE baselines shown
below from 9.4e16 to 4.3e19 FLOPs.

## Files

| file | contents |
|------|----------|
| [`launch.py`](launch.py) | ladder rungs, budget resolution (`--match`), Iris/W&B wiring |
| [`data_pipeline.py`](data_pipeline.py) | raw sources, DataKit artifact, and store mixture |
| [`add_dataset.py`](add_dataset.py), [`add_dataset_cli.py`](add_dataset_cli.py) | Bounded Hugging Face prefix and simulated exposure |
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

## DataKit sample results

These experiments used the implementation through commit `d3a6671929`, before the later model/data-track changes.
These runs use the completed 25B-target sample across all 292 registered sources,
with token-proportional training weights and training seed 0.
DataKit produced 29,858,746,027 usable tokens, above the largest ladder budget.
The input target estimates token counts. Filtering and the training tokenizer
set the measured usable-token count.

All eight runs completed on October 3, 2026, with verified final evaluations and
permanent checkpoints at their full training budgets.
BPB means bits per byte. Lower scores are better.
Each macro score is the unweighted mean across evaluation dataset tags with observed tokens.

| Model | Training tokens | Paloma macro BPB | Uncheatable macro BPB |
|---|---:|---:|---:|
| [d512 dense](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-dense) | 361,758,720 | 1.5345 | 1.2554 |
| [d512 MoE](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-moe) | 1,250,426,880 | 1.3241 | 1.0166 |
| [d768 dense](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d768-dense) | 1,069,547,520 | 1.3750 | 1.0715 |
| [d768 MoE](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d768-moe) | 3,633,315,840 | 1.2067 | 0.8925 |
| [d1024 dense](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d1024-dense) | 2,894,069,760 | 1.2635 | 0.9517 |
| [d1024 MoE](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d1024-moe) | 9,720,299,520 | 1.1077 | 0.7950 |
| [d1280 dense](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d1280-dense) | 5,230,297,088 | 1.1980 | 0.8868 |
| [d1280 MoE](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d1280-moe) | 17,478,713,344 | 1.0573 | 0.7448 |

The tokenizer comparison and scaling fit below use the existing-cache results.

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

The data artifact path includes the terminal DataKit store identity and tokenizer vocabulary size.
Runs with the same recipe and version reuse that artifact, independent of their training run IDs.
Changes to source recipes, tokenizer identity, or cluster configuration select a different artifact path.
Set a new `fast-track --version` to rebuild the outer artifact while retaining unchanged upstream caches.

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

After the baseline data store finishes, submit the comparison with its
`FastTrackDataStore` artifact directory as `--source-store`.
The [end-to-end data commands](#end-to-end-data-runs) produce this artifact.
Baseline and control training can run at the same time:

```bash
uv run iris --cluster marin job run --no-wait \
  --job-name fast-track-shuffled-d512 --target-cluster cw-us-east-02a \
  --priority interactive --cpu 2 --memory 8GB --disk 32GB --enable-extra-resources \
  --extra cpu --extra datakit -e WANDB_API_KEY "$WANDB_API_KEY" -- \
  python -m experiments.grug.fast_track.negative_control \
  --source-store '<completed-baseline-data-artifact>' \
  --run-id shuffled-d512-dense --size d512 --dense \
  --shuffle-seed 0 --seed 0 --data-seed 0 --version 2026.10.02 --run
```

For the mixture-of-experts (MoE) comparison, omit `--dense` and select different job
and run names. The two variants reuse the shuffled store when the source, shuffle
seed, and version match. Each run uses its variant's default training budget and
saves its final checkpoint. Compare final Paloma and uncheatable bits per byte
(BPB) with a baseline of the same size, variant, training seed, and token budget.
Higher BPB means worse prediction of the evaluation data.
`--shuffle-seed` controls token permutation. `--seed` and `--data-seed` control model initialization and training data order.
Use `--stop-after datakit` to build only the shuffled store.

### Completed d512 controls on the 25B sample

The October 3, 2026 controls use the completed sample across all 292 sources.
The 25B target estimates input tokens from registry corpus sizes.
Filtering and the training tokenizer determine the measured usable-token count.
DataKit produced 29,858,746,027 usable tokens across 39 training buckets.
The shuffled store preserves the document and token counts of all 40 output buckets.
One output bucket contains only 974 tokens and has no training weight.

The d512 models have a width of 512.
The dense comparison uses 361,758,720 training tokens and 690 updates.
The MoE comparison uses 1,250,426,880 training tokens and 2,385 updates.
Each control matches its baseline's model, optimizer, training seed, mixture weights,
token budget, and evaluation data. Training and token shuffling use seed 0.

| Model | Paloma baseline BPB | Paloma shuffled BPB | Uncheatable baseline BPB | Uncheatable shuffled BPB |
|---|---:|---:|---:|---:|
| Dense | 1.5345 | 2.8840 | 1.2554 | 2.7451 |
| MoE | 1.3241 | 2.8311 | 1.0166 | 2.6906 |

The four runs succeeded with final evaluation records and permanent checkpoints.
Token shuffling increased BPB on the two evaluation suites.
These comparisons use one seed and do not estimate variation across seeds.

For these controls, `--source-store` is
`s3://marin-us-east-02a/marin/datakit/fast-track/ft-testbed-25b-20261002-rno-d512-dense/2026.10.02`.

W&B records: [dense baseline](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-dense),
[dense control](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-dense-negative),
[MoE baseline](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-moe),
and [MoE control](https://wandb.ai/marin-community/marin_moe/runs/ft-testbed-25b-20261002-rno-d512-moe-negative).

## Prepare the reference pools

The hero sampler and the quality sampler have separate input policies.
Both use Zephyr and produce versioned artifacts for reuse across training scales.

The hero sampler uses the main-phase weights in
`experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json` and store `store_4d2e363d`.
It reads each component in the production loader's block-shuffle order, then converts the tokens to `hero-bpe-v16384`.
The resulting `PreparedHeroSample` retains separate component caches and the fixed mixture weights.
Training does not shuffle each component a second time.
The component shuffle determines the sequence order within each source.
The mixture loader assigns source sequences to fixed-size blocks, then permutes their positions within each block.
Conversion between tokenizers changes sequence boundaries, so this is not a byte-identical production stream.
The sample includes enough data for the final complete mixture block at the requested budget.

```bash
uv run python -m experiments.grug.fast_track.hero_sample_cli \
  --data-seed 0 --version 2026.10.04
```

The existing `hero_tok/v16384_shuf/train` cache remains available for historical comparisons.
Its final shuffle comes from `shuffle_cache.py`.
The [sample build record](https://github.com/marin-community/marin/issues/9188#issuecomment-5691769554)
describes a weighted sample from the same store and a shared text corpus for four tokenizer sizes.
The earlier sample and document-cap scripts are not in this checkout.
Use the checked-in hero sampler for a reproducible new reference.

The quality sampler reads the pinned normalized corpus paths in `hero_data_paths.json`.
It applies one document-hash inclusion probability across sources. Expected token shares therefore follow available corpus token mass.
Source estimates set the initial sample size; measured token counts determine whether another pass is necessary.
The output has a stable random order that does not depend on input partitions or worker order.
There are no topic quotas or quality-score filters in this stage.

```bash
uv run python -m experiments.grug.fast_track.corpus_sample_cli \
  --seed 0 --version 2026.10.04
```

Both commands derive their default capacity from the largest default ladder rung, currently 17,478,713,344 tokens.
The quality sampler requests ten times that budget: 174,787,133,440 raw candidate tokens.
Use `--max-training-tokens` to set a different maximum for the quality sampler.
Whole-document boundaries can add a small token excess.
Use repeatable `--source <registry-name>` options for a bounded pipeline experiment; omit them for all pinned sources.
The current implementation reads all input shards on each pass and tokenizes only selected documents.
Run the preparation in the source region. Full-scale memory use and runtime require measurement.
Tokenizer content hashes are part of the artifact identity, so plan construction needs tokenizer access.
Add `--run` in an Iris CPU coordinator to execute either preparation command.

Corpus sampling and new-dataset preparation use the fixed `fast-track-long-string-v1` encoding policy.
It uses the production `BatchTokenizer` mechanism and adds a 64 MiB limit on each document's UTF-8 text.
Long strings split at the first whitespace at or after 10,000 characters. Each next piece keeps that whitespace.
Separate pieces can produce different token IDs at a split. Raw text, document IDs, and one output per document stay intact.

The policy and its limits determine new artifact names. The corpus manifest records them in `corpus.json` under `spec`.
The prepared HF artifact records `tokenization_policy`.

An oversized sampled document stops preparation. The error identifies the corpus source and document ID, or the HF repository and row number.
Preparation does not drop or truncate the document. Text batches target 256 KiB, but one permitted document can exceed that target.
These limits reduce memory use but do not guarantee a fixed worker-memory bound for every tokenizer and input.

## Add a dataset

The add-dataset track tokenizes a Hugging Face prefix, limited by the calculated token cap and `--max-rows`.
One Zephyr task streams the source in order and tokenizes bounded batches; it does not load the full dataset in memory.
It combines that prefix with the frozen Hero cache.
The frozen Hero components keep their relative proportions within the remaining `1-p` share.
A fraction `p` assigns `p` of the training tokens to the new dataset.
The frozen Hero cache receives `1-p`. Preparation does not run clustering, quality scoring, or the DataKit graph.

Training requires `--baseline-artifact <PreparedHeroSample-path>` from the checked-in hero sampler.
The control and treatment must use the same reference artifact.

The source requires an immutable Hugging Face revision, split, and text field.
Add `--subset <name>` when the repository has a named dataset configuration. Without this flag, the loader uses the repository default.
The CLI uses `hero-bpe-v16384` for preparation and training. It has no tokenizer override.
State the desired token fraction `p` and the available unique dataset tokens `N`.
The production budget `T` defaults to the hero recipe's 18.75T tokens.
Count `N` and `T` with the same production tokenizer. The default recipe uses `marin-community/marin-tokenizer`.
The smaller experiment counts its budget and prepared sample with `hero-bpe-v16384`.
Use `--target-production-tokens` only when the intended production run has a different budget.
The selected `--size` rung and training options determine the fast-track budget `B`.
The sample cap before sequence rounding is `C = min(p*B, (N/T)*B)`.
Here `N/T` is a dimensionless production-data ratio. Both `B` and `C` use fast-track tokenizer units.
This is an exposure simulation, not a conversion between tokenizers or a claim of equal byte coverage.
When the production run repeats the new dataset, `p*B/C = p*T/N` preserves the number of epochs before sequence rounding.
When the production run has more new data than it consumes, the cap supplies one fast-track pass.
The fast-track budget must not exceed the target production budget.
The loader limit rounds down to complete training sequences. A zero-sequence limit rejects the experiment.
This rounding can increase repetition. For a sequence length `L`, the loader cap is `C_seq = floor(C/L)*L`.
Expected exposure after rounding is `p*B/C_seq` epochs.
The prepared cache records the requested cap and actual document and token counts.
Whole-document preparation can exceed the cap.
The loader applies the cap once and disables its global budget scaling so that it does not apply the exposure limit a second time.

The count `N` must come from a measured count or an explicitly recorded estimate.
A first-row prefix supports a result about that prefix. Sorted data can make the prefix unrepresentative of the full dataset.
The baseline keeps its existing exposure policy. This track simulates production exposure only for the new dataset.

```bash
uv run python -m experiments.grug.fast_track.add_dataset_cli \
  --run-id new-data-d512 --size d512 --dense --baseline-artifact <hero-sample-artifact> \
  --repository org/dataset --revision <immutable-commit-hash> \
  --split train --text-field text --fraction 0.1 \
  --available-unique-tokens 100000000000 \
  --max-rows 100000 --seed 0 --data-seed 0 --version 2026.10.04
```

Use `--prepare-only` to build one bounded prefix for reuse across runs. This mode requires `--prepare-token-cap` and does not require training fields or `--run-id`.
Pass the completed artifact as `--prepared-cache` to skip Hugging Face reads and tokenization.
Do not combine this option with repository, revision, subset, split, text-field, or row-limit options.

```bash
uv run python -m experiments.grug.fast_track.add_dataset_cli \
  --prepare-only --repository org/dataset --revision <immutable-commit-hash> \
  --split train --text-field text --max-rows 100000 \
  --prepare-token-cap 1000000000 --version 2026.10.04
```

Use the completed artifact directory from the build log, not its inner token-cache directory:

```bash
uv run python -m experiments.grug.fast_track.add_dataset_cli \
  --prepared-cache <prepared-dataset-artifact> --baseline-artifact <hero-sample-artifact> \
  --run-id new-data-d512 --size d512 --dense --fraction 0.1 \
  --available-unique-tokens 100000000000 \
  --seed 0 --data-seed 0 --version 2026.10.04
```

Without `--run`, the command prints a plan. Adoption of an existing cache reads its artifact metadata.
Add `--run` inside an Iris CPU coordinator to build the cache.
For a training run, add `--run` inside the coordinator to prepare the cache and submit training.
The coordinator requires the `cpu` and `datakit` dependency extras for tokenization and storage APIs.
This does not execute the DataKit clustering or quality-filtering graph. The training stage requests eight H100 GPUs.
Increase `--max-rows` only when the selected prefix cannot supply the calculated cap.
By default, preparation calculates one cap for the largest ladder budget and reuses that prefix across rungs.
For a smaller custom production budget, preparation stops at that budget.
Use `--prepare-token-cap` to set a different preparation limit. It must be at least the selected rung's calculated cap.
Each rung applies its own sequence-aligned exposure limit once.

For a prepared hero reference, run the matched control with:

```bash
uv run python -m experiments.grug.fast_track.hero_baseline_cli \
  --baseline-artifact <hero-sample-artifact> --run-id hero-control-d512 \
  --dense --size d512 --seed 0 --data-seed 0 --version 2026.10.04
```

Use identical model settings, token budgets, and evaluation data for both runs.
Use the [data-track comparison gate](#compare-data-track-runs) before a larger run.

## Improve a quality classifier

For a training budget `B`, this track takes the first `10*B` tokens of the fixed raw pool.
It scores that raw prefix, orders documents from highest to lowest score, and includes documents until their total reaches `B` tokens.
The final whole document can exceed the target; the training loader consumes only its exact budget.
Each rung gets its own selection. A smaller rung does not use a prefix of a larger rung's selected cache.
The default model is dense d512; `--size` selects another ladder rung and its data-match token budget.

A scorer factory returns a `DocumentQualityScorer` with `scores(batch)`.
The `QualityScoringBatch` contains document text, document IDs, and optional cached embeddings.
The result contains one finite scalar per input document.
Higher scores must mean higher quality.
The factory loads the model once per Zephyr shard.
Classifier identity must include `implementation`, `revision`, and every input that can change scores, including model and embedding revisions.
Scored artifacts include the raw-pool identity and the rung's candidate token budget.
Candidate, incumbent, and random-selection comparisons can reuse those scores.

For embedding-head experiments, `fit_quality_head` in `quality.py` accepts frozen `LabelledEmbedding` rows and a `QualityHeadConfig`.
The built-in `RidgeHeadConfig` fits a regularized linear head.
Duplicate-group hashing fixes the train/development/audit split at approximately 80/10/10.
The fit helper gives the head only training rows and reports development error. It does not return audit labels.
`EmbeddingHeadScorer` consumes the batch's cached embeddings. It does not run an embedding model.
The separate feature-preparation step joins Harrier vectors by original normalized shard and row offset, then verifies every document ID.
The step reads only the requested raw prefix and stores aligned feature shards for reuse across heads.
It does not shuffle raw text or token arrays.
The stored vectors use signed int8 values. The reader converts them to float32 and applies row-wise L2 normalization.

The default label sources come from [PR #8303](https://github.com/marin-community/marin/pull/8303)
and its [joined-label producer](https://github.com/marin-community/marin/blob/a8bf53bdc953b16cf69b937af47eb7896e5d6e0b/experiments/datakit/scripts/join_glm52_labels_harrier50m.py):

- Original labels: `s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet`
- Joined full text and embeddings: `s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1`.

`quality_labels_cli` validates the join and records exact input-file hashes before head fitting.
It rejects conflicting labels and text-ID mismatches.
For duplicate embeddings, it retains the first occurrence in sorted file-and-row order and records the vector differences.
GLM quality values range from 1 (lowest) to 5 (highest). Head fitting uses the target `(quality - 1) / 4`.
The GLM `valid=false` flag identifies junk content with quality 1, which gives target 0. These rows remain in head fitting.
Labels without a joined embedding remain excluded from the raw pool, but do not enter head fitting.
The frozen manifest reports missing labels by source and content type. These counts do not impose selection quotas.
Original label text can contain excerpts. Full-text exclusion hashes therefore come only from the joined text.

```bash
uv run python -m experiments.grug.fast_track.quality_labels_cli \
  --regularization 0.001 --split-seed 0 --version 2026.10.04

uv run python -m experiments.grug.fast_track.corpus_sample_cli \
  --label-exclusion-manifest <frozen-labels-artifact>/label_exclusion.json \
  --seed 0 --version 2026.10.04

uv run python -m experiments.grug.fast_track.quality_cli \
  --raw-pool <raw-corpus-artifact> --ridge-head-artifact <fitted-head-artifact> \
  --run-id quality-ridge-d512 --size d512 \
  --seed 0 --data-seed 0 --version 2026.10.04
```

Run these commands in order with `--run` in an Iris CPU coordinator.
`quality_labels_cli` builds a `FrozenQualityLabels` artifact and then a `FittedRidgeQualityHead` artifact.
Use their completed artifact directories from the build log for the exclusion manifest and `--ridge-head-artifact`, respectively.
The corpus-sampler output is the `RawCorpusPool` directory for `--raw-pool`.
The ridge artifact supplies the head, label exclusions, and feature identity together.
The quality track does not use the content-type classifier in [PR #9495](https://github.com/marin-community/marin/pull/9495).

Exclude every labelled duplicate group from the raw candidate pool, including development and audit groups.
This prevents label examples from entering model training or the candidate selection that the evaluation measures.
The same text in another source belongs to the same group.
`frozen_label_duplicate_groups` produces this union from the supplied label rows.
For a custom scorer, write a `LabelExclusion` manifest with a fixed `label_revision` and the `duplicate_groups` array.
Groups must be SHA-256 hashes of the UTF-8 bytes in the pinned normalized corpus's `text` field.
There is no additional whitespace or case normalization at this stage.
A manifest can also contain `normalized_document_ids` with the `xxh3_128_utf8` scheme.
The GLM preparation includes every original content ID, including labels without joined embeddings.
A scorer that has no fitted labels must explicitly declare an empty group array.
Use the same manifest during corpus preparation and scoring:

```bash
uv run python -m experiments.grug.fast_track.corpus_sample_cli \
  --label-exclusion-manifest <labels-excluded.json> --seed 0 --version 2026.10.04

uv run python -m experiments.grug.fast_track.quality_cli \
  --raw-pool <raw-corpus-artifact> --scorer-factory <module>:<factory> \
  --classifier-identity '{"implementation":"candidate-head","revision":"<frozen-model-identity>"}' \
  --label-exclusion-manifest <labels-excluded.json> \
  --run-id quality-candidate-d512 --size d512 \
  --seed 0 --data-seed 0 --version 2026.10.04
```

Corpus preparation omits the declared label groups before it fills the token budget.
Scoring rejects remaining overlap by text hash or normalized content ID.
The artifact identity includes the two exclusion sets.
Neither stage requires topic clusters, source quotas, or incumbent quality bins.

Without `--run`, the commands print their artifact plans.
The quality command defaults to `--stage train`.
Use `--stage select --run` to score and prepare the selected cache without model training.
In this mode, `--training-tokens` can set a small experiment budget.
For a training run, add `--run` inside an Iris CPU coordinator.

The default scoring workers use CPUs. Use the Python builder's `worker_resources` for a scorer that requires GPUs.

For a custom head that reads cached Harrier vectors, inspect the feature plan:

```bash
uv run python -m experiments.grug.fast_track.quality_cli \
  --raw-pool <raw-corpus-artifact> --stage features --size d512 --version 2026.10.04
```

Run this command with `--run` inside an Iris CPU coordinator to build the feature artifact.
Pass that completed artifact as `--prepared-features` with the generic scorer options.
The feature budget must match the scorer's raw-prefix budget. The ridge-head command prepares and reuses these features automatically.

Use `--incumbent-scorer-factory` and `--incumbent-identity` to score the same raw prefix with a fixed incumbent.
Then use `--selection-method incumbent` for its training selection.
Use `--selection-method random` for a random-selection control.
Keep the raw pool, label exclusions, training budget, model seed, and data seed fixed across comparisons.
An unchanged selected set supplies no new treatment; reuse the matched result instead of interpreting training noise as a classifier improvement.

Selection keeps whole documents and records the token excess at the cutoff.
It keeps the selected documents in the raw pool's fixed random order before cache preparation.
This order does not depend on classifier scores.
The training source checks tokenizer content, exact rung budget, and available capacity.
A candidate whose scores are all equal on the rung fails before training; random controls can use equal scores.
A large candidate pool alone does not prove that the classifier can distinguish useful data.
Compare downstream evaluation results and inspect score and source distributions before a larger run.

## Compare data-track runs

Each data-track CLI runs one rung at a time. Select the next rung only after the matched comparison passes its declared gate.
For comparisons, use final Paloma macro BPB as the primary metric and Uncheatable macro BPB plus domain results as guardrails.
BPB means bits per byte: the model's prediction loss divided by the evaluated text's byte count.
Each macro metric averages the dataset scores in its evaluation suite. Lower values are better.
Measure matched-seed noise at d512 before selecting a non-inferiority margin.
Record that margin in BPB, domain-regression limits, and the confidence-interval method before candidate runs.
Use at least three matched model/data seeds to calculate candidate-minus-control differences.
A rung passes when the upper one-sided 95% confidence bound is below the declared margin and all declared guardrails pass.
Use a zero margin when the gate requires an improvement.
After a pass, keep the same declared gate for d768 and then d1024.
Record unresolved results as inconclusive. A nonsignificant regression does not prove non-inferiority.
For quality-head comparisons, use a second, independently sampled pool for final confirmation.
On that pool, build new candidate and control selections, then run matched training seeds and apply the same declared gate.
Repeated selection on one pool can overfit that pool.
