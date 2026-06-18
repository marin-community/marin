# 300M MDE Vertex Experts Plan

## Goal

Train true single-domain MDE-style vertex experts for the existing 39-domain `300m_6b` data-mixing topology, score those experts on all existing smooth feature surfaces, compact the results into local dense matrices, and test whether vertex-expert features improve mixture-to-metric modeling across all available 300M metrics.

## Motivation

The existing `300m_checkpoint_features_full_swarm` artifacts are useful but are not true MDE vertex experts: they score mixture-trained checkpoints. The MDE paper's strongest interpretation uses single-domain experts that approximate domain conditionals. The 300M/6B setting is the clean first test because, without simulated epoching, domain-only training at a 6B budget has mild repetition: median `0.049` epochs, q95 `0.415` epochs, and only two domains exceed one epoch.

## Scope

- Train `39` cap-1 vertex experts: one per `DOMAIN_NAMES` entry in `experiments/domain_phase_mix/two_phase_dolma3_dolmino_top_level.py`.
- Train `2` full-6B over-epoch controls for the only domains where 6B exceeds one epoch: `dolma3_wikipedia` and `dolmino_stem_heavy_crawl`.
- Score all `41` checkpoints on the same feature surfaces used by checkpoint-feature extraction:
  - raw-text loss features for `paloma`, `uncheatable`, `raw_ppl_priority`, and `agentic_coding`;
  - teacher-forced GSM8K/HumanEval request features;
  - MCQ smooth-proxy request/choice features.
- Compact each surface into local dense matrices small enough for local modeling.
- Fit and compare DSP-only, MDE-only, and DSP+MDE models for every sufficiently covered target metric in `raw_metric_matrix_300m.csv`.

## Non-Goals

- Do not train production-167p experts yet.
- Do not change the partitioning or add data.
- Do not use simulated epoching for vertex-expert training.
- Do not claim arbitrary-mixture improvement until features are used through a queryable mixture-to-feature map.
- Do not submit live Iris jobs until implementation has CC review and east5 launch-safety validation.

## Statistical Design

### Vertex Expert Definition

For domain `i`, train a checkpoint `E_i` on only that domain:

```text
phase_0/domain_i = 1.0
phase_1/domain_i = 1.0
all other domains = 0.0
target_budget = None
num_train_steps_i = floor(min(6B, tokens_i) / (batch_size * seq_len))
```

Using the same one-hot weights in both phases preserves compatibility with the two-phase training interface while making the data distribution single-domain throughout training.

### Controls

The cap-1 experts stop at one epoch for small domains. Two controls intentionally train the small domains for full 6B:

```text
dolma3_wikipedia: 6B / 3.669B = 1.635 epochs
dolmino_stem_heavy_crawl: 6B / 5.214B = 1.151 epochs
```

These controls test whether the 1-epoch cap loses useful expert quality or whether mild repetition is harmless.

### LR Schedule

For each expert, recompute optimizer schedule from its actual `num_train_steps_i`. Keep optimizer family, peak LR, batch size, sequence length, tokenizer, model config, and initialization policy aligned with `300m_6b`. Scale warmup/decay by actual steps rather than reusing an absolute 6B schedule for short runs.

Implementation detail: `create_two_phase_dolma3_dolmino_top_level_optimizer_config(...)` already derives WSD timing from `experiment_budget`, `batch_size`, and `seq_len`; the expert launcher should pass per-expert `experiment_budget_i` and `target_budget=None`.

## Implementation Plan

### 1. Add Expert Training Launcher

Create:

```text
experiments/domain_phase_mix/launch_300m_mde_vertex_experts.py
```

Responsibilities:

- Load `DOMAIN_NAMES` and `TOP_LEVEL_DOMAIN_TOKEN_COUNTS`.
- Compute one run spec per domain:
  - `run_name = mde_vertex_cap1_<domain_slug>`;
  - `domain_name`;
  - `domain_tokens`;
  - `train_tokens = min(6_000_000_000, domain_tokens)`;
  - `num_train_steps = train_tokens // (128 * 2048)`;
  - `materialized_epochs = train_tokens / domain_tokens`;
  - `is_control = false`.
- Add two full-6B controls:
  - `mde_vertex_full6b_dolma3_wikipedia`;
  - `mde_vertex_full6b_dolmino_stem_heavy_crawl`;
  - `train_tokens = 6_000_000_000`;
  - `is_control = true`.
- For each expert run, build a separate `MixtureExperiment` through `create_two_phase_dolma3_dolmino_top_level_experiment(...)` with:
  - `model_config=regmix_300m_proxy`;
  - `optimizer_config=regmix_300m_muonh_base`;
  - `target_budget=None`;
  - `experiment_budget=train_tokens_i`;
  - `runtime_cache_region="us-east5"`;
  - `resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"], zone="us-east5-a")`;
  - `eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS` or a deliberately empty task tuple if training eval harness should be skipped and all eval is deferred to feature scoring.
- Use one-hot `WeightConfig` rows with identical `phase_0` and `phase_1` weights.
- Enumerate all 39 domains explicitly in each phase weight dict, with the active domain at `1.0` and all others at `0.0`. This avoids relying on implicit missing-weight padding for the core training semantics.
- Do not use one shared `MixtureExperiment` for all experts: `MixtureExperiment` stores `num_train_steps` and `optimizer_config` at construction time, so per-expert budgets and LR schedules require per-expert instances.
- Patch the type annotation on `create_two_phase_dolma3_dolmino_top_level_experiment(..., target_budget=...)` from `int` to `int | None` if needed so `target_budget=None` passes type checking.
- Write local manifests:
  - `mde_vertex_expert_training_manifest.csv`;
  - `mde_vertex_expert_run_specs.json`;
  - `mde_vertex_expert_epoch_summary.csv`;
  - `summary.json`.
- Support:
  - `--dry-run`;
  - `--include-domain`;
  - `--skip-controls`;
  - `--max-concurrent`;
  - `--executor-prefix`;
  - `--tpu-region`, `--tpu-zone`, `--tpu-type`.

Dry-run assertions:

- Exactly `41` runs by default: `39` cap-1 plus `2` controls.
- Every cap-1 run has `materialized_epochs <= 1.0`.
- The two control runs have `materialized_epochs > 1.0`.
- `target_budget is None` in generated training configs.
- Each training step has the intended per-expert `num_train_steps_i`; cap-1 small-domain runs must not silently train for full 6B.
- The optimizer schedule is recomputed from `experiment_budget=train_tokens_i`.
- All runs are `us-east5-a`, `MARIN_PREFIX=gs://marin-us-east5` at parent launch, and no central-region paths are introduced.

### 2. Register Runs in Fieldbook

Add a new Fieldbook experiment, unless the user chooses to attach this to `exp_01ksq566229zr47dntr7a81yvy`:

```text
name: MDE vertex experts 300M
tags: mde, vertex-experts, data-mixing, 300m
```

Before live launch:

- Add `41` expected runs with kind `training`.
- Add launch manifest and epoch summary artifacts.
- Add a validation row for dry-run shape and region safety.

After launch:

- Add the Iris parent job.
- Link all training run rows to the parent job or to child job ids if available.
- Keep recovery/retry state in Fieldbook.

### 3. Score Expert Checkpoints on All Feature Surfaces

Create or extend:

```text
experiments/domain_phase_mix/launch_300m_mde_vertex_expert_features.py
```

Reuse the surface builders from:

```text
experiments/domain_phase_mix/launch_300m_checkpoint_features_canary.py
experiments/domain_phase_mix/launch_300m_checkpoint_features_full_swarm.py
```

Input:

- Training manifest with checkpoint roots and expected final steps.

Output:

- A feature surface index CSV with one row per `(run_name, surface, artifact)`:
  - `raw_text_loss_features/scored_documents.parquet`;
  - `teacher_forced_request_features/request_loglikelihoods.parquet`;
  - `mcq_request_features/request_loglikelihoods.parquet`;
  - any scalar eval outputs emitted by training.

Use the same scoring defaults as the full-swarm feature extraction unless explicitly changed:

- raw-text bundles: `paloma`, `uncheatable`, `raw_ppl_priority`, `agentic_coding`;
- teacher-forced request cache: current `launch_300m_generative_smooth_proxy_evals.DEFAULT_REQUEST_CACHE_URI`;
- MCQ request cache: current `launch_300m_mcq_smooth_proxy_evals.DEFAULT_REQUEST_CACHE_URI`;
- no bounded smoke limits for final scoring, except if required by memory/runtime constraints.

Before live feature scoring, resolve or mirror every eval/request cache to east5. The dry-run must assert that raw-text bundles, teacher-forced request caches, MCQ request caches, checkpoint roots, executor prefixes, and output paths do not reference central1/central2.

Validation:

- All `41` checkpoints have all three feature surfaces.
- Artifacts are under `gs://marin-us-east5`.
- Raw-text artifacts have non-empty `scored_documents.parquet`.
- Teacher-forced and MCQ artifacts have non-empty request-level parquet outputs.

### 4. Compact Dense Matrices for Local Modeling

Create:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/compact_mde_vertex_expert_features_300m.py
```

This should be a local/GCS-capable script with PEP 723 dependencies. It should read the feature index and write one compact directory:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mde_vertex_expert_dense_features_300m_<date>/
```

Dense outputs:

- `expert_manifest.csv`: `41` rows with domain, control flag, train tokens, epochs, checkpoint root, and feature artifact paths.
- `raw_text_document_nll_matrix.npy`: shape `[41, num_raw_text_documents]`.
- `raw_text_document_bpb_matrix.npy`: diagnostic-only shape `[41, num_raw_text_documents]`, derived from document NLL and bytes.
- `raw_text_document_metadata.parquet`: document key, dataset, bundle, bytes.
- `raw_text_token_nll_matrix.npy`: required bounded token sketch, shape `[41, num_selected_tokens]`.
- `raw_text_token_metadata.parquet`.
- `teacher_forced_request_matrix.npy`: shape `[41, num_teacher_forced_request_features]`.
- `teacher_forced_request_metadata.parquet`.
- `mcq_choice_matrix.npy`: shape `[41, num_mcq_choice_features]`.
- `mcq_choice_metadata.parquet`.
- `scalar_eval_matrix.parquet`: checkpoint-level scalar metrics if available.
- `summary.json` with shapes, dtypes, missingness, sizes, and source index hash.

For raw-text token sketches, choose a deterministic reference expert. Preferred reference: the cap-1 expert for the largest broad domain `dolmino_common_crawl_hq`; fallback: first expert in sorted manifest. Use stable hash sampling by `(dataset_name, request_id, token_index)` so all experts align to identical token coordinates.

Primary MDE composition must operate on logprob/NLL substrates. For raw text, compose token NLLs or document summed NLLs with logsumexp first, then convert to BPB using bytes after composition. Never apply logsumexp to already-normalized BPB ratios.

### 5. Build Queryable Candidate Features

Create:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/build_mde_vertex_query_features_300m.py
```

Input:

- `raw_metric_matrix_300m.csv`;
- 39-domain phase weights for each 300M candidate;
- vertex expert dense matrices.

Output:

- `candidate_mde_vertex_features.parquet`;
- `candidate_mde_vertex_feature_summary.json`.

For each candidate row and feature matrix `L` with expert dimension `[39, F]`, compute at least:

```text
phase0_mde_nll = -logsumexp(log(w_phase0_i) - nll_i_feature, over i)
phase1_mde_nll = -logsumexp(log(w_phase1_i) - nll_i_feature, over i)
phase_weighted_mde_nll = 0.8 * phase0_mde_nll + 0.2 * phase1_mde_nll
phase_delta_mde_nll = phase1_mde_nll - phase0_mde_nll
```

This logprob-level composition is the primary MDE estimator. Linear loss features are only a diagnostic baseline:

```text
phase0_linear_loss = w_phase0 @ L
phase1_linear_loss = w_phase1 @ L
phase_weighted_linear_loss = 0.8 * phase0_linear_loss + 0.2 * phase1_linear_loss
```

Do not use the two over-epoch controls for candidate-queryable features by default. Keep them only for expert-quality diagnostics. Optionally create a sensitivity feature set where cap-1 small-domain experts are replaced by full-6B controls.

Hard alignment checks:

- Exclude the two control rows from default queryable feature matrices.
- Reorder the remaining 39 expert rows exactly to `DOMAIN_NAMES`.
- Assert `expert_domain_order == list(DOMAIN_NAMES)` before any matrix multiplication or logprob composition.
- For zero weights, use masked log weights (`-inf`) rather than adding an epsilon that would leak support.

### 6. Fit All Metrics Locally

Create:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/fit_mde_vertex_all_metrics_300m.py
```

Targets:

- All numeric metric columns in `raw_metric_matrix_300m.csv` with enough complete observations.
- Preserve metric orientation metadata but fit on raw metric values first; orient only for summary plots.

Model families:

- `dsp_only`: canonical/effective-exposure DSP baseline per metric.
- `mde_vertex_linear_only`: ridge/PLS on queryable vertex features.
- `dsp_plus_mde_vertex_residual`: fit DSP in each fold, fit MDE residual on train fold only.
- `dsp_plus_mde_vertex_factor_residual`: factor/PCA/PLS-compressed MDE features, then residual.
- `control_swap`: same as residual model but replacing the two cap-1 experts with full-6B controls, only for sensitivity.

Evaluation:

- Same outer folds across all models for a target.
- No leakage: feature reducers and residual regressors fit inside each train fold.
- Report OOF Spearman, Pearson, R2, RMSE, top-k overlap where meaningful.
- Compare against the existing DSP-only result per metric.
- Include readiness metadata: target coverage, DSP OOF fit, MDE lift, and whether lift is statistically meaningful over folds.

Outputs:

- `mde_vertex_all_metric_metrics.csv`;
- `mde_vertex_all_metric_oof_predictions.parquet`;
- `mde_vertex_all_metric_lift_summary.html`;
- `mde_vertex_metric_family_heatmap.html`;
- `report.md`;
- `summary.json`.

### 7. Validation and Reviews

Before live training submission:

```bash
uv run python -m py_compile \
  experiments/domain_phase_mix/launch_300m_mde_vertex_experts.py \
  experiments/domain_phase_mix/launch_300m_mde_vertex_expert_features.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/compact_mde_vertex_expert_features_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/build_mde_vertex_query_features_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/fit_mde_vertex_all_metrics_300m.py
```

Focused tests should cover:

- expert run count and control run count;
- one-hot weights in both phases;
- cap-1 epoch calculation;
- full-6B control epoch calculation;
- no simulated epoching;
- LR schedule recomputation for short runs;
- region-local east5 paths;
- region-local eval/request caches;
- exact `DOMAIN_NAMES` row alignment before composition;
- logprob-level MDE composition checked against a small hand-computed fixture;
- raw-text composition tests must use NLL inputs and convert to BPB after composition, not compose BPB directly;
- dense matrix shape and missingness validation;
- fold-local residual fitting with no feature reducer leakage.

CC review gates:

- Plan review before implementation.
- Code review before live training submission.
- Code review before feature scoring submission.
- Analysis review before treating all-metric conclusions as final.

Use the Marin Claude session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, `env -u ANTHROPIC_API_KEY`, Opus 4.8, max effort.

### 8. Stop Criteria

Proceed to production-scale thinking only if at least one of these holds:

- DSP+MDE vertex features give consistent fold-level lift on multiple held-out targets, not just in-sample or observed-checkpoint leakage.
- The full-6B small-domain controls materially improve feature quality, indicating cap policy matters.
- MDE features reveal task clusters or residual modes that are not captured by DSP but are queryable from mixture weights.

Stop or de-prioritize if:

- MDE vertex features provide no meaningful lift over DSP across all metrics.
- Lift exists only when using held-out checkpoint-specific observed features.
- Features are dominated by expert maturity/undertraining rather than domain identity.

## Open Questions for CC Review

1. Is cap-1 with no simulated epoching the right estimator for finite single-domain conditionals at 300M/6B?
2. Should expert training keep the two-phase WSD optimizer schedule for comparability, or switch to a simpler single-cycle schedule now that the data distribution is constant?
3. Should the two small-domain controls be full-6B only, or should we also include a proportional/mixed-data control checkpoint for calibration?
4. For non-generative scalar surfaces that are not naturally per-token/per-choice logprobs, which transformation gives the least misleading queryable MDE feature?
5. Which feature surfaces should be mandatory for “all metrics”: scalar evals, raw text documents/tokens, teacher-forced requests, MCQ choices, or all of the above?

## CC Plan Review Notes

Initial CC plan review used:

```bash
env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-8 --effort max -p '<plan review prompt>'
```

The review found two blockers that this revision addresses:

- Per-expert budgets and LR schedules require one `MixtureExperiment` instance per expert, not one shared experiment.
- Primary MDE composition must be logprob-level `-logsumexp(log w_i - nll_i)`, with linear loss averaging demoted to a diagnostic baseline.

It also flagged serious concerns now encoded as assertions: expert-domain row alignment, explicit east5 cache validation, and documentation of the inherent comparability gap between simulated-epoch mixture checkpoints and no-simulated-epoch vertex experts.
