# Debugging log for domain-mix data provenance audit

Audit the domain/scale mixture data path before more modeling work. The goal is to determine whether the registry can be treated as the single source of truth, whether the completed datapoints satisfy the intended simulated-epoching semantics, and whether fitting code consumes those datapoints under the same assumptions.

## Initial status

The active modeling thread is moving from nominal model-size labels to non-embedding parameter counts. Under that convention the current scale labels map approximately as:

| Historical key | Non-embedding N | Actual train budget D | D/N |
|---|---:|---:|---:|
| `130m_2p6b` | 22.8M | 2.6B | 114.0 |
| `60m_1p2b` | 59.0M | 1.2B | 20.3 |
| `300m_6b` | 102.6M | 6.0B | 58.5 |
| `520m_10p4b` | 339.8M | 10.4B | 30.6 |
| `1_2b_24b` | 906.0M | 24.0B | 26.5 |

This immediately means plots against N alone are not Chinchilla-style scale curves after relabeling. The old `130M/2.6B` cell is really a very token-rich `20M/2.6B` point, so it can legitimately beat some `60M/1.2B` rows without implying a corrupted BPB.

## Registry as single source of truth

Hypothesis: `run_registry/logical_runs.csv` is a complete single source of truth for the modeling dataset.

Result: partially true.

The run registry is the best operational provenance table for current strong-tier runs:

- `logical_runs.csv`: 701 rows, updated 2026-04-23.
- `run_attempts.csv`: 265 attempts.
- `strong_tier_perplexity_ready.csv`: 107 rows.
- Strong-tier rows with explicit target-budget metadata: 121.
- Strong-tier budget/step consistency issues: 0.
- Duplicate strong-ready keys over `(scale, study_path, source_experiment, run_name, target_budget_multiplier)`: 0.
- Strong-ready objective metric: all 107 rows use `eval/uncheatable_eval/bpb`.

But the registry is not yet a full single source of truth for all rows used in modeling:

- 580 rows in `logical_runs.csv` have missing `experiment_budget`.
- Missing-budget rows include the older `60m_1p2b`, `300m_6b`, and historical optimum-validation families.
- The metric registry is stale relative to the run registry: `metric_registry/runs.csv` has 536 rows and was generated on 2026-04-17, while the run registry has 701 rows and was refreshed on 2026-04-23.
- `metric_registry/materialize_fit_dataset.py` reads `metric_registry/metrics_wide.csv`, not the fresh run registry directly, so any fit path using it must rebuild the metric registry first.

Conclusion: use `run_registry/logical_runs.csv` as the canonical provenance layer, but do not call it a complete modeling SSOT until historical rows are backfilled with canonical budget, scale, architecture, and epoch metadata. The fitting SSOT should be a materialized analysis dataset derived from the registry, not ad hoc packet CSVs.

## Simulated epoching semantics

Hypothesis: the strong-tier launch code obtains datapoints with the intended simulated-epoching invariant: each proxy sees the same effective epochs per domain as the target budget would produce; this is invariant across actual proxy N and D for a fixed target-budget multiplier.

Result: the core code path is semantically correct.

Launch path:

- `MixtureExperiment.create_training_step()` passes `target_budget` into `simulated_epoching_train()` when `target_budget` is set.
- `simulated_epoching_train()` writes `target_budget`, `experiment_budget`, and `simulated_epoch_subset_seed` onto the `LMMixtureDatasetConfig`.
- Levanter dataset loading then slices every domain to `true_length * experiment_budget / target_budget`.

The effective epoch count for domain `d` in phase `p` is:

```text
tokens_drawn / sliced_domain_tokens
= phase_fraction[p] * mixture_weight[p,d] * experiment_budget
  / (domain_tokens[d] * experiment_budget / target_budget)
= phase_fraction[p] * mixture_weight[p,d] * target_budget / domain_tokens[d]
```

So the actual proxy experiment budget cancels. For a fixed `target_budget_multiplier`, the intended effective epochs are invariant across model scale and actual train budget.

Strong-tier registry checks also match the intended budget metadata:

- `target_budget == floor(6_325_183_647_689 * target_budget_multiplier)` for all 121 strong-tier rows.
- `num_train_steps * batch_size * seq_len` matches `experiment_budget` within one batch for all 121 strong-tier rows.

Conclusion: the simulated-epoching mechanism itself is not the current suspected source of modeling weirdness.

## Target-step eval gating

Hypothesis: completed executor status is sufficient for inclusion in fitting.

Result: false. The correct inclusion gate is target-step perplexity availability.

Current strong-tier status by readiness:

| Logical status | `is_perplexity_ready=False` | `is_perplexity_ready=True` |
|---|---:|---:|
| `completed` | 4 | 95 |
| `failed` | 6 | 7 |
| `running` | 4 | 5 |

Important cases:

- Four completed rows are not fit-ready because they overshot and do not have target-step evals: the three 520M stratified rows and the 1.2B stratified holdout.
- Twelve failed/running 520M qsplit rows are fit-ready because they have target-step evals in the attempt artifacts.

Conclusion: fitting code must filter on `is_perplexity_ready` / target-step objective availability, not executor status. Completed-but-not-ready rows should be excluded or backfilled at target step.

## Fishy configurations

Hypothesis: the named scale rungs form a clean monotonic model family under the new non-embedding convention.

Result: false. The rows are valid N,D observations, but the labels are confounded if read as a clean Chinchilla scale ladder.

Findings:

- `60m_1p2b` is semantically close to a 60M non-embedding RegMix-style proxy.
- `130m_2p6b` is actually only 22.8M non-embedding params because it reuses the tied-embedding local `llama_150m` geometry.
- `300m_6b`, `520m_10p4b`, and `1_2b_24b` are approximately 100M, 340M, and 900M non-embedding params.
- The old budgets were tied to nominal labels, not the new non-embedding counts. Therefore `20M/2.6B` is much more token-trained than `60M/1.2B`.
- The apparent scale-trajectory anomaly where some 20M rows beat 60M is therefore not by itself a BPB bug. It is a plotting/modeling-axis bug if the plot implies D is controlled.

Conclusion: keep historical scale keys as stable IDs, but every modeling table and packet should carry explicit `non_embedding_params`, `experiment_budget`, `target_budget`, and display labels like `20M/2.6B`, `60M/1.2B`, `100M/6B`, `340M/10.4B`, `900M/24B`.

## Fitting consumption

Hypothesis: current fitting code consumes the scale axis and registry labels consistently.

Result: mixed.

Good:

- The corrected q/support evaluator can patch fresh strong-tier target-step labels from `run_registry/strong_tier_perplexity_ready.csv`.
- It also has an explicit `ACTUAL_NON_EMBEDDING_PARAMS` table and can overwrite the packet model sizes for a non-embedding scale-axis evaluation.

Bad:

- The v28 packet data file `data/nd_scale_runs.csv` still has `model_size` set to nominal values: 60M, 130M, 300M, 520M, 1.2B.
- Most packet code reads `packet.model_sizes` directly, so fresh ChatGPT Pro sessions that used `model_size` without overriding it were using the stale nominal N axis.
- `evaluate_qsupport_1p2b_holdout_20260423.py` still orders scales nominally.
- `evaluate_qsupport_corrected_data_20260423.py` fixes N for q/support, but that is an ad hoc patch layer, not a canonical data contract.

Conclusion: existing ChatGPT Pro packet work that used raw packet `model_size` is affected by the scale-axis convention bug. We can still salvage qualitative architecture ideas, but quantitative cross-scale claims should be rerun against a corrected packet.

## Current reproducibility blocker

Hypothesis: current launch code can cleanly instantiate the strong-tier graph for reproduction/resubmission.

Result: false.

Direct check:

```text
TypeError: LmEvalHarnessConfig.__init__() got an unexpected keyword argument 'eval_datasets_cache_path'
```

Cause:

- `experiments/defaults.py` passes `eval_datasets_cache_path` into `LmEvalHarnessConfig`.
- `lib/levanter/src/levanter/eval_harness.py::LmEvalHarnessConfig` does not currently define that field.

This appears to be a current-code reproducibility issue, not evidence that already-finished datapoints are invalid. Still, it should be fixed before any resubmissions or graph regeneration work.

## Results

Audit conclusions:

- Strong-tier simulated epoching and budget metadata are internally consistent.
- The run registry is valid as the operational provenance source for strong-tier rows.
- The registry is not yet a complete modeling SSOT because older rows lack canonical budget/scale/epoch metadata.
- Metric registry outputs are stale relative to the refreshed run registry.
- Target-step eval availability is the correct inclusion gate.
- The non-embedding scale switch invalidates nominal-only scale plots and packet `model_size` fields.
- Existing fresh-session packets should be regenerated with explicit non-embedding N and actual D columns before more cross-scale modeling.

## Future Work

- [ ] Add canonical scale metadata columns to the registry/materialized analysis dataset: `non_embedding_params`, `tied_total_params`, `input_embedding_params`, `output_head_params`, `tie_word_embeddings`, and human display label.
- [ ] Backfill budget and simulated-epoch metadata for the historical 60M and 300M rows that remain important for fitting.
- [ ] Rebuild `metric_registry` after every run-registry refresh, or add a freshness assertion that refuses to materialize fits from stale metrics.
- [ ] Create one canonical ND analysis dataset consumed by packets and local evaluators; stop patching packet labels and N axes in separate scripts.
- [ ] Fix the `eval_datasets_cache_path` / `LmEvalHarnessConfig` mismatch before launching or reproducing strong-tier jobs.
- [ ] Add automated audit checks for duplicate keys, budget realization, target-budget multiplier consistency, target-step eval availability, and stale metric-registry timestamps.

## Implementation: canonical analysis dataset

Changes:

- Added `experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/build_analysis_dataset.py`.
- Added `analysis_dataset/README.md` documenting the corrected data contract.
- Generated local outputs:
  - `analysis_dataset/nd_scale_runs.csv`
  - `analysis_dataset/nd_scale_packet.npz`
  - `analysis_dataset/summary.json`
- Created packet `chatgpt_pro_hybrid_data_mixing_packet_v29/` and archive `chatgpt_pro_hybrid_data_mixing_packet_v29.zip` from the analysis dataset outputs.

Important implementation detail:

- The builder drops all strong-tier rows that are not present in `strong_tier_perplexity_ready.csv`, even if the old packet had a historical BPB value. This removed the stale 520M stratified row that initially survived as `packet_historical_metric`.

Final validation:

- output rows: `629`
- label sources: `522` packet historical, `107` strong-tier target-step
- fixed-520M qsplit target-step rows: `27`
- 1.2B target-step rows: `2`
- duplicate canonical modeling keys: `0`
- rows missing primary labels: `0`
- `model_size == non_embedding_params`: true
- max phase-sum error: `8.88e-16`

Smoke test:

```bash
PYTHONPATH=experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29/code \
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from nd_scale_packet import load_packet, subset_indices

root = Path("experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v29")
packet = load_packet(root)
frame = packet.frame
assert len(frame) == len(packet.model_sizes)
assert np.array_equal(packet.model_sizes, frame["non_embedding_params"].to_numpy(int))
assert not np.any(packet.model_sizes == 130_000_000)
assert np.allclose(packet.weights.sum(axis=2), 1.0)
fixed = frame[(frame["scale"].eq("520m_10p4b")) & (frame["path"].eq("qsplit_representative12"))]
assert len(fixed) == 27
assert set(fixed["label_source"]) == {"run_registry_target_eval"}
onepoint2 = frame[frame["scale"].eq("1_2b_24b")]
assert len(onepoint2) == 2
assert set(onepoint2["label_source"]) == {"run_registry_target_eval"}
PY
```

Note:

- The packet warns that bundled q/support reference metrics were generated before the latest `2.0x` fixed-520M target-step row was added. Exact q/support baseline comparisons should be rerun on v29 before using them as final numbers.
