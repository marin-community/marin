# 300M DCLM Core Eval Completion Plan

## Goal

Collect enough DCLM Core evaluation coverage for the 300M data-mixture swarm to fit and optimize against DCLM Core after the eval pass completes.

## Current State

- The canonical 300M raw metric matrix has 242 signal rows and 1309 metric columns at `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv`.
- Marin's `CORE_TASKS` is explicitly only a subset of the DCLM paper's Core task set and omits generation-heavy tasks.
- Existing 300M launchers already provide the right mechanics for candidate discovery, exact HF checkpoint validation, east5 locality checks, missing-task state CSVs, retry-state generation, collection from executor prefixes, and result merging.

## DCLM Core Task Coverage

DCLM Core is 22 tasks in the paper. The relevant task list and action plan:

| Task | DCLM setting | Current 300M coverage | Action |
| --- | --- | --- | --- |
| AGI Eval LSAT-AR | 3-shot | absent | add to MCQ launcher |
| ARC Easy | 10-shot | partial, `lm_eval/arc_easy/*` over 94 rows | fill all 242 rows |
| ARC Challenge | 10-shot | absent for 10-shot, 5-shot exists | add to MCQ launcher |
| BigBench QA Wikidata | 10-shot | absent | generation canary, then full run if stable |
| BigBench Dyck Languages | 10-shot | absent | MCQ or generation canary depending harness output |
| BigBench Operators | 10-shot | absent | generation canary, then full run if stable |
| BigBench Repeat Copy Logic | 10-shot | absent | generation canary, then full run if stable |
| BigBench CS Algorithms | 10-shot | absent | MCQ or generation canary depending harness output |
| BigBench Language Identification | 10-shot | absent | MCQ or generation canary depending harness output |
| BoolQ | 10-shot | complete over 242 rows | reuse |
| CommonsenseQA | 10-shot | absent, 5-shot proxy exists | add to MCQ launcher |
| COPA | 0-shot | complete over 242 rows | reuse |
| CoQA | 0-shot | absent | generation canary, then full run if stable |
| HellaSwag | 0-shot and 10-shot | 0-shot partial over 94 rows, 10-shot absent | fill 0-shot and add 10-shot |
| Jeopardy | 10-shot | absent and not exposed by installed lm-eval | decide: custom dataset/task or explicit omission |
| LAMBADA | 0-shot | complete over 242 rows | reuse |
| OpenBookQA | 0-shot | complete over 242 rows | reuse |
| PIQA | 10-shot | partial, `lm_eval/piqa/*` over 94 rows | fill all 242 rows |
| SQuAD v2 | 10-shot | absent | generation canary, then full run if stable |
| WSC273 | 0-shot | complete over 242 rows | reuse |
| WinoGrande | 0-shot | absent, 5-shot proxy exists | add to MCQ launcher |

## Implementation Plan

1. Create `experiments/evals/dclm_core.py`.
   - Define a `DCLMCoreTask` dataclass with `name`, `alias`, `num_fewshot`, `mode`, `metric_key`, `random_baseline`, and `status`.
   - Add functions for MCQ tasks, generation tasks, and centered-accuracy aggregation.
   - Mark Jeopardy as `requires_custom_task` until a source and scoring implementation are selected.

2. Create `experiments/domain_phase_mix/launch_300m_dclm_core_evals.py`.
   - Use the state-row, launch-manifest, retry-state, and collect-prefix structure from `launch_300m_english_lite_evals.py`.
   - Use `_candidate_records()`, `_exact_hf_checkpoint()`, `_executor_prefix()`, `_checkpoint_region()`, and `_read_eval_metrics()` rather than rebuilding candidate discovery.
   - Emit local artifacts under `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_dclm_core_completion/`.
   - Support `--task-alias`, `--mode {mcq,generation,all}`, `--include-run-name`, `--dry-run`, `--collect-from-prefix`, and `--write-retry-state-from-prefix`.
   - Use east5 defaults: `v5p-8`, `us-east5`, `us-east5-a`, `MARIN_PREFIX=gs://marin-us-east5`, and exact parent placement on submission.

3. Split launches by evaluator mode.
   - MCQ/logprob tasks run through `evaluate_levanter_lm_evaluation_harness`.
   - Generation/extractive tasks run through `evaluate_lm_evaluation_harness` with the same conservative generation engine kwargs used by existing GSM8K/HumanEval launchers.
   - Run a canary first on `baseline_proportional` and one non-proportional qsplit row with `--max-eval-instances 20`.
   - Only after canary collection passes, run the full missing-task launch over all eligible 242 rows.

4. Add scoring and collection.
   - Flatten raw lm-eval outputs into `lm_eval/<alias>/<metric>` columns.
   - Compute DCLM-centered accuracy per task as `(accuracy - random_baseline) / (1 - random_baseline)`.
   - Compute `lm_eval/dclm_core/centered_accuracy_macro`, `lm_eval/dclm_core/task_count`, and `lm_eval/dclm_core/missing_task_count`.
   - Keep per-task raw metrics so later modeling can choose between hard accuracy, centered accuracy, choice logprob, and BPB where available.

5. Update metric registry ingestion.
   - Teach `build_raw_metric_matrix_300m.py` or its registry inputs to merge `300m_dclm_core_eval_results_merged.csv`.
   - Validate that no already-populated metrics are overwritten with nulls.
   - Emit a coverage summary for the 22 DCLM Core tasks.

6. Tests.
   - Add `tests/test_300m_dclm_core_evals.py`.
   - Test exact DCLM task inventory, random baselines, metric-key selection, centered accuracy, state-row skip/launch decisions, region mismatch deferral, and retry-state filtering.
   - Add a dry-run test asserting no central-region checkpoint/data paths are introduced and all launch defaults are east5.

7. Submission workflow.
   - Run `uv run python -m py_compile experiments/evals/dclm_core.py experiments/domain_phase_mix/launch_300m_dclm_core_evals.py`.
   - Run `uv run pytest tests/test_300m_dclm_core_evals.py -q`.
   - Run launcher dry-run and inspect state/manifest counts.
   - Validate Iris command with `uv run python -m experiments.domain_phase_mix.east5_launch_safety --command '<iris job run ...>'`.
   - Get CC review before canary submission.
   - Submit canary, collect, patch failures, get CC follow-up if needed.
   - Get CC review before full submission.
   - Submit full run with maximal safe concurrency and record jobs/runs/validations in Fieldbook.

## Open Decisions

- Jeopardy: decide whether to implement a custom task from the Kaggle Jeopardy dataset or report DCLM Core as 21/22 tasks until this is implemented.
- BigBench task mode: use installed lm-eval multiple-choice variants where they match DCLM semantics; otherwise use generation variants and document the metric.
- Exact DCLM reproduction: DCLM used LLM Foundry; Marin uses lm-eval/Levanter/vLLM paths. Treat the result as DCLM-Core-aligned unless we reproduce their exact task formatting.

## Result Notes

### 2026-06-10 - Merged DCLM Core 300M matrix

- Collected the successful gap-fill retry prefix `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_300m_dclm_core_gapfill_retry1_20260609`.
- Wrote the merged matrix to `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_dclm_core_completion/300m_dclm_core_eval_results_merged.csv`.
- Coverage: 242 rows total; 241 east5-launched rows have 22/22 DCLM Core `raw_score` and `centered_accuracy` metrics. `baseline_stratified` remains 0/22 because its checkpoint root is in `us-central1` and was intentionally not evaluated in the east5 job.
- Aggregate score column: `lm_eval/dclm_core/centered_accuracy_macro`, non-null for 241 rows.
- SHA-256: `12cc74faaa3a9d0ab337081cf9a159bceb2c1e6174a7795d8bb88864bb1e54ec`.
- Fieldbook artifacts: `art_01ktr7x8zw7ny4j2dwz4trhyqp` for the merged matrix and `art_01ktr7xq1sgg02gde2smx013cw` for the pre-gap-fill backup.
- Fieldbook coverage validation: `val_01ktr7wwmp5vsbtqv26jp97fdb`.

### 2026-06-10 - DSP fit and raw optimization on DCLM Core macro

- Added `experiments/domain_phase_mix/exploratory/two_phase_many/fit_dclm_core_dsp_300m.py`.
- Fit DSP to `-lm_eval/dclm_core/centered_accuracy_macro` over the 241 completed east5 300M rows, then reported predictions and raw optima back in higher-is-better centered-accuracy units.
- Artifact directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_core_dsp_300m_20260610/`.
- Variant comparison: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_core_dsp_300m_20260610_variant_comparison.csv` and `.html`.
- Main result: DCLM Core macro is weakly fit by the current DSP forms at 300M. The best OOF rank correlation came from the lower-capacity `dsp_phase_benefit_no_penalty_nnls` control with OOF Spearman `0.1803`, OOF \(R^2\) `-0.0745`, and OOF RMSE `0.008868`.
- The default effective-exposure DSP had train Spearman `0.5013` but OOF Spearman `0.0002` and OOF \(R^2\) `-0.4836`.
- The raw DSP optima are therefore mathematical surrogate optima, not recommended deployment candidates. The least-bad no-penalty raw optimum predicted `0.097909` DCLM centered accuracy, but it sits at TV `0.6514` from proportional and its nearest observed row is still `baseline_proportional`, indicating substantial extrapolation.
- Observed best row remains `run_00185` at `0.062849`; `baseline_proportional` is close behind at `0.062372`.

### 2026-06-10 - DCLM component-level DSP debug

- Added `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_dclm_component_dsp_fit_300m.py`.
- Artifact directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_core_component_dsp_diagnostic_20260610/`.
- Component-wise no-penalty DSP fits show the full macro failure is not a direct aggregation implementation bug. Averaging separately fit component OOF predictions gives macro OOF Spearman `0.1471` and OOF R2 `-0.4749`, close to the failed direct macro fit.
- Only `5/22` components have OOF Spearman at least `0.5`; only `7/22` have positive OOF R2. Median component OOF Spearman is `0.1568`.
- Four components are exactly zero across the completed 300M swarm: `bb_cs_algorithms_10shot`, `bb_operators_10shot`, `bb_qa_wikidata_10shot`, and `bb_repeat_copy_logic_10shot`.
- Macro variance is dominated by weakly modeled components: `boolq_10shot` contributes `67.7%` of macro variance but has OOF Spearman `0.0155` and OOF R2 `-0.2191`; `copa_0shot` contributes `11.6%` with OOF Spearman `0.1708` and OOF R2 `-0.1356`.
- The predictable-subset aggregates are modelable but do not identify a non-proportional observed improvement: `predictable_ge0p5_5` has OOF Spearman `0.5686` and OOF R2 `0.2703`, but `baseline_proportional` is the best observed row; `predictable_ge0p4_7` has OOF Spearman `0.5183` and OOF R2 `0.2020`, also with `baseline_proportional` best.
- CC review in session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490` agreed with this diagnosis: do not propose a DCLM-optimized mixture from the current 300M full-macro DSP fit; treat the full DCLM macro as a noisy guardrail unless larger scale or a pre-registered fittable subset changes the evidence.

### 2026-06-10 - DCLM-aligned smooth proxy DSP diagnostic

- Added `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_dclm_smooth_proxy_dsp_300m.py`.
- Artifact directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_smooth_proxy_dsp_300m_20260610/`.
- Built two diagnostic smooth proxies from the broader 300M matrix, not exact DCLM Core: `dclm_bpb_proxy` uses 11 overlapping BPB/perplexity components; `dclm_choice_norm_proxy` uses 10 overlapping `choice_logprob_norm` components. Several components are shot/alias approximations.
- Smooth proxies are more modelable than the hard DCLM macro: BPB proxy OOF Spearman `0.7546` and OOF R2 `0.5324`; choice-normalized proxy OOF Spearman `0.5813` and OOF R2 `0.2864`.
- Modelability does not transfer to hard DCLM Core. Coupling to hard DCLM macro is weak: BPB proxy hard-macro Spearman `0.1393`; choice-normalized proxy hard-macro Spearman `0.2727`.
- The all-component proxy-selected rows are directionally worse than proportional on hard DCLM: BPB proxy best observed `run_00207` is `-0.016772` hard macro below proportional; choice-normalized proxy best observed `run_00203` is `-0.024636` below proportional.
- Posthoc positive-coupling subsets are diagnostic only because the filters use the same 300M hard-DCLM data. The BPB positive subset selects `baseline_proportional` as best observed; the choice-normalized positive subset selects `run_00230`, still `-0.005991` below proportional on hard DCLM.
- Local coupling near proportional is not estimable from this swarm: `TV <= 0.2` contains only `baseline_proportional`; the first non-proportional baseline is at average phase TV `0.2037`, and most swarm rows are around TV `0.5+`.
- CC follow-up review in session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490` agreed with the no-submit decision and recommended stating the negative evidence more strongly: optimizing these proxies would select rows that hurt hard DCLM, so they should be treated as diagnostics rather than DCLM Core objectives.

### 2026-06-13 - Rank-percentile DSP dashboard comparison note

- Recorded Fieldbook note `note_01kv22a8bahk4jfx2tfzanmedg` on the distinction between Will's deployed Grug-MoE dashboard workflow and our current DCLM diagnostics.
- The deployed Grug-MoE dashboard at `https://oa.williamheld.com/` uses per-task rank-percentile DSP: each metric is oriented higher-is-better, observed candidate ranks are transformed to normal scores `z = Phi^-1((rank - 0.5) / n)`, and one DSP predictor is fit per metric.
- Its displayed "predicted Pareto-improver" is an offline stored mixture from `origin/will/datakit-moe-mix:experiments/grug/moe/swarm_dashboard2.py`, not an in-browser constrained optimizer. The offline objective starts from proportional phase logits and minimizes `sum_t relu(z_prop,t + 0.05 - z_hat_t(w))^2`.
- Current deployed JSON predicted all 41 Grug-MoE metrics improve over proportional by at least about `+0.142` z, with phase TV from proportional about `0.136` / `0.133`. This remains model-predicted only: no uncertainty/LCB objective, no explicit distance regularizer beyond initialization, and no validation evidence by itself.
- Our current DCLM factor pipeline is not this procedure: it z-scores smooth DCLM component utilities, builds aggregate/factor targets, and fits DSP to those targets. A cleaner next comparison is a DCLM analogue of the deployed workflow: per-component rank-INT DSP plus an aggregate objective over predicted component percentiles or z scores, after excluding stale pre-scoring-fix DCLM rows.
