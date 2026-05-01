# StarCoder Subset Sampling: Research Logbook

## Scope
- Goal: find small StarCoder subset selections that preserve DS-RE-CEQ fit quality and distill a static sampler that can replace Dirichlet + rejection sampling in nextgen/design.
- Primary metric(s): `eval/paloma/dolma_100_programing_languages/bpb`, observed regret@1.
- Constraints: static batch design only; use the existing 2-phase and 3-phase StarCoder datasets; keep nextgen compatibility.

## Baseline
- Date: 2026-03-08
- Code refs: `experiments/domain_phase_mix/exploratory/starcoder_subset_selection.py`, `experiments/domain_phase_mix/static_batch_selection.py`
- Baseline numbers:
  - 2-phase StarCoder: 116 completed runs
  - 3-phase StarCoder: 160 completed runs

## Experiment Log
### 2026-03-08 00:00 - Initial scaffold
- Hypothesis: a D-optimal static batch selector over DS-RE-CEQ sensitivities can approximate the best observed subsets better than random or Dirichlet-replay baselines.
- Command: pending first run
- Config: pending first run
- Result: implemented analysis entrypoint, selection module, and nextgen design hook.
- Interpretation: ready for first end-to-end study run.
- Next action: run the analysis entrypoint, inspect oracle-vs-random gap, and review the recommended `k` per dataset.

### 2026-03-08 01:00 - Reduced pipeline smoke
- Hypothesis: the full oracle/deployable study pipeline should run end-to-end before we trust any larger batch result.
- Command: inline Python smoke invoking `_search_dataset` on the first 12 rows of `two_phase_starcoder` with `k=4`.
- Config:
  - `SEARCH_RANDOM_SUBSETS=2`
  - `SEARCH_KCENTER_SUBSETS=1`
  - `SEARCH_DOPT_SUBSETS=1`
  - `SEARCH_TOP_SUBSETS=2`
  - `SEARCH_TOP_FINAL=2`
  - `DEPLOYABLE_SEEDS=1`
  - `SEARCH_RESTARTS=2`, `SEARCH_MAXITER=80`
  - `FINAL_RESTARTS=2`, `FINAL_MAXITER=120`, `FINAL_SEEDS=2`
- Result:
  - `search_rows=6`
  - `final_rows=2`
  - `deployable_rows=2`
- Interpretation: the analysis entrypoint, retrospective selector, prospective selector, replay evaluation, and final rescoring all execute successfully on a reduced slice.
- Next action: run a real full-dataset pilot and record oracle-vs-random and deployable-vs-sampler gaps.

### 2026-03-08 17:39 - Two-phase full-dataset quick pilot (`k=10`)
- Hypothesis: even a reduced-budget oracle search should find a materially better 10-point subset than the median random 10-point subset on the 116-run two-phase StarCoder table.
- Command: inline Python calling `_search_dataset` on `two_phase_starcoder` with `k=10`, `workers=4`, and reduced search/final budgets; outputs written under `experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/pilot_two_phase_k10_quick/`.
- Config:
  - `SEARCH_RANDOM_SUBSETS=6`
  - `SEARCH_KCENTER_SUBSETS=2`
  - `SEARCH_DOPT_SUBSETS=2`
  - `SEARCH_TOP_SUBSETS=2`
  - `SEARCH_TOP_FINAL=3`
  - `HILL_CLIMB_MAX_ACCEPTED=4`
  - `DEPLOYABLE_SEEDS=3`
  - `SEARCH_RESTARTS=2`, `SEARCH_MAXITER=180`
  - `FINAL_RESTARTS=4`, `FINAL_MAXITER=300`, `FINAL_SEEDS=2`
- Result:
  - Best subset policy: `oracle_hill_climb`
  - Best subset indices: `[32, 50, 51, 56, 62, 70, 72, 84, 109, 114]`
  - Best median regret@1: `0.000805`
  - Random-search median regret@1: `0.003434`
  - Random-search best regret@1: `0.000000`
  - K-center median regret@1: `0.034554`
  - D-opt observed median regret@1: `0.004687`
  - Deployable D-opt replay median regret@1: `0.005129`
  - Sampler replay median regret@1: `0.002755`
- Interpretation:
  - The oracle subset is materially better than the median random subset in this pilot, about a 4.3x regret reduction.
  - The deployable static D-opt policy is not yet better than the current sampler replay in this reduced-budget two-phase pilot.
  - Pure k-center is clearly too coarse for this objective.
  - The oracle subset spans both monotone StarCoder-exposure anchors (`low-low`, `mid-mid`, `high-high`) and strong phase-transition anchors (`low-high`, `high-low`), while avoiding near-duplicate rows.
- Next action: inspect what makes the oracle subset good and tune the deployable prospective selector before scaling to the full `k` sweep and 3-phase study.

### 2026-03-08 18:05 - Three-phase long-running oracle search launched (`k=16`)
- Hypothesis: the three-phase oracle subset will show the same broad structure as the two-phase oracle signal: coverage over total StarCoder exposure, explicit phase-transition anchors, and stronger anti-duplication than naive D-optimal or random subsets.
- Command: `uv run python -u experiments/domain_phase_mix/exploratory/starcoder_subset_selection.py --datasets three_phase_starcoder --k-values 16 --workers 8 --search-random-subsets 96 --search-kcenter-subsets 12 --search-dopt-subsets 12 --search-top-subsets 12 --search-top-final 24 --hill-climb-max-accepted 20 --hill-climb-swap-candidates 12 --search-restarts 4 --search-maxiter 300 --final-restarts 8 --final-maxiter 500 --final-seeds 3 --save-top-search-n 100 --skip-deployable --output-dir experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/three_phase_oracle_k16_longrun_20260308`
- Config:
  - Objective: `eval/paloma/dolma_100_programing_languages/bpb`
  - Dataset: `three_phase_starcoder`
  - Subset size: `k=16`
  - Saved search bank: top `100` unique subsets from the combined search stage
  - Final rescoring set: top `24` unique subsets
  - Deployable replay: skipped for this run so compute goes into oracle search
  - Output dir: `experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/three_phase_oracle_k16_longrun_20260308`
  - PID: `80087`
- Result:
  - Status: running
  - Initial log line: `[study] dataset=three_phase_starcoder k=16 starting (random=96, kcenter=12, dopt=12)`
- Interpretation: this is the first long-running three-phase oracle job with enough retained top-search mass to inspect many near-optimal subsets rather than only a handful of finalists.
- Next action: when complete, compare the best three-phase subset and the saved top-100 bank against the two-phase oracle signal to derive a better static sampling procedure.

### 2026-03-08 18:40 - Three-phase oracle search relaunched with process parallelism
- Hypothesis: process-backed subset scoring and D-opt seed generation should materially increase CPU utilization on this machine and reduce wall-clock time for the same three-phase oracle budget.
- Command: `uv run python -u experiments/domain_phase_mix/exploratory/starcoder_subset_selection.py --datasets three_phase_starcoder --k-values 16 --workers 14 --parallel-backend process --search-random-subsets 96 --search-kcenter-subsets 12 --search-dopt-subsets 12 --search-top-subsets 12 --search-top-final 24 --hill-climb-max-accepted 20 --hill-climb-swap-candidates 12 --search-restarts 4 --search-maxiter 300 --final-restarts 8 --final-maxiter 500 --final-seeds 3 --save-top-search-n 100 --skip-deployable --output-dir experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/three_phase_oracle_k16_longrun_proc2_20260308`
- Config:
  - Objective: `eval/paloma/dolma_100_programing_languages/bpb`
  - Dataset: `three_phase_starcoder`
  - Subset size: `k=16`
  - Backend: `process`
  - Workers: `14`
  - Saved search bank: top `100` unique subsets
  - Final rescoring set: top `24` unique subsets
  - Deployable replay: skipped
  - Output dir: `experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/three_phase_oracle_k16_longrun_proc2_20260308`
  - PID: `97626`
- Result:
  - Status: running
  - Initial log line: `[study] dataset=three_phase_starcoder k=16 starting (random=96, kcenter=12, dopt=12)`
  - Observed worker fanout: 12 active `spawn_main` worker processes under the main Python PID during early search.
- Interpretation:
  - The earlier thread-backed three-phase run was underutilizing the host and has been superseded by this process-backed run.
  - This relaunch is the run to monitor going forward.
- Next action: wait for completion, then inspect `top_search_candidates.csv` and `oracle_finalists.csv` to generalize subset structure across 2-phase and 3-phase.

### 2026-03-08 20:07 - Three-phase full-dataset oracle result (`k=16`)
- Hypothesis: the three-phase oracle subset would reveal whether the two-phase signal generalizes cleanly or whether the good-subset geometry becomes more combinatorial/noisy.
- Command: completed process-backed run logged under `experiments/domain_phase_mix/exploratory/starcoder_subset_selection_outputs/three_phase_oracle_k16_longrun_proc2_20260308/`.
- Result:
  - Best final subset policy: `random_observed`
  - Best final regret@1: `0.000000`
  - Random-search median regret@1: `0.009085`
  - Best observed-pool D-opt regret@1: `0.002093`
  - Best hill-climb regret@1: `0.002093`
  - Recommended `k`: `16`
  - Saved search bank: top `100` unique subsets
- Interpretation:
  - The search still found a materially better-than-median subset, but unlike 2-phase the single best finalist came from the random pool rather than the model-guided search.
  - Aggregate subset descriptors separate less cleanly than in 2-phase; the stronger signal comes from which anchor rows recur in the top-100 bank.
  - The most recurrent rows are extreme schedule prototypes such as `(0,0,0)`, `(1,1,1)`, `(1,0,0)`, `(0,0,1)`, and middle-phase peaks/valleys, suggesting that reusable shape primitives matter more than a single smooth descriptor.
  - Pure observed-pool D-opt still overconcentrates on boundary points and is not sufficient on its own.
- Next action: distill a shape-aware static selector that explicitly covers flat-low, flat-high, single-phase concentration, monotone ramps, peaks, and valleys, then compare it against random and D-opt on both 2-phase and 3-phase pools.
### 2026-03-07 23:19 - generic selector benchmark implementation smoke
- Hypothesis: generic feature-space selectors can be benchmarked end-to-end with saved predicted optima and DS-RE-CEQ sample-efficiency plots.
- Command: experiments/domain_phase_mix/exploratory/starcoder_generic_selector_benchmark.py --output-dir experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308 --datasets two_phase_starcoder --subset-sizes 4,10 --workers 1 --random-bootstrap-seeds 1 --retrospective-dopt-seeds 1 --prospective-seeds 1 --prospective-pool-size 32 --dsre-fit-seeds 1 --opt-search-points 128 --opt-restarts 2 --opt-maxiter 40 --row-limit 12 --skip-two-phase-oracle-ensure
- Output dir: experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308
- Selection rows: 14
- Model score rows: 42
- Result: emitted benchmark artifacts and plots under experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308.
- Interpretation: harness is runnable; full results depend on the actual benchmark budget used.
- Next action: run the full benchmark configuration and review selector_summary.csv / predicted_optima exports.
### 2026-03-08 00:23 - generic selector benchmark implementation smoke
- Hypothesis: generic feature-space selectors can be benchmarked end-to-end with saved predicted optima and DS-RE-CEQ sample-efficiency plots.
- Command: /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_benchmark.py --output-dir /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308_rerun --datasets two_phase_starcoder --subset-sizes 4,10 --workers 2 --random-bootstrap-seeds 1 --retrospective-dopt-seeds 1 --prospective-seeds 1 --prospective-pool-size 32 --dsre-fit-seeds 1 --opt-search-points 128 --opt-restarts 2 --opt-maxiter 40 --row-limit 12 --skip-two-phase-oracle-ensure
- Output dir: /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308_rerun
- Selection rows: 14
- Model score rows: 42
- Result: emitted benchmark artifacts and plots under /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/smoke_two_phase_20260308_rerun.
- Interpretation: harness is runnable; full results depend on the actual benchmark budget used.
- Next action: run the full benchmark configuration and review selector_summary.csv / predicted_optima exports.
### 2026-03-08 09:09 - generic selector benchmark implementation smoke
- Hypothesis: generic feature-space selectors can be benchmarked end-to-end with saved predicted optima and DS-RE-CEQ sample-efficiency plots.
- Command: /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_benchmark.py --output-dir /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/full_20260308_rerun2 --datasets two_phase_starcoder,three_phase_starcoder --workers 14
- Output dir: /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/full_20260308_rerun2
- Selection rows: 1717
- Model score rows: 5151
- Result: emitted benchmark artifacts and plots under /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/full_20260308_rerun2.
- Interpretation: harness is runnable; full results depend on the actual benchmark budget used.
- Next action: run the full benchmark configuration and review selector_summary.csv / predicted_optima exports.

### 2026-03-11 15:32 - three-phase WSDS repeat implementation smoke
- Hypothesis: the dedicated WSDS repeat pipeline can reuse the three-phase StarCoder schedule pool, enforce a strict Feature-Bayes-Linear prefix chain, emit WSDS proxy/validation plans, fit DS-RE-CEQ on contiguous prefixes, and regenerate the predicted-vs-actual BPB plot without touching the cosine benchmark outputs.
- Command: `uv run python experiments/domain_phase_mix/exploratory/three_phase_starcoder_feature_bayes_linear_wsds_repeat.py --output-dir /tmp/wsds_repeat_smoke_workers2 --stage all --row-limit 16 --subset-sizes 4,8 --workers 2 --dry-run-launches --skip-wandb --collect-from-source-observed --dsre-fit-seeds 1 --dsre-restarts 2 --dsre-maxiter 40 --opt-search-points 128 --opt-restarts 2 --opt-maxiter 40`
- Output dir: `/tmp/wsds_repeat_smoke_workers2`
- Result:
  - Emitted `proxy_selection_plan.csv`, `proxy_launch_plan.json`, `proxy_results.csv`, `fit_status.json`, `curve_points.csv`, `predicted_optima.csv`, `predicted_optima.jsonl`, `three_phase_starcoder_validation_launch_plan.json`, and `plots/three_phase_starcoder_feature_bayes_linear_wsds_predicted_vs_actual_bpb.png`
  - Ready subset sizes: `[4, 8]`
  - Validation launch-plan size: `2`
- Interpretation:
  - The WSDS pipeline is wired end to end and supports strict-prefix reuse of a single `k=128` Feature-Bayes-Linear ranking.
  - Plot lookup now derives from launch-plan run names, so cosine and WSDS validations can coexist in W&B without suffix collisions.
  - The `--workers` flag now parallelizes local DS-RE-CEQ subset fitting during `fit_predict`.
- Next action: run the real three-phase WSDS proxy sweep and validation stages on cluster, then compare the resulting WSDS predicted-vs-actual BPB curve against the existing cosine curve.

### 2026-03-13 12:02 - pivot to two-phase boundary-aligned WSD repeat
- Hypothesis: the earlier three-phase WSDS learning-rate shape was a specification mistake rather than a scheduler bug; the intended comparison is a single continuous WSD schedule in the two-phase StarCoder setting, with the phase transition aligned to the WSD stable-to-decay boundary.
- Command:
  - implementation/tests: `uv run pytest tests/test_domain_phase_mix_two_phase_wsd_boundary_aligned_repeat.py tests/test_domain_phase_mix_starcoder_optima_validation_plot.py -q`
  - dry-run smoke: `uv run python experiments/domain_phase_mix/exploratory/two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_repeat.py --output-dir /tmp/t2s-wsd-boundary-pyTX4d --stage all --subset-sizes 4,8 --row-limit 16 --workers 2 --dry-run-launches --skip-wandb --collect-from-source-observed --dsre-fit-seeds 1 --dsre-restarts 2 --dsre-maxiter 40 --opt-search-points 128 --opt-restarts 2 --opt-maxiter 40`
- Result:
  - Added `create_two_phase_wsd_boundary_aligned_optimizer_config(...)` and `create_two_phase_wsd_boundary_aligned_experiment(...)` in `experiments/domain_phase_mix/two_phase_starcoder_experiment.py`
  - Resolved schedule values are `total_steps=3814`, `warmup_steps=38`, `boundary_step=1904`, `decay_steps=1910`
  - Direct LR samples confirm a single plateau through the phase boundary and then monotone decay: `step 38 -> 0.02`, `step 1903 -> 0.02`, `step 1904 -> 0.02`, `step 1968 -> 0.019944644`, `step 3813 -> 1.3709068e-08`
  - Added dedicated repeat pipeline `experiments/domain_phase_mix/exploratory/two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_repeat.py`
  - Smoke emitted `proxy_selection_plan.csv`, `proxy_launch_plan.json`, `proxy_results.csv`, `fit_status.json`, `curve_points.csv`, `predicted_optima.csv`, `predicted_optima.jsonl`, `two_phase_starcoder_validation_launch_plan.json`, and `plots/two_phase_starcoder_feature_bayes_linear_wsd_boundary_aligned_predicted_vs_actual_bpb.png`
- Interpretation:
  - The intended WSD semantics are now implemented as one continuous schedule with no cycle restarts.
  - The repeat pipeline reuses the existing `feature_bayes_linear` coordinate ranking as a strict prefix chain, so subset-size comparisons stay nested.
- Next action: submit the real two-phase WSD boundary-aligned proxy sweep and validation jobs, then compare the resulting predicted-vs-actual BPB curve against the existing cosine result.
