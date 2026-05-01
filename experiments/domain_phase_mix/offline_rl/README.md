# Offline RL Notes

This directory contains the offline-control and rollout code for the domain/phase mixture experiments.

## Current State

As of March 12, 2026, the main historical online policy asset to resume from is:

- [selected_policy_artifact.json](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/policy_assets/three_phase_starcoder_outcome_planner_v2/selected_policy_artifact.json)
- companion defaults: [decision_state_defaults.json](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/policy_assets/three_phase_starcoder_outcome_planner_v2/decision_state_defaults.json)

This asset is the pooled `sklearn_outcome_planner_v2` selected from the offline v2 benchmark. The planner bundle itself is unchanged from the pooled benchmark selection; the important online fixes were:

1. phase-boundary history is now collected with `scan_history`,
2. `build_wide_history` no longer drops rollout rows when `local_run_id` is missing,
3. phase 0 uses decision-specific defaults instead of all-decision artifact means.
4. chained rollout stages now preserve the native multi-phase data schedule and full simulated-epoching budget instead of
   rebuilding each phase as a one-phase cumulative-budget experiment.

The current offline frontier is no longer that pooled v2 benchmark. A new three-phase-only dense v3 benchmark now exists under:

- `/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v3_three_phase_dense_20260312`

That v3 run completed on March 12, 2026 with:

- dataset: `160` finished three-phase runs, `480` decision rows, `38,240` interior windows
- candidates: `sklearn_dynamic_q_planner_v3`, `torch_gru_q_v3`, `torch_transformer_q_v3`
- result: no passing methods
- best v3 candidate: `dynamic_q_planner_v3`
  - `fqe_value_mean = 4.0462`
  - `dr_value_mean = 4.0466`
  - `unsupported_rate_mean = 0.0021`
  - `boundary_rate_mean = 0.2208`
- online action: no rollout was launched because the offline gate failed

The supporting writeup is:

- [offline_rl_v3_benchmark_report.md](/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v3_benchmark_report.md)

A pooled-auxiliary v4 benchmark now also exists under:

- `/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v4_three_phase_target_pooled_aux_20260312`

That v4 run reused the same three-phase evaluation target but added the `143` finished two-phase runs as auxiliary training data. It completed grouped CV on March 12, 2026 with:

- dataset: `303` total runs, `766` decision rows, `61,120` pretraining windows
- candidates: `dynamic_q_planner_v4_pooled`, `gru_q_v4_pooled`, `transformer_q_v4_pooled`
- result: no passing methods
- best v4 candidate: `dynamic_q_planner_v4_pooled`
  - `fqe_value_mean = 4.0492`
  - `dr_value_mean = 4.2855`
  - `unsupported_rate_mean = 0.0760`
  - `boundary_rate_mean = 0.2324`
  - `canonical_phase0_pass_rate = 1.0`
- online action: no rollout was launched because the offline gate failed

The supporting writeup is:

- [offline_rl_v4_pooled_aux_benchmark_report.md](/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v4_pooled_aux_benchmark_report.md)

A pooled-direct v5 follow-up now exists under:

- `/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v5_three_phase_target_pooled_direct_20260314/bench_v5`

That v5 run reused the v4 pooled dense dataset and re-evaluated the current legacy `outcome_planner` against two new pooled-direct methods:

- `dense_direct_v5`
- `hybrid_q_direct_v5`

Result:

- `dense_direct_v5` is the first new offline method in this thread that clearly beats the current legacy `outcome_planner` on the same folds.
  - `fqe_value_mean = 4.0860` vs legacy `4.0709`
  - `dr_value_mean = 4.1887` vs legacy `3.1376`
  - `beat_legacy_fqe_folds = 5/5`
  - `beat_legacy_fqe_and_dr_folds = 4/5`
- `hybrid_q_direct_v5` also beats legacy on FQE and improves DR materially, but is weaker than `dense_direct_v5` overall.
  - `fqe_value_mean = 4.0870`
  - `dr_value_mean = 4.0522`
- neither v5 candidate beats `fixed_best_schedule`, so no new rollout is justified from v5 yet

The supporting writeup is:

- [offline_rl_v5_pooled_direct_benchmark_report.md](/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v5_pooled_direct_benchmark_report.md)

The fourth fix matters for result interpretation. Before March 8, 2026, chained rollout jobs were not directly comparable
to native static schedule runs because the rollout evaluator changed the dataset slicing semantics in early phases.
Regression coverage for this now lives in [test_domain_phase_mix_offline_rl.py](/Users/calvinxu/Projects/Work/Marin/marin/tests/test_domain_phase_mix_offline_rl.py).

## Key Entry Points

- [collect_three_phase_starcoder_dense_dataset.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/collect_three_phase_starcoder_dense_dataset.py)
  - cadence-aware W&B collector for three-phase-only v3 data
  - keeps train/LR, norm, and eval histories in separate queries to avoid mixed-key history collapse
- [build_three_phase_dense_policy_dataset.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/build_three_phase_dense_policy_dataset.py)
  - builds the v3 decision table plus dense pre-boundary sequence windows
  - action grid is now `[0.0, 1.0]` with 21 bins, matching historical three-phase runs
  - now uses per-run cached dense arrays so the full real dataset build finishes in minutes instead of thrashing pandas filters
- [collect_pooled_starcoder_dense_dataset.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/collect_pooled_starcoder_dense_dataset.py)
  - collects pooled dense two-phase plus three-phase telemetry for v4
  - can reuse an existing three-phase dense collector output and only fetch the missing two-phase dense histories
- [build_pooled_dense_policy_dataset_v4.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/build_pooled_dense_policy_dataset_v4.py)
  - builds the pooled v4 decision table while preserving `num_phases_total` and `remaining_decisions`
  - evaluation remains three-phase-only; the pooled two-phase rows are auxiliary training data
- [train_three_phase_policy_bench_v4.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/train_three_phase_policy_bench_v4.py)
  - trains three-phase-target policies with pooled two-phase auxiliary data
  - current best v4 result is `dynamic_q_planner_v4_pooled`, but it still fails the rollout gate
- [train_three_phase_policy_bench_v5.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/train_three_phase_policy_bench_v5.py)
  - compares the legacy `outcome_planner` against:
    - `dense_direct_v5`
    - `hybrid_q_direct_v5`
  - reuses the pooled v4 dense dataset and evaluates only on the three-phase target folds
  - current best offline replacement for the legacy planner is `dense_direct_v5`
- [train_three_phase_policy_bench_v3.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/train_three_phase_policy_bench_v3.py)
  - trains and compares:
    - `sklearn_dynamic_q_planner_v3`
    - `torch_gru_q_v3`
    - `torch_transformer_q_v3`
  - keeps `fixed_best_schedule` and `discrete_bc` as offline baselines
  - disables the transformer nested-tensor fast path so the sequence benchmark runs on Apple `mps`
- [train_offline_policy_bench.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/train_offline_policy_bench.py)
  - trains and evaluates the offline policy families
  - now includes the finite-horizon backward-induction planner artifact kind `sklearn_dynamic_q_planner_v2`
- [evaluate_policy_three_phase_starcoder.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/evaluate_policy_three_phase_starcoder.py)
  - runs chained 3-phase online validation
  - supports both older pooled `reward_models` planner bundles and newer `q_models` planner bundles
- [evaluate_policy_two_phase_starcoder.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/evaluate_policy_two_phase_starcoder.py)
  - inspects or evaluates the same policy family on the 2-phase StarCoder setup
  - `--inspect-only` computes phase-0 and hypothetical phase-1 actions from family-specific historical defaults without submitting a run
- [collect_three_phase_starcoder_dataset.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/collect_three_phase_starcoder_dataset.py)
  - shared W&B history collection and wide-history normalization for rollout feature extraction

Related one-off helper:

- [three_phase_starcoder_static_schedule.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/three_phase_starcoder_static_schedule.py)
  - launches a single fixed 3-phase schedule for sanity checks against policy rollouts

## Most Important Online Results

Earlier partially open-loop rollout:

- W&B runs:
  - `phase_0_7400e5c4_r00`
  - `phase_1_7400e5c4_r00`
  - `phase_2_7400e5c4_r00`
- executed schedule: `(0.815, 0.320, 0.365)`
- final programming BPB: `0.8461607695`

Corrected closed-loop rerun of the same pooled planner family:

- W&B runs:
  - `phase_0_1816d44e_r00`
  - `phase_1_1816d44e_r00`
  - `phase_2_1816d44e_r00`
- executed schedule: `(0.4550000131, 0.3199999928, 0.2750000060)`
- final programming BPB: `0.8346716762`

Interpretation:

- the online control-path fix materially changed the initial policy action
- phase 0 dropped from `0.815` to `0.455`
- the corrected closed-loop rerun improved over the earlier rollout by `0.0114890933` BPB
- it also improved over the best previously observed historical 3-phase run (`0.8612235188`) by `0.0265518426` BPB

## Practical Resume Point

If resuming this thread in a later session, the most useful sequence is:

1. inspect the repo-local asset under `policy_assets/three_phase_starcoder_outcome_planner_v2/`,
2. inspect the chained rollout code in [evaluate_policy_three_phase_starcoder.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/evaluate_policy_three_phase_starcoder.py),
3. compare against the external writeup in `/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/`,
4. check whether the fixed-schedule replay submitted via [three_phase_starcoder_static_schedule.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/three_phase_starcoder_static_schedule.py) finished and whether it reproduces the earlier `r00` schedule outcome.

## Known Open Questions

1. The March 7 rollout results should be treated as provisional because they predate the chained-rollout budget fix.
2. Corrected reruns on March 8 are the right apples-to-apples comparison point for both two-phase and three-phase.
3. `train_lm.py` now waits for the async checkpointer before exit, and chained rollout should resume from the exact boundary checkpoint (`step == cumulative_steps - 1`), not the latest durable checkpoint.
4. Native static replays of the corrected executed schedules are still needed to separate schedule quality from boundary-conditioned adaptivity.
5. v3 is now available on the three-phase-only dense dataset and currently rejects all candidates before rollout.
6. The next serious offline-control iteration should now start from `dense_direct_v5`, not the older Q-only planners.
7. `dense_direct_v5` beats the legacy `outcome_planner` on the same v4 folds, but still does not beat the strongest fixed historical schedule offline.
