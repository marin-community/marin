# Offline RL Notes

This directory contains the offline-control and rollout code for the domain/phase mixture experiments.

## Current State

As of March 7, 2026, the main online policy asset to resume from is:

- [selected_policy_artifact.json](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/policy_assets/three_phase_starcoder_outcome_planner_v2/selected_policy_artifact.json)
- companion defaults: [decision_state_defaults.json](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/offline_rl/policy_assets/three_phase_starcoder_outcome_planner_v2/decision_state_defaults.json)

This asset is the pooled `sklearn_outcome_planner_v2` selected from the offline v2 benchmark. The planner bundle itself is unchanged from the pooled benchmark selection; the important online fixes were:

1. phase-boundary history is now collected with `scan_history`,
2. `build_wide_history` no longer drops rollout rows when `local_run_id` is missing,
3. phase 0 uses decision-specific defaults instead of all-decision artifact means.

## Key Entry Points

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

1. The corrected closed-loop result is strong, but it is still a single successful run.
2. The fixed-schedule replay is needed to separate schedule quality from rollout-path effects more cleanly.
3. The pooled planner is empirically strong online, but it is still a surrogate planner rather than a learned policy with a clean finite-horizon value interpretation.
