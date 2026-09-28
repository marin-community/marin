# Offline RL v5: Research Logbook

## Scope
- Goal: find a new three-phase offline-control method that beats the current legacy `outcome_planner` under the pooled dense v4 evaluation setup.
- Primary metric(s): `fqe_value_mean`, `dr_value_mean`, fold win counts against `legacy_outcome_planner`.
- Constraints: keep evaluation on the existing three-phase target folds; do not justify rollout unless the method also beats `fixed_best_schedule`.

## Baseline
- Date: 2026-03-14
- Code refs:
  - `experiments/domain_phase_mix/offline_rl/train_offline_policy_bench.py`
  - `experiments/domain_phase_mix/offline_rl/train_three_phase_policy_bench_v4.py`
- Baseline numbers on the pooled dense v4 folds:
  - `legacy_outcome_planner`: `fqe_value_mean = 4.0709`, `dr_value_mean = 3.1376`
  - `fixed_best_schedule`: `fqe_value_mean = 4.1268`, `dr_value_mean = 4.3350`

## Experiment Log
### 2026-03-14 16:00 - pooled direct/hybrid follow-up
- Hypothesis: the old planner's advantage came from direct final-objective scoring, while v3/v4 Q-only variants improved support behavior but lost ranking power. A pooled direct planner on dense features should beat the legacy planner, and a hybrid `Q + direct` planner may also help.
- Command:
  - `uv run python /tmp/eval_v5_candidates.py`
- Config:
  - dataset: `/Users/calvinxu/Projects/Coursework/CS234/Project/RL_Bench/offline_rl_v4_three_phase_target_pooled_aux_20260312/dataset_v4`
  - candidates:
    - `dense_direct_v5`: pooled dense direct planner with `reward_bonus_weight = 0.08`, `support_lambda = 0.02`
    - `hybrid_q_direct_v5`: pooled dynamic-Q plus direct-utility hybrid with `direct_alpha = 2.0`, `support_lambda = 0.05`
- Result:
  - `dense_direct_v5`
    - `fqe_value_mean = 4.0860`
    - `dr_value_mean = 4.1887`
    - `beat_legacy_fqe_folds = 5/5`
    - `beat_legacy_fqe_and_dr_folds = 4/5`
  - `hybrid_q_direct_v5`
    - `fqe_value_mean = 4.0870`
    - `dr_value_mean = 4.0522`
    - `beat_legacy_fqe_folds = 5/5`
    - `beat_legacy_fqe_and_dr_folds = 3/5`
- Interpretation:
  - `dense_direct_v5` is the best new offline method so far.
  - The direct objective model was the right inductive bias to recover; pure Q-style models were too noisy on this action-sparse dataset.
  - Neither v5 method beats `fixed_best_schedule`, so rollout remains unjustified.
- Next action:
  - keep `dense_direct_v5` as the new offline baseline and focus future work on closing the gap to `fixed_best_schedule`.

### 2026-03-14 17:10 - v6/v7/v8 follow-ups against fixed schedule
- Hypothesis:
  - v6: the remaining gap comes from overusing StarCoder in phase 0; cap phase 0 and train only on three-phase targets.
  - v7: preserve the best historical prefix and only adapt later phases.
  - v8: use a conservative phase-2 adapter on top of the fixed best schedule.
- Command:
  - `uv run python /tmp/eval_v6_candidates.py`
  - `uv run python /tmp/eval_v7_fixed_prefix.py`
  - `uv run python /tmp/eval_v8_conservative.py`
  - `uv run python /tmp/eval_v8_conservative_hi.py`
- Result:
  - best v6 candidate: `three_only_capped_hybrid_v6`
    - `fqe_value_mean = 4.1002`
    - `dr_value_mean = 4.1781`
  - best v7 candidate: `fixed_phase0_plus_hybrid_v7`
    - `fqe_value_mean = 4.1026`
    - `dr_value_mean = 4.1869`
  - best v8 conservative candidate: `conservative_phase2_direct_m0.10`
    - `fqe_value_mean = 4.1201`
    - `dr_value_mean = 4.4583`
- Interpretation:
  - phase-0 control was part of the issue, but fixing it alone was not enough.
  - the conservative phase-2 adapter gets very close to the fixed schedule and even exceeds it on mean DR, but still misses on mean FQE.
  - none of these variants beats `fixed_best_schedule` on both FQE and DR, so none is rollout-ready.
- Next action:
  - keep `dense_direct_v5` as the best replacement for the legacy planner, and treat conservative/fixed-prefix adapters as promising but still incomplete branches.
