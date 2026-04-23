# Debugging log for epoch-feature budget semantics

Verify that packet/runtime epoch-sensitive features match the actual simulated-epoching
training semantics, then fix all affected helper paths.

## Initial status

We had been reasoning as if epoch-sensitive GRP features should scale with the actual
train-token axis `D` across `60M/130M/300M/520M`. The user flagged that there had been a
past bug or misunderstanding around simulated epoching, so the actual training code needed
to be checked directly.

## Hypothesis 1

The packet/runtime helpers may be reconstructing simulated epoch multipliers from realized
train tokens, even though the actual trainer keeps effective epochs fixed across scales by
subsampling datasets according to `experiment_budget / target_budget`.

## Changes to make

- Inspect the actual training path:
  - `experiments/defaults.py`
  - `lib/levanter/src/levanter/data/text/datasets.py`
  - `experiments/domain_phase_mix/scaling_study_recipes.py`
- Inspect the packet builder and packet-local helpers:
  - `.../chatgpt_pro_swarm_transfer_packet/build_packet.py`
  - `.../chatgpt_pro_swarm_transfer_packet/code/run_continuous_nd_grp_law.py`
  - mirrored `v22` packet-local copies
- If mismatched:
  - change custom epoch-multiplier helpers to use simulated `target_budget` semantics
    rather than realized proxy tokens
  - rerun the most affected benchmarks

## Future Work

- [ ] Refresh older candidate-geometry-dependent sweeps that still rely on stale outputs.
- [ ] Audit any external ChatGPT Pro packet copies beyond `v22` for the same helper bug.
- [ ] Revisit direct-backbone recommendations now that custom candidate geometry changed.

## Results

- Actual training semantics:
  - `simulated_epoching_train()` stores both `experiment_budget` and `target_budget`.
  - dataset slicing uses `simulated_data_ratio = experiment_budget / target_budget`.
  - effective epochs for a phase/domain are therefore proportional to
    `weight * phase_fraction * target_budget / domain_tokens`.
  - `experiment_budget` cancels, so at fixed `target_budget` the effective epochs are
    intentionally constant across scales.
- Packet builder semantics were already correct:
  - `build_packet.py` computes packet `simulated_epoch_multipliers` from `target_budget`.
  - the packet assumption text already says:
    “Simulated epoch multipliers depend on target_budget, not realized proxy tokens.”
- Runtime/helper bug:
  - `run_continuous_nd_grp_law.py` was reconstructing custom epoch multipliers from actual
    `D`, which is inconsistent with the actual training semantics.
  - downstream custom-feature builders inherited that bug.
- Fixes made:
  - added `target_budget_for_multiplier`, `resolve_target_budget`, and a corrected
    keyword-only `simulate_epoch_multipliers()` in both packet-local copies of
    `run_continuous_nd_grp_law.py`
  - updated custom-feature/candidate helpers to pass target-budget semantics explicitly
  - updated the parallel scaling-law benchmark to use per-row `target_budgets` rather than
    realized `D` when rebuilding epoch-sensitive features
- Immediate impact from rerunning `run_direct_backbone_merge_benchmark.py`:
  - direct holdout metrics moved only modestly
  - candidate geometry moved a lot
  - previously “healthy” direct backbones are now much sharper / more tech-heavy under the
    corrected semantics
  - example on the updated seed-7 geometry summary:
    - `direct_shared_score_tilt_poly4`
      - `mean_nearest_observed_mean_phase_tv` from about `0.419` to `0.643`
      - `min_phase1_effective_support` from about `19.7` to `1.00`
      - `max_phase1_max_weight` from about `0.10` to `0.9998`
- Interpretation:
  - the earlier candidate-geometry read for the direct backbones was materially wrong.
  - any recommendation that depended on those old custom-geometry diagnostics needs to be
    revisited.

## Hypothesis 2

The low-rank shared-curve / `rho` sweeps that used row-level custom GRP-lite features from
realized `D` are also stale under the corrected semantics.

## Changes to make

- rerun:
  - `run_parallel_scaling_law_benchmark.py`
- compare the updated summary against the old one, focusing on geometry-sensitive or
  multiplier-sensitive model families

## Results

- In progress during this session.

## Hypothesis 3

Even after the earlier helper fixes, the packet payload itself or the current packet-local
 helper stack might still encode scale-dependent simulated epochs incorrectly.

## Changes to make

- inspect the stored packet arrays directly in:
  - `.../chatgpt_pro_swarm_transfer_packet/data/nd_scale_packet.npz`
  - `.../chatgpt_pro_hybrid_data_mixing_packet_v23/data/nd_scale_packet.npz`
- check whether `simulated_epoch_multipliers` are invariant across scales at fixed
  `target_budget_multiplier`
- verify that current helper code consumes those packet semantics rather than rebuilding
  epochs from realized proxy tokens

## Results

- Both packet copies are internally consistent with the actual training semantics.
- Direct check on the stored packet arrays:
  - for `target_budget_multiplier in {0.5, 1.0, 2.0}`, both packet copies have exactly
    **one** unique `simulated_epoch_multipliers` pattern across all included scales
  - over the same rows, `raw_full_corpus_epoch_multipliers` has multiple distinct patterns
    by scale, which is expected because realized full-corpus epochs do vary with proxy size
- Concretely in the current swarm packet:
  - multiplier `0.5`: `unique_sim_patterns = 1`, `unique_raw_patterns = 3`
  - multiplier `1.0`: `unique_sim_patterns = 1`, `unique_raw_patterns = 5`
  - multiplier `2.0`: `unique_sim_patterns = 1`, `unique_raw_patterns = 3`
- So the packet payload matches the intended invariant:
  - same target-budget multiplier => same effective simulated epochs per domain across scales
  - only the raw full-corpus epoch counters vary with scale
- The runtime/helper layer is also aligned now:
  - `run_continuous_nd_grp_law.py` reconstructs domain token counts from stored packet
    `target_budgets` and `simulated_epoch_multipliers`
  - `simulate_epoch_multipliers()` now resolves from `target_budget` /
    `budget_multiplier`, not from realized `D`
  - downstream custom feature helpers such as `direct_backbone_candidates.py` call that
    corrected helper rather than rebuilding from proxy tokens
- Important nuance:
  - the direct-law feature builders still include explicit scale terms such as `uN`, `uD`,
    and `gain = uN + uD`
  - that means model features remain scale-dependent by design
  - but that is no longer an epoching-semantics bug; it is just part of the model class
