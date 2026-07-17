# Sweep Tool Contracts

The CLI accepts an operation and JSON request path. A point is an ordered list of actual axis values matching `axes`; grid positions are never persistent trial identities. This keeps points stable when a lower value is added to an axis. Times are numeric in one consistent unit. All operations are deterministic and perform no I/O beyond JSON input and output.

## check-convergence

Input:

- `axes`: ordered numeric `values` and hard `domain.min`/`domain.max`. `scale`, `preferred_max_gap`, and `special_values` are advisory search metadata consumed by the Orchestrator; convergence does not enforce them.
- `resource_levels`: ordered positive rung values.
- `objective.direction`: `minimize` or `maximize`.
- `trials`: completed logical trials with `rung`, `point`, and finite final `objective`.

For each completed point, the operation checks the next lower and higher grid value on one axis at a time. The point must be no worse than every neighbor. A missing side passes only when the current value equals that side's hard domain bound. Output contains qualifying dominant points and rung convergence. No tolerance or uncertainty enters this calculation.

## predict-objectives

Input extends `check-convergence` with `candidates`, each containing a unique `candidate_id`, `rung`, and uncompleted `point`.

The operation fits `GradientBoostingRegressor(random_state=0)` to all completed trials together. It resolves each point value to its current grid position, then uses normalized positions plus normalized resource-rung index as features. Output contains `predicted_objective` and `rank_within_rung` for every candidate. With no completed trials, predictions are unavailable.

## rank-targets

Input:

- `now`, `resource_levels`, corresponding positive `resource_ratios`, positive `wall_time_limit`, and `max_inflight_chips`.
- `current_rung` and `full_exploitation_rung` as zero-based resource indices.
- `recovery`: positive `startup_relocation_timeout`, `same_target_restart_timeout`, `same_region_relocation_timeout`, and `cross_region_restart_timeout`, with `same_target_restart_timeout < same_region_relocation_timeout < cross_region_restart_timeout`.
- `targets`: unique target ID, region, TPU slice, and explicit chips.
- `observations`: Dispatcher history flattened to `trial_id`, `regional_run_id`, `dispatch_id`, `rung`, `target`, `state`, `submitted_at`, `observed_at`, `wandb_run_id`, and `run_progress`. Progress may be `null` before it is observable; intervals with an unknown endpoint do not contribute throughput evidence.

For each observation interval in a dispatch:

```text
normalized_progress_throughput =
    resource_ratio[rung] * change_in_run_progress / elapsed_wall_time
```

Interval rates are duration-weighted by `weight(age) = (1 + cos(pi * age / wall_time_limit)) / 2`. The implementation integrates this weight over each interval, so irregular polling does not change its influence. Weight is one at age zero, positive before the wall-time limit, and zero at and beyond it.

The output includes mean target throughput for diagnosis. Placement ranks a target by its slowest active dispatch rate; when it has no active dispatch, it uses weighted normalized progress divided by weighted observed wall time. A fresh active dispatch without a measurable interval has marginal rate zero. This makes queued or stalled work immediately lower the target's rank. Targets without active work or positively weighted evidence sort last.

Exploration decays linearly to zero at `full_exploitation_rung`. Among `F` chip-feasible targets, the selection depth is:

```text
1 + floor(exploration_fraction * (F - 1))
```

The output's `recovery` records report the currently permitted actions. They distinguish an execution absent from W&B from a registered run whose progress stalled. `same_target_restart` requires the current Iris submission to remain running without progress for `same_target_restart_timeout`; it does not reset the regional no-progress clock. Cross-region restart requires a same-region recovery dispatch after the condition began, the total cross-region timeout, and a full applicable timeout on the current replacement dispatch. Each record includes the eligible target, including the current target for a restart. Impossible same-region relocations remain blocked. The Orchestrator retains restart and relocation authority.
