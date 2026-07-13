---
name: design-adaptive-sweep
description: Design and save an operator-approved policy for tuning a small set of numeric hyperparameters at each rung of an ordered resource ladder, such as increasing epochs, tokens, parameters, or FLOPs. Use before run-adaptive-sweep. A rung converges when one evaluated grid point is no worse than every immediate grid neighbor, all of which must also be evaluated.
---

# Design Adaptive Sweep

Inspect the experiment, teach the operator how the proposed sweep will behave, draft the smallest useful policy, and save it after approval. Do not launch trials.

Use this method for a comparable objective, a few numeric hyperparameters, and one ordered resource axis whose lower rungs can inform higher rungs. Stop when the search is high-dimensional or structural, or when resource rungs are unrelated.

The sweep evaluates a grid at every rung. Lower-rung results forecast promising upper-rung configurations, so execution may interleave rungs instead of completing them in sequence. Every rung must converge. The grid may expand within hard bounds, normally preserving its preferred resolution, and execution keeps the approved chip budget occupied when useful unique trials remain. One `(configuration, rung)` is one logical trial; this workflow does not repeat configurations to estimate noise.

## Divide Responsibilities

The agent:

- Inspects the training entry point and existing experiment context.
- Proposes missing grid, ladder, execution, and recovery values from that context.
- Calculates the initial search size and relative work envelope.
- Explains material consequences and asks the operator to approve inferred values.
- Saves the approved policy and does not launch work.

The operator must identify the training entry point and confirm the objective. They state any hard search or compute constraints, target restrictions, and personal directives they already know. Infer optional details from context, then ask the operator to review the assumptions and approve any training-script edit. Do not ask them to choose runtime scoring or scheduling mechanics owned by `run-adaptive-sweep`.

After the script is available, prefer one complete draft and consolidated review over a sequence of questions about optional fields. Ask earlier only when a missing answer blocks inspection or materially changes the proposal.

## Require A Training Entry Point

Require a specific training script and a command that runs exactly one declared `(configuration, resource rung, region)`. The script must accept the TPU slice and region explicitly, log the exact objective, and expose monotonic W&B `run_progress` from `0` to `1` over the declared rung. TPU selection must drive adaptive batch sizing without changing regional run identity; region must determine W&B identity and checkpoint locality. The same regional run must resume on another compatible TPU slice, while a region change starts a fresh run without reading the prior region's checkpoint.

Inspect this interface. Stop and ask for the script when none is provided. Explain any required change and obtain operator approval before editing the script.

## Policy Format

Write the policy as Markdown with four sections:

1. `Required Inputs` is the stable YAML contract consumed by `run-adaptive-sweep`, grouped as `experiment`, `search`, and `execution`.
2. `Execution Preferences` is structured for agent readability but has no fixed schema.
3. `Operator Directives` contains prose rules and personal preferences.
4. `Reviewed Assumptions` records every inferred value approved by the operator.

## Example Policy

### Required Inputs

```yaml
experiment:
  training_script: experiments/protein/exp75_sweep.py
  # TPU selects adaptive batch sizing; REGION sets the regional W&B and checkpoint identity.
  single_job_command: >
    EPOCHS={epochs} LR={learning_rate} WD={weight_decay} TPU={tpu_slice} REGION={region}
    uv run python -m experiments.protein.exp75_sweep
  objective:
    # Full W&B metric key used for comparisons across every trial.
    wandb_metric: eval/contacts-v1-val/loss
    # Select the value recorded at the final training step.
    observation: final_step
    direction: minimize

search:
  grid:
    learning_rate:
      values: [1.0e-5, 1.0e-4, 1.0e-3]
      # Log scale makes spacing multiplicative rather than additive.
      scale: log10
      # Preferred largest transformed grid gap. One decade is 10x on log10 scale.
      preferred_max_gap: 1.0
      # Hard search bounds; current values may expand toward them.
      domain: {min: 1.0e-5, max: 1.0e-2}
    weight_decay:
      values: [0, 0.01, 0.03, 0.1]
      # Positive values use approximately 3x multiplicative spacing.
      scale: log10
      # Zero is legal but excluded from logarithmic spacing checks.
      special_values: [0]
      # Preferred largest transformed grid gap. The positive points stay below this 3.55x limit.
      preferred_max_gap: 0.55
      domain: {min: 0, max: 1}
  resource_ladder:
    name: epochs
    # Ordered rungs that must each converge.
    levels: [1, 4, 16]
    # Expected work relative to the first rung. These values normalize throughput and estimate rung cost.
    resource_ratios: [1, 4, 16]

execution:
  # Local orchestration record; choose a new path for each sweep.
  state_db: scratch/exp75-adaptive-sweep.sqlite
  # Elapsed sweep limit, including queueing and retries.
  wall_time: 3 weeks
  # Maximum requested TPU chips across submitted, running, or retrying dispatches.
  max_inflight_chips: 64
  # Dispatcher cadence for polling Iris, logs, and W&B. Throughput is recomputed on each observation.
  observation_interval: 15m
  # Resource level at which placement uses only the best currently observed target.
  # Earlier rungs admit progressively more lower-ranked or untried targets.
  full_exploitation_level: 16
  stagnation:
    # Initial execution may move within its region when no W&B run appears by this time.
    initial_wandb_timeout: 1h
    # A W&B-registered run may move within its region after this long without progress.
    progress_stall_timeout: 4h
    # A stalled run may restart elsewhere only after a same-region move also fails to progress.
    cross_region_restart_timeout: 48h
```

### Execution Preferences

```yaml
targets:
  allow:
    - region: europe-west4
      tpu_slices: ["v5litepod-{4,8,16,32,64,128,256}", "v6e-{4,8,16,32,64,128,256}"]
    - region: us-central1
      tpu_slices: ["v5p-{8,16,32,64,128,256,512}"]
    - region: us-east5
      tpu_slices: ["v5p-{8,16,32,64,128,256,512}", "v6e-{4,8,16,32,64,128,256}"]
  block: {regions: [], tpu_slices: []}
```

### Operator Directives

- Append `--user "$USERNAME"` to every Iris job submission and resubmission.
- Show me the assembled Iris job-run command and ask for review before the first submission. Seeing the command helps me catch launch details I often overlook until they are concrete.

### Reviewed Assumptions

- Trial duration is approximately proportional to epochs, giving ratios `1:4:16`.
- Training is deterministic enough that duplicate configurations are unnecessary.
- The registration and progress timeouts are suitable for this training script's startup behavior.

## Review What Matters

Review the following consequences with the operator, not just the field values:

- **Objective:** `wandb_metric`, observation point, and direction define every comparison. The metric must be comparable across all trials and stable enough to tune without replicated configurations.
- **Grid:** Values, `scale`, and `preferred_max_gap` describe the intended resolution of the initial grid and later extensions. Use them as the default when proposing adjacent values so local dominance remains meaningful. The Orchestrator may choose different spacing when evidence justifies it, but must record the value and reason. Hard domains permit convergence at a legal endpoint and limit expansion; an edge short of its hard bound cannot pass the convergence tool. `special_values` are legal exceptions to the regular transformed sequence.
- **Resource ladder:** Every level must converge, but evidence transfers across levels and may reduce expensive upper-rung sampling. More rungs add information and mandatory work.
- **Execution envelope:** `wall_time` is the hard elapsed limit. `max_inflight_chips` is the requested-capacity ceiling that the Orchestrator tries to fill until all rungs converge; propose `64` when unspecified. Target preferences determine which regions and slices may consume that budget. Checkpoints remain in their original region, so a cross-region placement starts that trial again.
- **Placement exploration:** `full_exploitation_level` is the one operator-facing exploration control for region and slice selection. Earlier rungs admit progressively more lower-ranked or untried targets; at and beyond this level only the best currently observed feasible target is admitted. Use the highest rung when unspecified.
- **Recovery:** Observation cadence and stagnation timeouts should reflect normal startup and progress cadence. When context supplies no better values, propose `15m` polling, `1h` for initial W&B registration, `4h` for a registered-run stall, and `48h` for cross-region restart after a failed same-region recovery. Present them for approval and request alternatives when the training system makes these values questionable.

`resource_ratios` deserve explicit review because they have two effects. They normalize target throughput across rungs:

```text
resource_ratio[rung] * change_in_run_progress / t
```

For example, `0.25` progress on a `4x` rung in one hour and complete progress on a `1x` rung in one hour both produce normalized throughput `1`. They also provide the Orchestrator's initial relative-cost model when deciding whether another cheap-rung result is worth waiting for before launching expensive-rung work. They do not alter objective values or convergence. Confirm that they describe expected work rather than copying rung labels when work does not scale proportionally.

## Present A Review Brief

Before requesting approval, summarize:

- What constitutes one trial, how rungs may interleave, how rung convergence is tested, and when the grid may expand.
- Initial configurations per rung, maximum trials on the initial grid, and exhaustive relative work `configurations * sum(resource_ratios)`. Include the full hard-domain grid envelope when its expansion sequence is unambiguous.
- The hard wall-time limit, eligible regions and slices, maximum inflight chips, and maximum requested chip-hours implied by those limits.
- A duration estimate from prior runs when credible. Otherwise state that duration is initially unknown and will be revised after early progress establishes throughput.
- Values inferred by the agent, missing blockers, script changes requiring permission, and assumptions that could materially change cost or behavior.

Do not promise that adaptive execution will beat exhaustive evaluation. Explain that forecasts are intended to reduce upper-rung trials and wall time, while rung convergence and grid expansion determine the work actually required.

## Review And Save

Return the versioned policy path only after the operator understands the review brief and approves the required inputs, preferences, directives, and inferred assumptions.
