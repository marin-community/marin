# TRC Experiment Simulator

This is a standalone black-box simulator for TRC execution and experiment results. A scenario combines an experiment definition with a TRC environment; `run-adaptive-sweep` is one client used to evaluate orchestration behavior without live Iris, W&B, TPU, or network calls.

Keep these layers distinct:

- **Experiment shape:** parameter axes, resource levels, objective surface, and hidden optima.
- **TRC environment:** target throughput over a declared horizon, chip capacity, outages, retries, and regional resume semantics.
- **Evaluation adapter:** the policy, sweep tools, SQLite record, and blind Orchestrator used for a particular test.

The bundled scenarios are shaped like the sweep under evaluation, but learning rate, weight decay, and epochs are scenario parameters rather than intrinsic simulator concepts.

## Components

- `.agents/skills/run-adaptive-sweep/scripts/trc_simulator.py` is the black-box simulator.
- `.agents/skills/run-adaptive-sweep/scripts/test_trc_simulator.py` verifies its state-transition invariants through the CLI.
- `.agents/skills/run-adaptive-sweep/scripts/sweep_tools.py` supplies convergence, objective prediction, throughput, placement, and recovery calculations.
- Full state histories, agent transcripts, plots, and reports belong under `scratch/`, not in the skill.

Run deterministic tests with:

```text
uv run --with scikit-learn pytest -q \
  .agents/skills/run-adaptive-sweep/scripts/test_sweep_tools.py \
  .agents/skills/run-adaptive-sweep/scripts/test_trc_simulator.py
```

## Simulator Model

The simulator exposes `init`, `status`, `snapshot`, `predict`, `targets`, `launch`, `expand-grid`, `relocate`, `advance`, and `report`. `report` rejects access until every rung strictly converges. `targets` takes a zero-based rung index. Simulator `point` values are zero-based scenario coordinates; the adapter translates them to the sweep tools' stable axis-value points. Each `launch` action requires `rung`, `point`, and `target`; `expand-grid` requires `axis` and a declared `value`; `relocate` requires `trial_id` and `target`. `advance` returns every retry and completion that occurred within the requested steps so short failures cannot disappear between polls.

The current adaptive-sweep fixture uses:

- A `7 x 7` learning-rate and weight-decay grid with resource rungs and ratios `[1, 4, 16, 64]`.
- A deterministic quadratic objective with an interaction term and hash-derived local roughness. Its hidden interior center moves one grid step between the first two rungs and then remains fixed.
- A 15-minute observation step, target horizon equal to the three-week sweep wall-time limit, and a 64-chip limit.
- Explicit target throughput schedules over that horizon: `us-east5/v5p-32` and `us-central1/v5p-32` are constant; `us-east5/v5p-64` and `europe-west4/v6e-32` improve linearly; `us-east5/v5p-128` degrades linearly while remaining the fastest nominal slice.
- One outage window per region and two additional outage windows for `us-east5/v5p-128`. Outages remove capacity without changing the underlying throughput schedule.
- `us-east5/v5p-32` at 16 chips, `us-east5/v5p-64` at 32, `us-east5/v5p-128` at 64, `us-central1/v5p-32` at 16, and `europe-west4/v6e-32` at 32. Early conditions deliberately make nominal size a poor placement proxy.
- Progress equal to hidden normalized work divided by the rung's resource ratio. A regional run appears in W&B only after it first receives compute.
- Retryable Iris attempts that can fail after W&B progress. The simulated Dispatcher resubmits the same dispatch and target with a new Iris job ID, preserves regional progress, and eventually succeeds; only explicit Orchestrator relocation creates a new dispatch.
- Same-region TPU moves preserving regional identity, progress, and W&B registration. A new region starts at zero; returning later to an old region restores that region's prior state.
- Strict uniqueness of `(grid point, resource rung)`, one objective per logical trial, explicit chip accounting, and rejected invalid actions.

Throughput and outage times are normalized to the target horizon:

| Target | Start rate | End rate | Capacity | Additional outage |
| --- | ---: | ---: | ---: | --- |
| `us-east5/v5p-32` | 0.85 | 0.85 | 32 chips | none |
| `us-east5/v5p-64` | 0.90 | 1.80 | 64 chips | none |
| `us-east5/v5p-128` | 3.00 | 1.60 | 64 chips | `[0,.15)`, `[.75,.85)` |
| `us-central1/v5p-32` | 1.05 | 1.05 | 32 chips | none |
| `europe-west4/v6e-32` | 0.65 | 1.45 | 32 chips | none |

Region-wide outages are `us-east5 [.45,.50)`, `us-central1 [.70,.75)`, and `europe-west4 [.20,.25)`.

Pass `--seed N` to `init` for a reproducible regression. Omit it for a fresh sealed scenario. Set `--time-horizon` to the policy wall-time limit and `--full-exploitation-level` to the policy's selected resource level.

Use `--scenario predictable-off-grid` to test grid expansion. Its initial learning-rate axis stops short of the hard bound, and its sealed centers move monotonically toward larger learning rates and cross the initial edge by resource 16. The next declared value is the preferred extension. The simulator accepts a later declared value only when the action includes a reason, and records that resolution deviation. This finite declared sequence is a scenario constraint, not a production expansion policy.

The failure model is synthetic. Its shape reflects checked-in Iris incident records—short repeated failures, long-lived bad placements, and later resume from progress—but no rates were fitted from real Iris data. The simulator caps a dispatch at two injected failures to keep evaluation finite.

## Blind Agent Protocol

Before spawning, save a complete policy containing the simulator command and a fresh local SQLite path. Express the wall-time limit in the simulator's numeric time unit; for the bundled fixture, use hours such as `504h`, not mixed units such as `3 weeks`. Do not identify the scenario, expected behavior, or evaluation criteria in the policy. The Orchestrator receives only that policy, the execution skill, and the private-state access rule.

Render this prompt without adding requests for particular decisions or reporting:

```text
Act as the sole Orchestrator.

Execute the approved policy at {policy_path} using the skill at {skill_path}.
Use the policy's local SQLite database for orchestration state and its simulator
command for all runtime actions.

The simulator's state file is {state_path}. Pass this path to simulator commands,
but never read, search, import, execute, or otherwise inspect the file directly.
Treat the simulator CLI as an external runtime boundary. Do not inspect its source
or any simulator evaluation materials.

Run autonomously until the policy's stopping condition or limit is reached. Do not
request guidance unless execution is blocked. On termination, return only the final
status or blocker; the evaluator will derive behavior from SQLite and runtime logs.
```

Launch it without inherited conversation context:

```text
multi_agent_v1.spawn_agent({
  agent_type: "default",
  fork_context: false,
  message: rendered_prompt,
})
```

Wait with `multi_agent_v1.wait_agent` and do not send guidance while it runs. Checking file metadata for liveness is acceptable, but the parent must not inspect simulator or SQLite contents. After termination, close the subagent, invoke the gated `report`, and independently analyze SQLite and simulator history. Treat the agent's final narrative as ancillary, not as evidence of behavior.

After changing policy in response to a failure, create a fresh policy, SQLite database, simulator state, and `fork_context: false` agent. Do not disclose the prior failure mode or expected correction. Record failed or no-op CLI calls as protocol deviations even when they did not mutate state.

## Evaluation Criteria

- Upper-rung trial counts should usually fall as lower-rung evidence improves forecasts.
- Higher rungs should start before lower rungs finish when forecasts are informative, but weak early differences must not trigger expensive over-sampling.
- Target exploration should narrow with resource level while recent normalized throughput can displace a stalled nominal favorite.
- Changing `full_exploitation_level` should predictably change target-pool breadth, with exactly one placement candidate at and above that level.
- Initialization recovery, registered-run stalls, same-region resume, and cross-region reset must remain distinct.
- A failed attempt after positive progress should retain one logical trial and dispatch, resume from that progress, and eventually succeed.
- In `predictable-off-grid`, the Orchestrator should infer the trajectory, use the declared resolution as the normal extension spacing, record any reasoned deviation, and converge only after testing the expanded region.
- Strict convergence should match hidden optima without duplicate trials or unjustified grid expansion.
- The chip budget should remain full until every rung converges whenever another unique allowed trial fits. Report unavoidable slack.

## Suggested Visuals

Generate plots after `report` unlocks hidden truth. These four views were useful in the original evaluation, but they are suggestions rather than a required output set; add or replace views when another diagnostic explains the run better.

1. **Sampled objective grids:** one hidden-objective heatmap per resource rung, with completed trials numbered by launch order, active trials distinguished, and the final strict dominant point marked.
2. **Global forecast accuracy:** over virtual time, plot both Manhattan grid distance from the model's predicted global optimum to the hidden center and actual objective regret at that predicted point. Reconstruct the model from trials completed at each forecast time and score the entire grid, including completed and active points. Do not use only the remaining launchable candidates; that creates an artificial distance plateau after the center and its neighbors become occupied.
3. **Chip utilization and convergence:** plot inflight chips against the policy limit, with vertical markers at each rung's first strict-convergence time.
4. **Target throughput history:** use separate facets for the Orchestrator's observed placement score and the simulator's true throughput schedule over the actual sweep span. Mark relocations, retries, outages, and grid changes without blending hidden truth into the observed estimate.

Keep generated images and plotting code under the run's `scratch/` directory unless a plotting utility becomes stable enough to test and reuse.

## Independence

The evaluation is independent at the decision layer. A fresh agent receives only a scenario-neutral policy and the execution skill, with no prior conversation, evaluator reference, expected trajectory, hidden objective, or hidden capacity data. The CLI enforces actions, hidden truth is gated, and the parent neither steers the agent nor inspects state while it runs. Behavior is reconstructed from SQLite and runtime records rather than solicited in the prompt.

It is not an independent implementation audit. The adaptive-sweep adapter and Orchestrator share `sweep_tools.py`, and the bundled experiment scenario was authored for this workflow. The TRC model intentionally simplifies outages and retries. Unit tests validate tool and simulator arithmetic. Multiple sealed scenarios strengthen behavioral evidence, but a successful run does not prove optimal scheduling or production reliability.
