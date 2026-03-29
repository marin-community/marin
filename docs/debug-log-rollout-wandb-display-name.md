# Debugging log for rollout W&B display-name collision

Investigate why the two rollout W&B runs for `iris-rl-e4ms2-500` show the same
display name in the W&B sidebar even though the run URLs and ids are distinct.

## Initial status

The two rollout run URLs are:

- <https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-rollout-0>
- <https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-rollout-1>

But the W&B sidebar shows both runs with the same visible name:

- `iris-rl-e4ms2-500-20260328-031315-rollout`

## Hypothesis 1

The two rollout workers collided on W&B run id and one run partially overwrote the
other.

## Changes to make

None. Inspect the W&B metadata for both rollout runs directly.

## Results

Rejected.

The rollout runs have distinct W&B ids and URLs:

- rollout worker `0`
  - `id = iris-rl-e4ms2-500-rollout-0`
  - `url = https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-rollout-0`
- rollout worker `1`
  - `id = iris-rl-e4ms2-500-rollout-1`
  - `url = https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-rollout-1`

So the rollout runs are distinct objects in W&B. This is not an id collision.

## Hypothesis 2

The rollout workers were given a shared W&B display name while keeping unique run ids.

## Changes to make

None yet. Trace the rollout tracker naming path and compare it with the experiment
configuration.

## Results

Confirmed.

The rollout run metadata shows:

- both runs have the same `name` / `display_name`:
  - `iris-rl-e4ms2-500-20260328-031315-rollout`
- but their ids remain unique:
  - `iris-rl-e4ms2-500-rollout-0`
  - `iris-rl-e4ms2-500-rollout-1`

This means the W&B sidebar confusion is a display-name bug, not a run-identity bug.

Related configuration context:

- the experiment config sets a shared rollout tracker name:
  - `name=f"{instance_name}-rollout"`
- the coordinator still assigns unique rollout worker run ids with per-worker suffixes:
  - `...-rollout-0`
  - `...-rollout-1`

## Health check

The rollout runs themselves look broadly healthy apart from the final trainer-driven
shutdown:

- rollout worker `0`
  - has the full `inference.eval/...` series
  - summary includes `inference.eval/math_full/avg_at_1 = 0.432`
  - summary includes `inference.eval/math_full/total_count = 500`
- rollout worker `1`
  - has no `inference.eval/...` series, which is expected because full eval is gated
    to `worker_index == 0`
  - still logs normal rollout metrics such as `inference.rollout/math_full/pass_at_16`

So the duplicate display names are cosmetic. They do not indicate that eval or rollout
execution was merged incorrectly.

## Future Work

## Hypothesis 3

The robust fix is to make the rollout tracker use a stable per-worker W&B display name
that matches the stable per-worker run id, not the volatile Iris instance name.

## Changes to make

- Update the rollout tracker to use `config.name or run_id` for W&B `name`
- In the coordinator, stamp each rollout worker's `tracker_config.name` to the same
  stable per-worker id:
  - `...-rollout-0`
  - `...-rollout-1`

## Results

Implemented.

The important design choice is that rollout W&B naming now keys off stable logical
worker identity, not the timestamped coordinator instance id:

- W&B `id`: stable per-worker id
- W&B `name`: same stable per-worker id
- Iris child job name: still free to use the volatile instance-specific name

That makes the W&B labels robust to preemption retries while keeping the visible names
unique in the sidebar.

## Future Work

- [ ] Consider whether rollout receive counters should also be split into attempt-local vs resume-safe metrics, similar to the trainer-side weight-transfer counters
