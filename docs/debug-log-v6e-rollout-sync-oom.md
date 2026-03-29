# Debugging log for v6e rollout sync OOM

Understand why RL rollout workers on `v6e-8` fail even though ordinary Llama 8B
inference is known to run on that hardware, then define the next experiment ladder.

## Initial status

Fresh `v4` runs in:

- `/ahmed/iris-rl-v6e-e1d-0328-v4`
- `/ahmed/iris-rl-v6e-e5b-0328-v4`

cleared the earlier trainer-side multi-host Arrow Flight bug and the rollout
false-success-on-trainer-retry bug.

Both runs now fail in the rollout worker at bootstrap weight application.

## Hypothesis 1

This is not a generic "8B does not fit on `v6e-8`" problem. It is a rollout
hot-reload memory problem.

## Changes to make

- Collect exact rollout-side traceback and HBM stats
- Compare those to the rollout engine configuration (`tensor_parallel_size`,
  `gpu_memory_utilization`, `max_model_len`)
- Expose the rollout inference memory knobs in the region-aware experiment
  launcher so the next probes do not require one-off edits

## Results

What the live logs prove:

- The trainer bootstrap serve succeeds.
- The rollout bootstrap receive succeeds.
- The failure happens in `reload_model() -> sync_weights()`, not in ordinary
  token generation.

Concrete evidence from `/ahmed/iris-rl-v6e-e5b-0328-v4`:

- rollout log sequence:
  - `Received 291 params for weight_id -1 via Arrow Flight`
  - `reload_model: starting prefix cache reset`
  - `reload_model: converting state dict`
  - `reload_model: calling sync_weights`
- crash site:
  - `tpu_inference ... transfer_state_with_mappings`
  - `jax.device_put(...)`
  - `x._multi_slice(...)`
  - `RESOURCE_EXHAUSTED`
- allocator summary:
  - HBM limit: `33,550,233,600`
  - in use: `31,245,339,136`
  - free: about `1.17 GiB`
  - largest contiguous free block: about `448 MiB`
- the failing tensor right before the crash is:
  - `lm_head`
  - shape `(128256, 4096)`
  - dtype `bfloat16`
- the failing allocation is about `1002 MiB`, which matches that tensor.

Important configuration facts:

- the rollout job is a `v6e-8` slice, so the worker has 8 chips available
- but the current rollout vLLM config uses `tensor_parallel_size=4`
- the live kv-cache init log shows a mesh with `model: 4`, confirming only 4
  chips are participating in inference
- the same log shows the engine reserves about `28.12 / 31.25 GiB` HBM per
  active chip at `gpu_memory_utilization=0.90`

This explains why normal inference compatibility does not contradict the crash:

- steady-state inference fits
- RL bootstrap weight hot-reload needs extra temporary device buffers while
  resharing incoming tensors into the live TPU mesh
- with the cache reservation already taking most HBM, that extra temporary
  allocation fails due fragmentation

## Hypothesis 2

`gpu_memory_utilization=0.90` is too aggressive for RL hot-reload on `v6e-8`.

Prediction:

- lowering cache reservation should allow the ~`1002 MiB` `lm_head` transfer to
  find contiguous HBM and finish `sync_weights()`

## Hypothesis 3

`tensor_parallel_size=4` is wasting half of the `v6e-8` chips for this rollout
shape.

Prediction:

- increasing rollout TP to `8` should change the hot-reload memory profile and
  may reduce the effective pressure enough for bootstrap reload to succeed

## Hypothesis 4

Reducing rollout batch size is not the first lever.

Reason:

- the crash happens at `step=0`, before the first generation batch is produced
- so `n_prompts` / `n_generations_per_prompt` do not affect this bootstrap OOM

## Future Work

- [ ] Run short one-rollout-worker feasibility probes before another full
      500-step, 2-worker run
- [ ] Sweep rollout `gpu_memory_utilization`
- [ ] Sweep rollout `tensor_parallel_size`
- [ ] If needed, reduce rollout `max_model_len` as a last-resort cache knob

## Hypothesis 4

Increasing TP to `8` alone is not enough.

## Results

Ran short feasibility probe:

- root job: `/ahmed/iris-rl-v6e-e5b-rol-v6e8-002`
- rollout: `v6e-8`, `tensor_parallel_size=8`, `gpu_memory_utilization=0.90`
- trainer: `v6e-16`
- one rollout worker, `20` train steps, `eval_frequency=1000`

Observed outcome:

- trainer bootstrap serve succeeded
- rollout bootstrap receive succeeded
- rollout still failed in `reload_model -> sync_weights`
- root job failed after rollout exhausted retries

Most important negative result:

- TP=`8` did **not** remove the ~`1002 MiB` temporary allocation failure

Allocator comparison:

- TP=`4` run largest contiguous free block: about `448 MiB`
- TP=`8` run largest contiguous free block: about `541 MiB`

So TP=`8` improved fragmentation slightly but not enough to cross the threshold.

Next best hypothesis:

- lower `gpu_memory_utilization` while keeping TP=`8`
