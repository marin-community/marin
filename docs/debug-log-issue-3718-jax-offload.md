# Debugging log for issue 3718 JAX offload repro

Build a minimal JAX-only reproducer where training stays finite without checkpoint offload and produces NaNs with checkpoint offload enabled on TPU.

## Initial status

Issue [#3718](https://github.com/marin-community/marin/issues/3718) reports NaNs on TPU when a checkpoint offload mode is enabled for attention-path intermediates. The repo itself currently pins JAX 0.8.0, but the requested target is to verify whether the problem still reproduces on `jax==0.9.1` and `jaxlib==0.9.1`.

## Hypothesis 1

The failure is caused by JAX checkpoint offloading itself, not by higher-level Marin, Levanter, Haliax, or Optax logic.

## Changes to make

- Add a standalone JAX-only reproducer script.
- Keep the model synthetic and small enough for `v5p-8`.
- Expose checkpoint mode and named offload targets as CLI flags.

## Future Work

- [ ] Determine whether the bug requires multi-device sharding.
- [ ] Determine the smallest offloaded tensor subset that still reproduces NaNs.
- [ ] Capture the first NaN step and compare against non-offload controls on `jax==0.9.1`.

## Results

- Added [`experiments/jax_checkpoint_offload_nan_repro.py`](../experiments/jax_checkpoint_offload_nan_repro.py), a standalone JAX-only synthetic training reproducer with switches for:
  - checkpoint mode: `none` / `recompute` / `save` / `offload`
  - attention backend: dense reference attention or TPU Splash attention
  - checkpoint naming scope: global tensors or local tensors inside `shard_map`
- Verified local CPU smoke tests for the reference path.
- Verified a clean TPU env on `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.35`
- On `v5p-8`, all attempted control/offload pairs remained finite on JAX 0.9.1:
  - reference attention, single-device, `save` vs `offload`
  - reference attention, 4-chip mesh, `save` vs `offload`
  - Splash attention under `shard_map`, global checkpoint names, `save` vs `offload`
  - Splash attention under `shard_map`, local checkpoint names, `save` vs `offload`
- The most aggressive stable run so far used:
  - `--attention-backend splash`
  - `--checkpoint-scope local`
  - `--checkpoint-mode offload`
  - `--mesh-dim 4`
  - `--num-layers 8`
  - `--learning-rate 0.10`
  - `--weight-scale 0.12`
  - `--logits-dtype bfloat16`
  - It produced very large activations (`max_abs_hidden ~= 1.76e4`) but still no NaNs through 8 steps.
- Current conclusion: I do not yet have evidence that the issue still reproduces on JAX 0.9.1. The remaining possibilities are that:
  - the original bug was fixed between 0.8.0 and 0.9.1, or
  - the reproducer still misses one crucial aspect of the original failing path.

## Hypothesis 2

The missing ingredient is the ragged MoE communication path rather than attention offload alone.

## Changes to make

- Add the user's smaller JAX-only repro with explicit mesh + ragged `all_to_all` MoE routing.
- Add two ablation toggles:
  - disable MoE
  - disable `jax.checkpoint` wrapping entirely

## Results

- Added [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py).
- Verified on TPU `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.37`
- Using the reported config:
  - `REPRO_SEQ_LEN=128`
  - `REPRO_NUM_LAYERS=2`
  - `REPRO_HIDDEN=2`
  - `REPRO_NUM_HEADS=1`
  - `REPRO_HEAD_DIM=2`
  - `REPRO_NUM_EXPERTS=4`
  - `REPRO_TOPK=1`
  - `REPRO_MLP_DIM=1`
  - `REPRO_INIT_STD=0.02`
  - `REPRO_LR=4`
  - `REPRO_STEPS=30`
  - `REPRO_SEED=0`
- Observed matrix:
  - baseline: `REPRO_USE_OFFLOAD=1` -> `result=nan first_nan_step=1`
  - control: `REPRO_USE_OFFLOAD=0` -> `result=no_nan`
  - ablation: `REPRO_USE_OFFLOAD=1 REPRO_DISABLE_MOE=1` -> `result=no_nan`
  - ablation: `REPRO_USE_OFFLOAD=1 REPRO_DISABLE_CHECKPOINT=1` -> `result=no_nan`
- Current conclusion: the failure requires both:
  - the MoE path
  - checkpoint/offload wrapping

It does not reproduce with attention-only residual blocks under the same tiny unstable regime, and it does not reproduce when the checkpoint wrapper is removed even if `REPRO_USE_OFFLOAD=1` remains set.

## Hypothesis 3

Attention is not required; the essential interaction is checkpoint offload plus the MoE/ragged communication path.

## Changes to make

- Add an attention ablation toggle.
- Run the same offload/control split with attention replaced by identity.

## Results

- Added `REPRO_DISABLE_ATTENTION` to [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py).
- Using the same tiny config and keeping MoE + checkpointing intact:
  - `REPRO_USE_OFFLOAD=1 REPRO_DISABLE_ATTENTION=1` -> `result=nan first_nan_step=1`
  - `REPRO_USE_OFFLOAD=0 REPRO_DISABLE_ATTENTION=1` -> `result=no_nan`
- Current conclusion: attention is not required for the failure. The minimal dependency we have demonstrated is:
  - MoE/ragged path present
  - checkpoint/offload wrapper present
  - attention optional

## Hypothesis 4

The critical MoE ingredient is specifically the cross-device `ragged_all_to_all`, not merely local expert dispatch.

## Changes to make

- Add a `REPRO_DISABLE_RAGGED_A2A` ablation that keeps MoE structure but replaces cross-device routing with local-only dispatch.
- Run the no-attention variant to isolate the MoE communication path.

## Results

- Added `REPRO_DISABLE_RAGGED_A2A` to [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py).
- Using:
  - `REPRO_USE_OFFLOAD=1`
  - `REPRO_DISABLE_ATTENTION=1`
  - `REPRO_DISABLE_RAGGED_A2A=1`
  - same tiny config otherwise
  - result: `no_nan`
- Control with `REPRO_USE_OFFLOAD=0 REPRO_DISABLE_ATTENTION=1 REPRO_DISABLE_RAGGED_A2A=1` also remains `no_nan`.
- Current conclusion: the demonstrated failing core is narrower than “MoE + offload”. The failure depends on:
  - checkpoint/offload wrapper
  - cross-device ragged MoE routing via `ragged_all_to_all`

Attention is not required, and local-only expert dispatch is not sufficient to trigger the NaN.

## Hypothesis 5

The remaining core may be simpler than the full original named-tensor policy: perhaps only checkpoint-offloaded block carry plus ragged routing is enough.

## Changes to make

- Add `REPRO_OFFLOAD_BLOCK_INPUT_ONLY` to offload only `block_input`.
- Add `REPRO_FIXED_ASSIGNMENTS` to remove router/top-k and feed deterministic expert assignments into the ragged path.

## Results

- `REPRO_USE_OFFLOAD=1 REPRO_DISABLE_ATTENTION=1 REPRO_OFFLOAD_BLOCK_INPUT_ONLY=1` -> `result=nan first_nan_step=1`
- Matching control with `REPRO_USE_OFFLOAD=0 ... REPRO_OFFLOAD_BLOCK_INPUT_ONLY=1` -> `result=no_nan`
- So the attention-path checkpoint names are not required. Offloading `block_input` alone is sufficient when ragged routing is present.
- `REPRO_USE_OFFLOAD=1 REPRO_DISABLE_ATTENTION=1 REPRO_OFFLOAD_BLOCK_INPUT_ONLY=1 REPRO_FIXED_ASSIGNMENTS=1` -> `result=no_nan`
- Current conclusion: we can strip the repro to:
  - checkpoint-offloaded `block_input`
  - cross-device ragged `all_to_all`
  - no attention

But we cannot yet strip out the router/top-k behavior entirely. Deterministic fixed assignments are too simple and no longer reproduce.

## Hypothesis 6

RMSNorm may be part of the minimal failing surface in the stripped no-attention repro.

## Changes to make

- Add `REPRO_DISABLE_RMS_NORM`.
- Run the already-stripped case:
  - no attention
  - `block_input` offload only
  - ragged `all_to_all` still enabled

## Results

- Added `REPRO_DISABLE_RMS_NORM` to [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py).
- Using:
  - `REPRO_USE_OFFLOAD=1`
  - `REPRO_DISABLE_ATTENTION=1`
  - `REPRO_OFFLOAD_BLOCK_INPUT_ONLY=1`
  - `REPRO_DISABLE_RMS_NORM=1`
  - result: `no_nan`
- Matching control with `REPRO_USE_OFFLOAD=0 ... REPRO_DISABLE_RMS_NORM=1` also gives `no_nan`.
- Current conclusion: RMSNorm matters in the current stripped repro. With RMSNorm removed, the offload-only NaN disappears.

## Hypothesis 7

The repro can be simplified further by deleting attention and all ablation knobs, as long as the surviving MoE parameters keep the same random key positions as in the last known failing version.

## Changes to make

- Remove attention entirely from [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py).
- Remove the now-unnecessary ablation knobs from the file.
- Preserve the original PRNG split positions for router and expert weights so simplification does not accidentally change the failing initialization.

## Results

- A first stripped version removed attention successfully, but also reassigned the router and expert parameter keys to earlier split positions.
- That change erased the split:
  - `REPRO_USE_OFFLOAD=1` -> `result=no_nan`
  - `REPRO_USE_OFFLOAD=0` -> `result=no_nan`
- Restoring the original key positions for the surviving MoE weights brought the split back without reintroducing attention or the old ablation knobs.
- Verified on TPU `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.37`
- Current stripped file behavior:
  - `REPRO_USE_OFFLOAD=1` -> `result=nan first_nan_step=1`
  - `REPRO_USE_OFFLOAD=0` -> `result=no_nan`
- Current conclusion: the simplified repro can omit attention entirely, but the tiny failing regime is sensitive to initialization. Preserving the original random-key alignment is part of the minimal reproducing artifact.

## Hypothesis 8

The observed split is not just a fragile optimizer trajectory. If checkpoint offload is causing a real bug, the step-0 forward loss can remain identical while the step-0 backward pass already differs materially before any update-driven NaN amplification.

## Changes to make

- Compare one training step from the same initialization under:
  - `REPRO_USE_OFFLOAD=0`
  - `REPRO_USE_OFFLOAD=1`
- Hold inputs, targets, and parameters fixed.
- Inspect:
  - pre-update loss
  - step-0 gradients
  - parameters after one update

## Results

- On TPU `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.37`
- The pre-update forward losses are bit-identical:
  - `loss_offload_0=1.8225864171981812`
  - `loss_offload_1=1.8225864171981812`
  - `loss_abs_diff=0.0`
- Despite that, several step-0 gradients differ materially in the first layer:
  - `0/w2: max_abs_diff=1.686697e-04, max_rel_diff=1.168`
  - `0/w_up_gate: max_abs_diff=9.972733e-05, max_rel_diff=1.262`
  - `0/rms: max_abs_diff=2.622406e-06, max_rel_diff=1.236`
- The post-update parameters also differ immediately:
  - `0/w2: max_abs_diff=6.746789e-04, max_rel_diff=0.173`
  - `0/w_up_gate: max_abs_diff=3.989092e-04, max_rel_diff=1.018`
- All compared gradients remained finite in this step-0 comparison; the divergence is not just “offload grads are already NaN”.
- Current conclusion: this looks stronger than a benign roundoff perturbation. The forward pass is unchanged, but checkpoint offload changes the backward result by order-1 relative amounts in some leaves before the optimizer has a chance to amplify anything.

## Hypothesis 9

The repro should be framed directly as a gradient mismatch, not as a training instability. A cleaner JAX issue artifact is: same inputs, same params, same forward loss, but `offload=0` and `offload=1` gradients are not allclose.

## Changes to make

- Rewrite [`experiments/jax_issue_3718_ragged_repro.py`](../experiments/jax_issue_3718_ragged_repro.py) to:
  - remove the training-to-NaN loop
  - compute step-0 gradients once with offload disabled
  - compute step-0 gradients once with offload enabled
  - compare the two gradient pytrees leafwise with configurable `RTOL` / `ATOL`
- Make the offload choice an explicit function argument instead of a mutable module global.

## Results

- Verified on TPU `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.37`
- The rewritten script reports:
  - `loss_offload_0=1.8225864171981812`
  - `loss_offload_1=1.8225864171981812`
  - `loss_abs_diff=0.0`
- With default tolerances `REPRO_ATOL=1e-6` and `REPRO_RTOL=1e-2`, the following leaves are not allclose:
  - `0/rms`
  - `0/w2`
  - `0/w_up_gate`
- Example output:
  - `grad_leaf=0/w2 finite=True allclose=False max_abs_diff=0.0001686697214609012 max_rel_diff=1.168185830116272`
  - `grad_leaf=0/w_up_gate finite=True allclose=False max_abs_diff=9.972733096219599e-05 max_rel_diff=1.2617436647415161`
- Final script result:
  - `result=grads_not_allclose`
- Current conclusion: this is a cleaner issue artifact than the NaN-based framing. The observed problem is now directly stated as a backward inconsistency under checkpoint offload.

## Hypothesis 10

The reproducer can be simplified further than the original gradient-mismatch version: fixed input-dependent routing should be enough, the expert path can be linear, and a second layer or learned router weights may not be required.

## Changes to make

- Test targeted ablations on TPU:
  - random routing
  - linear expert instead of gated MLP
  - drop `w2` as a parameter
  - fixed input-dependent routing instead of learned router weights
  - one layer instead of two

## Results

- Verified on TPU `v5p-8` with:
  - `jax==0.9.1`
  - `jaxlib==0.9.1`
  - `libtpu==0.0.37`
- Observed matrix:
  - baseline gradient repro: `result=grads_not_allclose`
  - `REPRO_RANDOM_ROUTING=1`: `result=grads_allclose`
  - `REPRO_LINEAR_EXPERT=1`: `result=grads_not_allclose`
  - `REPRO_RANDOM_ROUTING=1 REPRO_LINEAR_EXPERT=1`: `result=grads_allclose`
  - `REPRO_LINEAR_EXPERT=1 REPRO_DROP_W2_PARAM=1`: `result=grads_not_allclose`
  - `REPRO_FIXED_ROUTING=1 REPRO_LINEAR_EXPERT=1 REPRO_DROP_W2_PARAM=1`: `result=grads_not_allclose`
  - `REPRO_NUM_LAYERS=1`: `result=grads_allclose`
- A fully stripped one-layer file with:
  - fixed input-dependent routing
  - linear expert
  - no `w2`
  - no combine-weight path
  - no extra dummy `w2` argument
  - unexpectedly became `result=grads_allclose`
- Restoring the mathematically-unused `combine_weights` path and a dummy sharded `w2` argument brought the mismatch back in the one-layer simplified file:
  - `loss_offload_0=1.8217421770095825`
  - `loss_offload_1=1.8217421770095825`
  - `grad_leaf=0/rms finite=True allclose=False max_abs_diff=0.06031022220849991 max_rel_diff=1.6602920293807983`
  - `grad_leaf=0/w_linear finite=True allclose=True max_abs_diff=0.0 max_rel_diff=0.0`
  - `result=grads_not_allclose`
- Current conclusion: the sharpest known repro is now:
  - one layer
  - fixed input-dependent routing
  - linear expert
  - no learned router weights
  - no `w2` parameter
  - but still with the extra `combine_weights` path and dummy sharded `w2` argument preserved inside the `shard_map` call

This suggests the bug is sensitive not just to the visible math, but also to the structure of the `shard_map` inputs and/or the resulting lowered program.
