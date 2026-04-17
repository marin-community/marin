# Debugging log for ragged-all-to-all

Investigating why the 1e23 Grug MoE run diverges under `ragged_all_to_all` while the ring configuration is healthy.

## Initial status

The `moe_1e23_d5120_bs2048_ep8_ragged_48l_rayuvtpu_20260417_011404` run started with healthy step-0 metrics, then diverged later in training. By step 1000 it was far behind the old `ep4_ring` baseline and by step 1250 it showed multi-million gradient norms, followed by `NaN` eval at step 1259.

## Hypothesis 1

The ragged dispatch path is not semantically equivalent to the ring path under expert parallelism and later-training router distributions. Capacity clipping, dropped-assignment accounting, or recombination may diverge in a way that only appears once routing sharpens.

## Changes to make

- Relaunch the 1e23 config as ring `ep4` on the current Ray cluster for a fresh control run.
- Audit `experiments/grug/moe/model.py` and `lib/levanter/src/levanter/grug/grug_moe.py`.
- Run targeted TPU tests on `v5p-8` and `v5p-32` to compare ring vs ragged gradients for the functional MoE MLP block.

## Future Work

- [ ] Check whether ring and ragged produce materially different MLP block gradients on TPU pods.
- [ ] Confirm whether dropped-assignment behavior differs between implementations at equal capacity.
- [ ] Compare router-sharpening behavior after a few hundred optimization steps, not just at initialization.

## Results

- The host-side routing simulation matches ring exactly, so the abstract clip/permute/unpermute math was not the bug.
- The actual bug was in [grug_moe.py](/Users/dlwh/.codex/worktrees/8989/marin/lib/levanter/src/levanter/grug/grug_moe.py): `_shard_a2a_params` was feeding `jax.lax.ragged_all_to_all` receiver-local output offsets instead of the sender-side remote offsets that the primitive expects.
  - JAX internally transposes `output_offsets` with an `all_to_all`.
  - Our old code pre-transposed them, so real distributed runs wrote returned slices into the wrong positions.
  - That explains why a pure Python/JAX simulation of the routing logic looked correct while TPU runs showed large ring-vs-ragged output and gradient deltas.
- Existing EP coverage in [test_grugformer_moe.py](/Users/dlwh/.codex/worktrees/8989/marin/lib/levanter/tests/grug/test_grugformer_moe.py) only checks output shape and finiteness, not ring-vs-ragged parity.
- Fresh ring control relaunch:
  - Ray submission: `ray-run-dlwh-moe-uvtpu-ep4-ring-manual-20260417_152005`
  - W&B run id (expected once training initializes): `moe_1e23_d5120_bs2048_ep4_ring_rayuvtpu_20260417_152005`
  - The first relaunch attempt via `ray_run.py` failed during Ray runtime-env pip setup because `kitoken==0.10.2` was not available through the cluster-visible pip indexes.
  - Manual `ray job submit` without Ray pip runtime-env is now running and has reached executor dispatch for `grug/moe_1e23_d5120_bs2048_ep4_ring`.
- TPU parity probes:
  - Initial `v5p-8` and `v5p-32` attempts failed because the probe lived under untracked `scratch/`, which the Iris workspace bundle did not include.
  - The probe now lives at [scripts/debug/grug_moe_grad_compare.py](/Users/dlwh/.codex/worktrees/8989/marin/scripts/debug/grug_moe_grad_compare.py) and compiles locally.
  - First tracked-path jobs submitted:
    - `/dlwh/grug-moe-grad-compare-v5p8-20260417-0828`
    - `/dlwh/grug-moe-grad-compare-v5p32-20260417-0828`
  - Those jobs later failed with entrypoint container OOM (`exit 137`) under the default `1GB` host-memory request, so they did not yet exercise the MoE kernel.
  - Region-widened jobs were submitted next:
    - `/dlwh/grug-moe-grad-compare-v5p8-20260417-0833`
    - `/dlwh/grug-moe-grad-compare-v5p32-20260417-0833`
  - Final corrected jobs use both `us-central1` and `us-east5` plus `--memory 8GB`:
    - `/dlwh/grug-moe-grad-compare-v5p8-20260417-0835`
    - `/dlwh/grug-moe-grad-compare-v5p32-20260417-0835`
  - Successful `v5p-8` probe after the `_shard_a2a_params` fix:
    - Job: `/dlwh/grug-moe-grad-compare-v5p8-20260417-094052`
    - Normal routed case now matches:
      - `ring_loss == ragged_loss == 995541.4375`
      - `ring_dropped == ragged_dropped == 9`
      - `output_diff.rel_l2 = 4.17e-08`
      - `grad_x_diff.rel_l2 = 4.20e-08`
      - `grad_w_up_gate_diff = 0`
      - `grad_w_down_diff = 0`
    - Forced-overflow case still matches exactly with zero diffs.
  - Local regression coverage added:
    - `_shard_a2a_params` now has a unit test asserting sender-side output offsets.
    - A parity test now checks `ring` vs `ragged_all_to_all` MoE outputs when EP is available on a non-CPU backend.
