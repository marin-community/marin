# Dion CPU Test Plan

File: `lib/levanter/tests/test_dion.py`

---

## Purpose and Scope

This test suite validates the Dion optimizer port on **CPU only**, including
simulated multi-device execution via
`XLA_FLAGS=--xla_force_host_platform_device_count=N`. Its job is to prove
three things:

1. **Algorithmic correctness** — the implementation reproduces the paper's
   update rule exactly when compared against a manual reference computation.
2. **Sharding equivalence** — running Dion under a simulated multi-device mesh
   produces bit-identical results to running it on a single device.
3. **Integration** — the optimizer wires into Levanter's config registry,
   multi-transform routing, scan layers, and LR scheduling correctly.

### What This Suite Does NOT Prove

- **TPU numerical equivalence.** CPU LAPACK QR and TPU XLA QR produce
  orthonormal bases that span the same subspace but may differ in column
  signs/rotations. The algorithm remains correct (any valid Q works), but
  outputs won't be bit-identical across backends.
- **Performance.** Wall-clock time, memory pressure, kernel fusion quality —
  none of these are observable on simulated CPU devices.
- **Multi-host communication.** Simulated devices share memory and use memcpy
  for collectives. ICI/DCN topology effects, bandwidth saturation, and
  latency-bound behavior are invisible here.
- **bfloat16 accumulation.** CPU emulates bf16 differently from TPU hardware.
  Mixed-precision correctness should be validated on real hardware.

This suite is a **necessary but not sufficient** gate. Passing it means the
logic is right; a short v4-8 run with `tensor_parallel_size=4` confirms the
same logic stays correct on real hardware.

---

## Mesh Configurations and What They Test

We use 8 simulated devices. Three mesh configurations exercise distinct
communication patterns:

| Mesh | Axes | What It Tests |
|------|------|---------------|
| `{data:8, model:1}` | Pure data parallel | Gradient all-reduce. No weight sharding. Dion update is entirely local. Baseline: should match single-device exactly. |
| `{data:2, model:4}` | TP=4 | Weight `[Out, In]` is sharded along In across 4 devices. `P = M @ V` contracts over the sharded dimension → XLA inserts all-reduce of `[Out, r]`. Column norm of W requires global reduction. This is the primary sharding test. |
| `{data:4, model:2}` | TP=2 | Same mechanics as TP=4 but with different shard sizes. Confirms correctness isn't accidentally tied to a specific divisor. |

### Axis Mapping

In all mesh configs, the Levanter/Haliax axis mapping is:
- `"mlp"` → `"model"` (weight's In dimension shards on model axis)
- `"heads"` → `"model"`
- batch → `("data",)`

A `hax.nn.Linear(In=Axis("in", 128), Out=Axis("out", 64))` with its `In`
mapped to `"model"` will be sharded as `[64, 128/model_size]` per device.

### What "sharded matches unsharded" means concretely

For a weight `[Out=64, In=128]` with mesh `{data:2, model:4}`:
- Each device holds `M_local` of shape `[64, 32]` (128/4 columns)
- Each device holds `V` of shape `[128, r]` replicated (NOT sharded — per
  the implementation decision to keep V replicated for correct column norms)
- `P = M @ V` is a local matmul `[64, 32] @ [32, r]` producing partial
  `[64, r]`, then all-reduced across model axis to get the true `[64, r]`
- QR operates on the replicated `[64, r]`
- `W = M.T @ U` is local: `[32, 64] @ [64, r]` → `[32, r]` per device
- Column norm of W: since V is replicated, W is a *shard* of the full W.
  Normalization must use the global norm (sum of squares across all shards
  along the `In` axis, then sqrt). If V is replicated, then W is NOT sharded
  (each device computes its own portion of the product, but the contraction is
  over the Out axis which is replicated). **Wait** — let's be precise:
  `W = M.T @ U` where M.T is `[In_shard, Out]` and U is `[Out, r]` replicated.
  Result: `[In_shard, r]` — W is sharded along In. So ColNorm(W) DOES need a
  global norm across the model axis. This is Sharp Edge #4.

The "sharded matches unsharded" test gathers all sharded results back to a
single host and compares against the unsharded reference. If they match at
`atol=1e-6`, the collective insertions are correct.

---

## Limitations of Simulated Multi-Device Testing

1. **Collectives are memcpy, not real networking.** All-reduce on 4 simulated
   devices sums 4 buffers in shared memory. There's no possibility of packet
   loss, deadlock, or timeout. The test validates logical correctness of
   data flow, not robustness under real communication.

2. **No pipelining or overlap.** On real hardware, compute and communication
   overlap. XLA may schedule collectives differently when it knows ICI
   bandwidth. The simulated backend executes everything sequentially.

3. **No memory pressure.** Real TP=4 on a v4-8 means each chip has ~16GB HBM.
   Simulated devices share CPU RAM with no per-device limit. OOM bugs on
   real hardware won't surface here.

4. **XLA optimization passes differ.** CPU XLA may make different fusion/layout
   decisions than TPU XLA. A test passing on CPU doesn't guarantee the same
   compiled HLO on TPU. (In practice this rarely causes correctness bugs, only
   performance differences.)

5. **QR implementation differs.** As noted above, column signs can flip. The
   tests assert allclose on the *final update* (which is invariant to Q's sign
   convention because both U and V_new flip together), not on intermediate Q
   values.

---

## Environment Setup

The XLA flag MUST be set before JAX is imported. Set it at module level in the
test file (not in a fixture):

```python
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import haliax as hax
from haliax import Axis
from haliax.partitioning import axis_mapping, set_mesh
from haliax.nn import ArrayStacked
```

### Fixtures

```python
@pytest.fixture(params=[
    pytest.param({"data": 8, "model": 1}, id="dp8"),
    pytest.param({"data": 2, "model": 4}, id="dp2-tp4"),
    pytest.param({"data": 4, "model": 2}, id="dp4-tp2"),
])
def mesh_config(request):
    """Parametrized mesh for distributed tests."""
    cfg = request.param
    devices = np.array(jax.devices()).reshape(cfg["data"], cfg["model"])
    mesh = Mesh(devices, ("data", "model"))
    mapping = {"mlp": "model", "heads": "model"}
    return mesh, mapping


@pytest.fixture
def single_device_mesh():
    """Single device — no sharding reference."""
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devices, ("data", "model"))
```

### Test Model

All distributed tests use the same model shape for comparability:

```python
In = Axis("in", 128)   # divisible by 1, 2, 4
Out = Axis("out", 64)  # not sharded (rows stay local)

# For scan tests:
Layers = Axis("layers", 4)
```

---

## Section 1: Registration and Config

### test_dion_registered_as_optimizer_config

Verify `OptimizerConfig.get_subclass("dion")` returns `DionConfig`.

### test_dion_config_defaults

Verify default hyperparameters match the paper:
- `mu=0.95`, `rank_fraction=1.0`, `power_iters=1`
- `learning_rate=6e-4` (OptimizerConfig default; Dion users will override)
- `adam_lr=6e-4`, `beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`

### test_dion_config_build_returns_optimizer

`DionConfig(learning_rate=0.02).build(1000)` returns an object with `.init`
and `.update` callables.

### test_dion_lr_schedule_integration

Inherited from `OptimizerConfig`. Verify warmup + cosine decay produces the
expected schedule values at step 0, warmup end, and final step.

---

## Section 2: Mask Routing

### test_dion_mask_linear_weight_to_dion

`hax.nn.Linear` weight → `"dion"`.

### test_dion_mask_bias_to_adamw

`hax.nn.Linear` bias → `"adamw"`.

### test_dion_mask_embedding_to_adamw

Path containing `"Embedding"` → `"adamw"`.

### test_dion_mask_lm_head_to_adamw

Path containing `"lm_head"` → `"adamw"`.

### test_dion_mask_eqx_linear_to_adamw

`eqx.nn.Linear` (not haliax) → `"adamw"` for weight and bias. Only
`hax.nn.Linear` gets spectral treatment.

### test_dion_mask_normalization_to_adamw

Path containing `"LayerNorm"` or `"RMSNorm"` → `"adamw"`.

---

## Section 3: Core Algorithm (Single Device)

No mesh. Pure math validation.

### test_dion_single_step_nonzero

One step with random gradient → update is non-zero.

### test_dion_matches_reference_5_steps

Run 5 steps of the Dion update rule manually in numpy/jax:

```
for step in range(5):
    M += G[step]
    P = M @ V
    U, _ = jnp.linalg.qr(P)
    W = M.T @ U
    M = M - (1 - mu) * U @ W.T
    V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
    O = jnp.sqrt(m / n) * U @ V.T
```

Compare `O` and final `M`, `V` against the implementation's output at each
step. `atol=1e-5`. This is the ground-truth correctness test.

Both the reference and the implementation must use the same:
- Initial V (deterministic from fixed seed)
- Gradient sequence (from `jax.random.normal` with known key)
- float32 precision

### test_dion_update_orthonormality

After removing the `sqrt(out/in)` scaling, compute SVD of the update.
Assert all singular values are within `[0.9, 1.1]` for full rank.
(Not exactly 1.0 because Dion produces `U @ V_new.T` which is only
approximately orthonormal when V_new is column-normalized but not
orthogonalized.)

### test_dion_spectral_scaling

Parametrize over shapes `[(64, 32), (32, 64), (64, 64)]`.
Verify that `norm(update) / norm(update_without_scaling) ≈ sqrt(out/in)`.

### test_dion_state_shapes

Parametrize over `(shape, rank_fraction, expected_r)`:
- `([64, 128], 0.5, 32)`
- `([256, 256], 0.25, 64)`
- `([32, 128], 1.0, 32)`  # r = min(m,n)
- `([128, 32], 1.0, 32)`

Verify `V.shape == (n, expected_r)` and `M.shape == (m, n)`.

---

## Section 4: Error Feedback

### test_dion_momentum_bounded

200 steps, constant unit-norm random gradients, `mu=0.95`.
Assert `norm(M) < 100 * norm(G)` at all times.
(With 5% drain per step, steady state is bounded.)

### test_dion_mu_zero_drains_M_completely

`mu=0.0` (beta=1.0), full rank. After one step on gradient G:
`M_new = M + G - 1.0 * U @ W.T`. Since U@W.T is the best rank-r
approximation of M+G (and rank=min(m,n) = full), M_new ≈ 0.
Assert `norm(M_new) < 1e-5 * norm(G)`.

### test_dion_mu_one_accumulates_without_bound

`mu=1.0` (beta=0.0). No error feedback. After N steps of constant G:
`norm(M) ≈ N * norm(G)`. Assert linear growth.

### test_dion_error_feedback_preserves_low_rank_signal

Gradient is rank-1: `G = u @ v.T`. Dion with `rank_fraction` giving `r=1`.
Power iteration on a rank-1 M should capture it perfectly.
After 10 steps, `norm(M)` should stabilize (not grow), because error
feedback drains the captured component.

---

## Section 5: Edge Cases

### test_dion_zero_gradient_no_nan

`G = zeros`. Assert update is zeros. No NaN, no Inf in state.

### test_dion_tiny_gradient_no_nan

`G = 1e-30 * ones`. QR on near-zero P must not explode.
Assert no NaN in update or state.

### test_dion_shapes (parametrized)

Parametrize: `[(64, 64), (256, 32), (32, 256), (7, 13), (128, 1)]`
All shapes should produce valid updates without shape errors.
(Odd shapes catch off-by-one in rank computation.)

### test_dion_rank_one

`rank_fraction` set so `r=1`. Verify update is exactly rank-1 (only 1
non-zero singular value after removing scaling).

---

## Section 6: Distributed Equivalence (Simulated Devices)

These are the critical tests. They use the `mesh_config` fixture (parametrized
over dp8, dp2-tp4, dp4-tp2).

### test_dion_sharded_matches_unsharded_1step

**Setup:**
1. Create `hax.nn.Linear(In=128, Out=64)` with In mapped to `"model"`.
2. Fix a random gradient G of shape `[64, 128]`.
3. Run one Dion step on single device → `ref_update`, `ref_M`, `ref_V`.
4. Run one Dion step with the parametrized mesh → `sharded_update`, etc.
5. Gather sharded results to full tensors.

**Assert:** `allclose(ref_update, gathered_update, atol=1e-6)` for the update,
M, and V.

**What this catches:**
- `P = M @ V` all-reduce is inserted correctly (model axis reduction)
- QR on the all-reduced P produces the same U
- `W = M.T @ U` stays correctly sharded
- ColNorm(W) uses global norm (if it used local norm, results diverge)
- Error feedback has correct sharding

### test_dion_sharded_matches_unsharded_10steps

Same structure, 10 steps with distinct gradients. Assert no drift:
`atol=1e-5` (allowing tiny float32 accumulation differences from
reduction order).

### test_dion_sharded_matches_unsharded_low_rank

Same as 1-step test but with `rank_fraction=0.25`. V is smaller, W is
sharded differently. Must still match.

### test_dion_column_norm_is_global

**Targeted regression test for Sharp Edge #4.**

Construct a W matrix `[In=128, r=16]` with values that are large in one
shard and small in another. Shard it across model=4 (each gets 32 rows).

Compute `ColNorm(W_sharded)` under the mesh. Gather result.
Compute `ColNorm(W_full)` on a single device.
Assert they match at `atol=1e-7`.

If the implementation incorrectly uses local norms, this test will fail
because `norm([big_shard; small_shard]) ≠ norm(big_shard)`.

### test_dion_dp_only_gradient_averaging

Mesh `{data:8, model:1}`. Each "replica" gets a different random gradient.
After Dion step, all replicas should have identical updates (gradients are
all-reduced before the optimizer sees them — this tests that the pipeline
external to Dion works correctly with it).

---

## Section 7: Scan/Stacked Layers

### test_dion_init_array_stacked

Create `ArrayStacked(num_layers=4)` wrapping a module with a `[64, 32]`
weight. Initialize Dion state. Assert state contains per-layer M and V with
leading dimension 4.

### test_dion_step_array_stacked

Run one step on the stacked model. Assert output has leading dim 4 and
per-layer updates differ (each layer's gradient is different).

### test_dion_stacked_matches_per_layer

Run Dion on 4-layer stack. Also run Dion on 4 independent `[64, 32]` params
with the same per-layer gradients. Assert results match per-layer
(`atol=1e-6`). Validates that vmap doesn't introduce cross-layer coupling.

---

## Section 8: Multi-Transform Integration

### test_dion_both_paths_produce_updates

Model with `Linear(In=32, Out=64)` + an embedding array.
One Dion step. Assert both the linear weight and embedding get non-zero
updates. Assert the linear update looks orthonormal-ish and the embedding
update looks Adam-ish (different magnitudes/structure).

### test_dion_weight_decay_exclusions

`DionConfig(weight_decay=0.1)`. After one step:
- Linear weight: verify weight decay was applied (param shrank toward zero
  slightly beyond what the update alone would do)
- Bias: no weight decay
- Embedding: no weight decay
- Norm layer: no weight decay

### test_dion_separate_lr_paths

`DionConfig(learning_rate=0.02, adam_lr=3e-4)`. Run one step on a model with
both Dion and AdamW params. Verify the magnitude of the Dion update is
proportional to 0.02 and the AdamW update is proportional to 3e-4 (inspect
the injected hyperparameters or compare update norms).

---

## Section 9: Power Iteration Quality

### test_dion_more_iters_better_approximation

`rank_fraction=0.25`, weight `[128, 128]`, random M (not from training, just
filled with random values to avoid trivial low-rank structure).

Run power iteration with `power_iters=1` → error₁ = `||M - U₁ @ W₁.T||`
Run power iteration with `power_iters=3` → error₃ = `||M - U₃ @ W₃.T||`

Assert `error₃ < error₁`.

### test_dion_warmstart_improves_over_steps

Run 20 Dion steps. Measure approximation error `||M - U @ W.T|| / ||M||`
at step 1 vs step 20. Assert step-20 error is significantly lower (V has
converged to the dominant subspace).

---

## Section 10: Convergence Smoke Test

### test_dion_loss_decreases

Tiny model: 2 `hax.nn.Linear` layers `[16→32]` and `[32→16]`, random
regression task, MSE loss, fixed seed.

200 steps with `DionConfig(learning_rate=0.02, rank_fraction=1.0)`.
Assert `loss[199] < 0.5 * loss[0]`.

This is deterministic (fixed seed) so the bound can be tight.

### test_dion_low_rank_converges

Same setup, `rank_fraction=0.25`, 500 steps.
Assert `loss[499] < 0.8 * loss[0]`.

---

## Running

```bash
# Full suite (XLA flag is set in the test file):
uv run pytest lib/levanter/tests/test_dion.py -v

# Just distributed tests:
uv run pytest lib/levanter/tests/test_dion.py -v -k "sharded or column_norm or dp_only"

# Just algorithm correctness:
uv run pytest lib/levanter/tests/test_dion.py -v -k "reference or orthonormal or scaling"
```

---

## Acceptance Criteria

The suite passes. The three load-bearing assertions are:

1. **`test_dion_matches_reference_5_steps`** at `atol=1e-5` — proves the
   implementation reproduces the paper's algorithm.
2. **`test_dion_sharded_matches_unsharded_1step`** at `atol=1e-6` — proves
   sharding doesn't corrupt results.
3. **`test_dion_loss_decreases`** — proves the optimizer actually optimizes.

If any of these three fail, the implementation has a bug. The remaining tests
are defense-in-depth and regression coverage.
