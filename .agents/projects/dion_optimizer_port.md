# Dion Optimizer Port to JAX/Levanter

## Overview

Port the Dion optimizer (Distributed Orthonormalized Updates) from PyTorch to
JAX as a first-class Levanter `OptimizerConfig`. Dion replaces Muon's
Newton-Schulz iteration with amortized power iteration on a momentum buffer,
enabling low-rank approximations that scale efficiently under weight sharding.

Reference materials:
- Paper: `/Users/varun/Documents/literature/Transformer-training-instabilities/Dion/main_V3.tex`
- PyTorch reference impl (simple): `…/Dion/dion/dion/dion_simple.py`
- PyTorch reference impl (distributed): `…/Dion/dion/dion/dion_reference.py`
- Existing Muon in Levanter: `lib/levanter/src/levanter/optim/muon.py`

---

## Files to Create

| File | Purpose |
|------|---------|
| `lib/levanter/src/levanter/optim/dion.py` | `DionConfig` + `scale_with_dion` optax transform |
| `lib/levanter/tests/test_dion.py` | Unit and integration tests |

## Files to Modify

| File | Change |
|------|--------|
| `lib/levanter/src/levanter/optim/__init__.py` | Add `DionConfig`, `ScaleByDionState` to `__all__` and imports |

No changes to `experiments/defaults.py` or `SimpleTrainConfig`. Users pass
`optimizer_config=DionConfig(...)` to `SimpleTrainConfig`; the existing
`optimizer_config` escape-hatch handles it.

---

## Architecture Decisions

### 1. Registration and Class Hierarchy

```python
@OptimizerConfig.register_subclass("dion")
@dataclass(frozen=True)
class DionConfig(OptimizerConfig):
    ...
```

Follows the exact pattern of `MuonConfig`, `NamoConfig`, `ScionConfig`.
Inherits LR scheduling, weight decay masking, warmup/decay/cycles from
`OptimizerConfig` base class.

### 2. Multi-Transform Routing (Matrix vs Scalar Parameters)

Dion applies orthonormalized updates **only to 2D matrix weights**. All other
parameters (embeddings, biases, normalization layers, lm_head) get a fallback
optimizer (AdamW by default, following the paper's recommendation and the Muon
precedent).

Implement a `create_mask(params)` method identical in structure to
`MuonConfig.create_mask`:

- `hax.nn.Linear` with 2D weight → label `"dion"`
- Embeddings, lm_head, biases, norms → label `"adamw"`

The `build()` method returns `optax.multi_transform({"dion": ..., "adamw": ...}, create_mask)`.

Use `label_linear_like_module(param, weight_label="dion", bias_label="adamw")`
from `levanter.optim.util` for consistent labeling.

### 3. The Dion Gradient Transformation

Create `scale_with_dion(...)` returning an `optax.GradientTransformation`. This
is the core algorithm.

**State** (`ScaleByDionState`, a `NamedTuple`):
- `momentum`: same pytree shape as params (full `[m, n]` per matrix weight)
- `right_vectors`: pytree of `[n, r]` matrices (one per matrix weight param)

**Update function** (per-parameter, applied via `map_flattened_linear_layers`):

```
Given: gradient G [m, n], momentum M [m, n], right vectors V [n, r]

1. M += G                              # accumulate gradient
2. P = M @ V                           # [m, r] — project onto subspace
3. U = QR(P).Q                         # [m, r] — orthonormalize
4. W = M^T @ U                         # [n, r] — compute right factor
5. M_new = M - (1 - mu) * U @ W^T     # error feedback
6. V_new = ColNorm(W)                  # [n, r] — warm-start for next step
7. O = U @ V_new^T                     # [m, n] — orthonormal update
8. return sqrt(m/n) * O                # spectral scaling
```

### 4. Handling Haliax Linear Layers

Use the same `map_flattened_linear_layers` / `flatten_linear_layers` /
`unflatten_linear_layers` utilities that Muon uses. This:
- Flattens multi-axis Linear weights to 2D `[Out, In]`
- Applies the Dion transform
- Unflattens back to original shape

This also handles scan layers (Stacked) automatically via
`scan_aware_tree_map` (called internally by `map_flattened_linear_layers`).

### 5. State Initialization for Right Vectors (`V`)

The `V` state tensor has shape `[n, r]` where `r = rank_fraction * min(m, n)`.
Since params in Levanter are `hax.NamedArray` wrapped in `Linear` modules,
we need to handle initialization carefully:

- In `init_fn`, iterate over the parameter tree
- For each Linear weight with shape `[Out, In]` (after flattening): create
  `V` as a random normal `[In, r]` array, column-normalized
- Use `jax.random.split` with a fixed seed per-param for reproducibility
- The `V` tree must mirror the structure of the params tree (same pytree structure)

**Critical**: `V` lives in the *flattened* 2D space. The init function must
flatten linear layers, compute V shapes, then store them in matching structure.
Since `map_flattened_linear_layers` only provides a transform function (not
init), we need a separate init path that mirrors the flatten/unflatten logic.

Recommended approach: store `right_vectors` as a pytree matching the *full*
params structure, where non-Dion leaves are `None` (or zeros). This matches
how `momentum_buffer` works in the Muon implementation.

### 6. Rank Computation

```python
r = int(rank_fraction * min(m, n))
r = max(r, 1)  # at least rank 1
```

The `rank_fraction` is a config hyperparameter (default: 1.0 = full rank).
When `rank_fraction=1.0` and `r = min(m, n)`, Dion is mathematically
equivalent to Muon (power iteration converges to full SVD basis).

Optional: add `rank_multiple_of: int = 1` for alignment (useful if we ever
want to shard V along the rank axis). Not required for initial implementation.

### 7. QR Orthonormalization

Start with `jnp.linalg.qr(P)` which returns `(Q, R)` where `Q` is `[m, r]`
orthonormal. This is:
- Numerically stable
- Efficient for tall-skinny matrices (m >> r)
- XLA-compilable and auto-differentiable

The paper offers RCQR (Randomized Cholesky QR) for distributed settings where
`P` is row-sharded. Under JAX SPMD, `jnp.linalg.qr` on a sharded array will
trigger an all-gather. For the initial port this is acceptable — the matrix is
only `[m, r]` which is much smaller than the `[m, m]` intermediates Muon
creates. Future optimization: implement RCQR as a follow-up.

### 8. Sharding Behavior Under JAX SPMD

With typical Levanter sharding (weight `[Out, In]` sharded along `In` on
the model axis):

- `M` is sharded identically to the weight: `[Out, In_shard]`
- `V` is `[In, r]` — the `In` dimension matches the weight's sharded dim.
  Store V replicated OR sharded along In (both work; sharded is memory-efficient).
  With V sharded as `[In_shard, r]`:
  - `P = M @ V` → contraction over `In_shard` → produces partial `[Out, r]` → XLA inserts all-reduce
  - `U = QR(P)` → `[Out, r]` replicated (since Out is not sharded)
  - `W = M^T @ U` → `[In_shard, r]` (stays local, no communication)
  - Error feedback: `U @ W^T` → `[Out, In_shard]` (same sharding as M, local)

This means:
- One all-reduce of `[Out, r]` per power iteration step (vs Muon's all-reduce of `[Out, Out]`)
- One all-reduce of `[r]` scalars for column normalization

Let `with_sharding_constraint` handle the specifics. The key point: the
algorithm is **naturally shard-friendly** without explicit collective calls.

### 9. Weight Decay

Apply decoupled weight decay (`optax.add_decayed_weights`) the same way Adam
and Muon do. Use the inherited `build_weight_decay_mask()` from
`OptimizerConfig` for consistent norm/embedding exclusion.

Weight decay applies to the full parameter update chain, not inside the Dion
transform itself.

### 10. Learning Rate Scaling

The paper specifies per-parameter scaling (Table 1 in paper):

| Parameter type | Scaling |
|---------------|---------|
| Weight matrix `[out, in]` | `sqrt(out / in)` |
| Bias | 1 |
| Embedding | 1 |
| LM Head (treated as `[1, in]`) | `1 / sqrt(in)` |
| Normalization | 1 |

Since non-matrix params go through AdamW (which already has its own LR), only
the matrix weight scaling `sqrt(out/in)` needs to be applied inside
`scale_with_dion`. Apply it as the final scaling of the orthonormal update O,
before returning from the per-layer transform (identical to Muon's approach).

**Important difference from Muon**: Muon uses `sqrt(max(1, out/in))` while
Dion uses `sqrt(out/in)` unconditionally. When `out < in` this produces a
value less than 1. This is the paper's recommendation for spectral norm
matching.

### 11. Momentum Convention

The paper uses `beta = 0.05` for the error feedback coefficient, which is
`1 - mu` where `mu = 0.95`. The config should expose `mu` (momentum
coefficient, default 0.95) to match the paper's API and the PyTorch reference.

Unlike Muon which uses classic momentum (`M = mu*M + G`) with optional
Nesterov, Dion accumulates into M additively (`M += G`) and applies error
feedback as a decay-like mechanism. There is NO separate momentum coefficient
for the accumulation — the `mu` parameter controls ONLY the error feedback
strength.

### 12. Config Hyperparameters

```python
@OptimizerConfig.register_subclass("dion")
@dataclass(frozen=True)
class DionConfig(OptimizerConfig):
    # Dion-specific
    mu: float = 0.95                    # error feedback retention (paper's mu)
    rank_fraction: float = 1.0          # r/min(m,n), 1.0 = full rank
    power_iters: int = 1                # number of power iteration steps

    # Fallback optimizer (for non-matrix params)
    adam_lr: float = 6e-4               # separate LR for AdamW fallback
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Weight decay (shared or separate)
    adam_weight_decay: float | None = None  # None → use self.weight_decay
```

The inherited `learning_rate` field is the Dion base LR (paper default: 0.02,
same as Muon). The `adam_lr` is for the AdamW fallback on non-matrix params.

### 13. Edge Cases

**Initial step (V not yet meaningful):** On the first call, V is random
normal (column-normalized). Power iteration with a random V still produces a
valid rank-r subspace — it's just not warm-started yet. No special-casing
needed; the algorithm converges in subsequent steps via error feedback.

**All-zero gradients:** If M is all-zeros (e.g., frozen params), `P = M @ V =
0`, and `QR(0)` is undefined. Handle this: if `norm(P) < eps`, skip the
update and return zeros. This matches the PyTorch reference's
`fix_all_zero_or_nan` function. Implement without data-dependent branching
for JIT compatibility: multiply the output by `(norm(P) > eps)`.

**Non-square matrices:** Fully supported. `r = rank_fraction * min(m, n)`
handles both tall and wide matrices.

**Matrices smaller than rank:** If `min(m, n) <= rank`, the algorithm
operates at full rank. `r = min(r, m, n)` caps it.

**Scan layers / Stacked modules:** Handled automatically by
`map_flattened_linear_layers` which uses `scan_aware_tree_map` internally.
Each layer in the stack gets its own V state via vmap.

---

## Testing Plan

All tests in `lib/levanter/tests/test_dion.py`.

### Test 1: Config Registration

Verify `DionConfig` can be instantiated and that
`OptimizerConfig.get_subclass("dion")` returns it.

### Test 2: Mask Routing

Create a model with `hax.nn.Linear` layers plus embeddings and verify:
- Linear weights → `"dion"`
- Biases → `"adamw"`
- Embeddings → `"adamw"`
- eqx.nn.Linear (no haliax) → `"adamw"` fallback (same as Muon's behavior)

Follow the pattern in `test_optimizer_linear_like.py`.

### Test 3: Optimizer Build and Step

Instantiate `DionConfig(...).build(num_train_steps=1000)`, apply it to a
small model, verify it produces non-trivial updates and doesn't crash.
Verify that the optax optimizer state has the expected structure.

### Test 4: Orthonormality of Update

For a single Linear weight `[m, n]`, run one Dion step and verify:
- The output update direction's singular values are approximately 1
  (orthonormal in spectral sense)
- Specifically: `U @ V_new^T` should have singular values close to 1

### Test 5: Rank Reduction

With `rank_fraction=0.5` on a `[64, 64]` weight, verify:
- State V has shape `[64, 32]`
- The update still has shape `[64, 64]`
- Training converges (loss decreases over N steps on a toy problem)

### Test 6: Error Feedback Accumulation

Run multiple steps and verify that M doesn't grow unboundedly — the error
feedback mechanism `M -= (1-mu) * U @ W^T` should keep M bounded.

### Test 7: Equivalence to Reference at Full Rank

With `rank_fraction=1.0` and a single power iteration, run 5 steps of:
- Levanter Dion
- A manual numpy/jax reimplementation of the reference algorithm

Verify updates match to `atol=1e-5`.

### Test 8: Scan/Stacked Layers

Following `test_scan_stack_optimizers.py`, verify that `DionConfig` can
init and step on an `ArrayStacked` module without error.

### Test 9: LR Schedule Integration

Verify that `DionConfig` LR scheduling works (warmup, cosine decay) by
checking the injected hyperparams after build. Same pattern as
`test_optimizer_config.py`.

### Test 10: Weight Decay Mask

Verify that weight decay is only applied to the correct parameters (not
LayerNorm, not Embedding, not bias) using the inherited mask.

---

## Conventions to Follow

1. **Copyright header**: `# Copyright The Levanter Authors` + SPDX
2. **Frozen dataclass**: `@dataclass(frozen=True)` for config
3. **NamedTuple state**: `class ScaleByDionState(NamedTuple)` for optax state
4. **No local imports** except in the `create_mask` method if needed to avoid
   circular deps
5. **Use `haliax` naming**: Reference `hax.nn.Linear`, `hax.NamedArray`
6. **scan_aware_tree_map**: All tree transforms must be scan-aware
7. **No comments explaining WHAT** — only WHY if non-obvious
8. **Tests use pytest functions** (no classes), with fixtures if shared setup

---

## JAX-Specific Sharp Edges

These are places where JAX's programming model creates non-obvious
requirements that don't exist in the PyTorch reference implementation.

### Sharp Edge 1: Random V Initialization Without a PRNG Argument

Optax `init_fn(params)` receives no PRNG key. Dion needs V initialized as
random normal `[n, r]`. Solution: use a fixed-seed `jax.random.PRNGKey(0)` in
`init_fn` (same as Kron at `kron.py:479`), split deterministically per param
based on tree position. Column-normalize V immediately after init.

Do NOT initialize V as zeros — the first step's `P = M @ V = 0` makes QR
undefined.

### Sharp Edge 2: State Shape Mismatch vs `map_flattened_linear_layers`

`map_flattened_linear_layers` applies a function to each Linear layer in the
updates tree. Its API is `f(layer: Linear) -> Linear`. It does NOT provide a
mechanism to thread per-layer auxiliary state (like V) through the transform.

Muon avoids this because its only state (momentum buffer) has the same shape
as the weight. Dion's V has shape `[In, r]` ≠ `[Out, In]`.

**Solution:** Do NOT use `map_flattened_linear_layers` for the full Dion
update. Instead, write a custom tree traversal that:
1. Flattens linear layers in both updates AND state trees in parallel
2. Applies the per-layer Dion update (reading M and V, producing new M, V,
   and the update)
3. Unflattens everything back

This is the same approach SOAP takes — managing its own tree traversal rather
than using the Muon helper. The `flatten_linear_layers` and
`unflatten_linear_layers` utilities can still be used individually.

### Sharp Edge 3: `optax.MaskedNode` in Multi-Transform

When `optax.multi_transform` routes params, non-Dion params appear as
`optax.MaskedNode()` sentinels. `otu.tree_zeros_like` handles these correctly
for the momentum buffer. But V initialization must explicitly check for
MaskedNode and preserve it:

```python
right_vectors = jax.tree.map(
    lambda p: _init_v(p) if not isinstance(p, optax.MaskedNode) else p,
    params,
    is_leaf=lambda x: isinstance(x, optax.MaskedNode),
)
```

### Sharp Edge 4: Column Normalization Under Sharding

If V is stored sharded as `[In_shard, r]` (matching the weight's sharding),
then `jnp.linalg.norm(W, axis=0)` computes only the LOCAL shard's norm — it
does NOT all-reduce to get the global column norm. This silently produces
incorrect normalization.

**Solution for initial impl:** Store V replicated (not sharded along In).
The memory cost is `[In, r]` per layer, which for `rank_fraction=0.25` on a
4096-dim model is just 4096×1024×2 bytes = 8MB per layer — negligible vs the
sharded weight.

Future optimization: explicitly `jax.lax.psum` the squared norms if we want
to shard V.

### Sharp Edge 5: QR Under Vmap (Scan Layers)

`scan_aware_tree_map` vmaps over the Block axis for stacked transformer
layers. This means `jnp.linalg.qr` will be batched. This works correctly on
XLA but:
- All layers must have identical `[m, r]` shapes (true for transformers)
- Batched QR on TPU may be slower than batched matmuls due to sequential
  Householder reflections. For skinny matrices `[m, r]` where `r << m`, this
  is fast enough. Monitor this in profiling.

### Sharp Edge 6: No In-Place Mutation

All state updates must be returned as new values. The optax pattern:
```python
def update_fn(updates, state, params=None):
    M = state.momentum
    V = state.right_vectors
    # ... compute ...
    return new_updates, ScaleByDionState(momentum=M_new, right_vectors=V_new)
```

This is straightforward but means the error feedback update
(`M_new = M - (1-mu) * U @ W^T`) allocates a new buffer rather than mutating.
XLA's buffer donation should eliminate the extra allocation in practice.

### Sharp Edge 7: `rank_fraction` Must Be Static

Array shapes must be known at XLA compile time. `r = int(rank_fraction *
min(m, n))` is computed once in `init_fn` and baked into V's shape. Changing
`rank_fraction` after training starts requires recompilation and breaks
checkpoint loading. Document this as immutable.

---

## What Becomes EASIER in JAX vs PyTorch

1. **No explicit collectives needed.** The 1D-sharded and 2D-sharded
   algorithms in the paper (Algorithms 3-4) exist because PyTorch requires
   manual ReduceScatter/AllGather calls. In JAX SPMD, XLA inserts the optimal
   collectives automatically from sharding annotations. We implement the
   unsharded algorithm and let the compiler handle distribution.

2. **Scan layers are free.** PyTorch Dion must manually batch over layers.
   In Levanter, `scan_aware_tree_map` + vmap handles this automatically — each
   stacked layer gets its own independent V state.

3. **No `torch.compile` boundary.** The PyTorch impl wraps `dion_update` in
   `@torch.compile()` to fuse kernels. In JAX, the entire optimizer step is
   already inside `jit` — no annotation needed, XLA fuses everything.

4. **No DTensor complexity.** The PyTorch reference has 600+ lines of
   DTensor/DeviceMesh/ProcessGroup plumbing. In JAX, sharding is declarative
   via PartitionSpec and the mesh context — zero explicit tensor distribution
   code.

5. **Functional purity means no checkpoint sync.** The PyTorch Dion has a
   `synchronize_for_checkpoint()` method because decoupled momentum can
   diverge across DP ranks. In JAX SPMD, all ranks execute the same program
   with the same logical state — there is no divergence to synchronize.

---

## What NOT to Do

- Do NOT modify `experiments/defaults.py` or `SimpleTrainConfig`
- Do NOT implement distributed collectives manually (let XLA handle it)
- Do NOT implement RCQR / Cholesky QR in the initial port (standard QR is fine)
- Do NOT add a `compressed_all_reduce` mode (JAX SPMD handles gradient sync)
- Do NOT handle DTensor/ProcessGroup concepts (PyTorch-specific)
- Do NOT add mixed-precision dtype configs initially (use param dtype throughout)
- Do NOT special-case `is_transposed` — Levanter's `_out_first` convention
  on `hax.nn.Linear` already ensures weight shape is `[Out, In]` after
  `flatten_linear_layers`
- Do NOT touch Muon's code or refactor shared utilities unless strictly necessary

---

## Open Questions (Pre-Resolved)

**Q: Should V be a NamedArray?**
A: No. V lives in the optimizer state as a raw jax array `[n, r]`. It's not a
model parameter. The `r` axis has no semantic name in the model. Store as
plain `jax.Array` inside the flattened-layer structure.

**Q: How does V's pytree structure work with scan layers?**
A: `init_fn` receives the full params tree. Use `map_flattened_linear_layers`
(or a parallel init variant) to create V for each linear layer. For scan
layers, the vmap over the Block axis automatically gives each stacked layer
its own V. Store V inside the same tree structure as momentum.

**Q: What about `power_iters > 1`?**
A: The paper recommends 1 iteration with warm-starting. Support >1 via a
simple loop in the update function, but default to 1. Each additional
iteration repeats steps 2-3 (project + QR) using the updated U to refine W,
then re-orthonormalize. Match the PyTorch reference's loop structure.

**Q: `sqrt(out/in)` when out < in — is this correct?**
A: Yes. The paper explicitly uses `sqrt(|I|/|J|)` unconditionally (Algorithm
caption, line "return sqrt(|I|/|J|) * O"). This differs from Muon's
`sqrt(max(1, out/in))`. Follow the paper.

**Q: Is the fallback optimizer always AdamW?**
A: Default yes, matching the paper and PyTorch impl. The paper also mentions
Lion as an option. For simplicity, hardcode AdamW as fallback. A future
enhancement could make it configurable.
