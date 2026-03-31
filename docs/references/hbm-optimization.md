# Making Things Fit in HBM

This guide is a practical checklist for JAX/Levanter/Haliax training runs that are close to OOM.

The main knobs are:

1. Shard more.
2. Checkpoint and offload activations.
3. Offload optimizer/parameter state.
4. Use model parallelism where it actually helps.
5. Use nested (`sqrt(n)`) checkpointing for scanned stacks.
6. Reduce per-device batch (and sequence length if needed).

## 1) Shard More (Usually the First Lever)

If arrays are accidentally replicated instead of partitioned, HBM disappears fast.

Use explicit placement at boundaries:

- `hax.shard(...)` for Haliax `NamedArray` trees.
- `jax.device_put(...)` for explicit initial placement.
- `jax.sharding.reshard(...)` when you need to change sharding mid-pipeline.
- For LMs, explicitly shard output projection / vocab-axis tensors so logits are partitioned rather than replicated.

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

# Example: shard parameters across data/model axes instead of replicating.
param_sharding = NamedSharding(mesh, P("data", "model"))
params = jax.device_put(params, param_sharding)
```

For FSDP-style setups, confirm large parameter tensors are split across the data axis rather than replicated.
In classic Levanter/Haliax codepaths, this is usually handled for you, but custom tensors and custom losses may still need explicit resharding.

## 2) Activation Checkpointing and Activation Offloading

Checkpointing (rematerialization) trades compute for memory by saving fewer intermediates in forward and recomputing them in backward.

Activation offloading is a variant: selected activations are moved from device memory to pinned host memory after forward, then moved back before backward.

Conceptually, with JAX checkpoint policies you choose, per named intermediate, whether to:

- Save on device.
- Offload to host.
- Recompute.

In Haliax/Levanter scanned stacks, this is typically exposed via `gradient_checkpointing` policies (e.g. standard recompute, offload variants, nested variants).

References:

- [JAX: Gradient checkpointing (`jax.checkpoint` / `jax.remat`)](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [JAX Memories and Host Offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)

## 3) Explicit Offloading of Optimizer State (and Sometimes Params)

Optimizer state is often one of the largest memory consumers (especially Adam-family optimizers).

A common pattern is:

1. Keep optimizer state in pinned host memory between steps.
2. Bring it to device only for update math.
3. Return updated state back to host.

```python
import jax
import optax

from levanter.utils.jax_utils import move_tree_to_memory_kind

s_dev = params_sharding
s_host = params_sharding.with_memory_kind("pinned_host")
opt_state = move_tree_to_memory_kind(opt_state, memory_kind="pinned_host")

@jax.jit(donate_argnums=(0,), out_shardings=(s_dev, s_host))
def train_step(params, opt_state, batch):
    opt_state = jax.device_put(opt_state, s_dev)
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, jax.device_put(opt_state, s_host)
```

This usually buys substantial HBM headroom, at the cost of transfer bandwidth/latency.

Reference:

- [JAX Memories and Host Offloading (optimizer state + parameter offloading)](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)

## 4) Model Parallelism Can Beat "Max FSDP" in Some Regimes

Sometimes parameter tensors or activations are too large even with aggressive data-axis sharding.
In that case, giving devices to model/tensor parallel axes can reduce peak HBM even though it reduces FSDP degree.

Rule of thumb: sweep a small grid of mesh shapes (for example, more `data` vs more `model`) and compare:

- Peak HBM
- Step time
- Achievable global batch

The best throughput-at-memory-budget point is often not the "maximum data parallel" point.

## 5) Keep Layer Stacks Scanned (`Stacked` or `ArrayStacked`)

When your model has deep repeated blocks, use a scanned stack container instead of Python-unrolling layer calls.

- Use [`haliax.nn.Stacked`](https://github.com/marin-community/marin/blob/main/lib/haliax/docs/scan.md#stacked) when your stack uses named axes (`Axis` / `NamedArray`).
- Use [`haliax.nn.ArrayStacked`](https://github.com/marin-community/marin/blob/main/lib/haliax/docs/array-stacked.md) when your stack is array-native (no named axes).

Observed advantages of staying on a scanned stack path:

- Shorter compile time, especially in deep networks.
- Modestly improved peak MFU (about `+1%` to `+2%` absolute in observed runs).
- Sometimes significantly reduced peak HBM, particularly in deep networks.

For Grug-specific array-stacked variant wiring, see the reference branch:
https://github.com/marin-community/marin/tree/codex/array-stacked-grug-variant-pointer

## 6) `sqrt(n)` Checkpointing for Scanned Layer Stacks

For a stack length `N`, nested checkpointing chunks the work into blocks of size `B` and stores only block boundaries.

When `B ~= sqrt(N)`, memory for saved boundaries is `O(sqrt(N))` instead of `O(N)`, with recomputation overhead.

This is useful for deep scanned stacks where plain checkpointing/offloading still does not fit.
In Haliax scanned modules, nested checkpointing is available as a policy option.

## 7) Reduce Per-Device Batch (and Sequence Length)

If you are right at the limit:

- Reduce microbatch/per-device batch.
- If needed, reduce sequence length.
- Recover global batch with gradient accumulation.

These are the most direct and reliable HBM controls.

## 8) Buffer Donation (`donate_argnums`)

Donation lets JAX reuse input buffers for outputs at JIT boundaries, reducing peak live memory.

Reference:

- [JAX Buffer Donation](https://docs.jax.dev/en/latest/buffer_donation.html)

## 9) Optimizer Choice Matters for Memory

For equal parameter count, optimizer state memory can differ drastically.

- Adam-like methods keep multiple full-size state tensors.
- Memory-lean alternatives (where acceptable for your training regime) can materially reduce HBM pressure.

If you keep Adam-family optimizers, offloading their state is often the practical compromise.

## 10) Profile Memory Before and After Each Change

Use JAX memory profiling tools to confirm what changed:

- [JAX: Profiling device memory](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX: GPU memory allocation notes](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

Memory tuning is much faster when each knob change is measured, not guessed.

## 11) Avoid Giant Temporary Tensors

Large temporaries can dominate peak memory even when parameter state fits.

- Avoid materializing full-size intermediates when a fused/chunked computation exists.
- For language models, the full logits tensor (`batch x seq x vocab`) is often the worst offender.
- Use memory-efficient attention kernels/backends where available in your model stack.

## 12) Keep EMA and Other Replicas Off HBM

Extra full-parameter copies (for example EMA weights) can be expensive in HBM.

- Keep long-lived replicas in host memory when possible.
- Materialize them on-device only when needed (for eval/export windows).

## 13) Use Lower Precision Where Safe

HBM scales linearly with dtype size.

- Prefer BF16 activations/weights on hardware where it is standard.
- Be explicit about which states must remain FP32 (often optimizer moments), then offload those if needed.

## 14) Tune Eval Memory Separately from Train

Evaluation often has different memory pressure than training.

- Set eval batch size independently.
- Reduce concurrent eval tasks/checkpoints when needed.
- Keep eval from overlapping peak-memory parts of training if your pipeline allows it.
