# Precision Explained: JAX on TPU, from first principles

**Related investigation:** [debug_accum_tpu_type.md](./debug_accum_tpu_type.md)

## TL;DR

The current `v5p-8` LoRA DPO story in
[debug_accum_tpu_type.md](./debug_accum_tpu_type.md) is directionally right
about the sharding/collective path, but too aggressive about what the precision
experiments proved.

The important distinction is:

1. **Stored dtype** of model parameters
2. **Compute dtype** of activations and weights at the point of use
3. **Dot/matmul algorithm** chosen by JAX/XLA for TPU MXUs
4. **Collective reduction dtype** used by `psum` / `all-reduce`

These are four different levers. `jmp.Policy` touches levers 1 and 2;
`jax_default_matmul_precision` / explicit `precision=` touches lever 3; lever
4 is determined entirely by the dtype of the tensor when it crosses the
`shard_map` output boundary, because there is no "reduce a bf16 tensor in
fp32" escape hatch in XLA.

There is also one structural fact that the original logbook glossed over:
**Marin/Haliax FSDP does not contain any explicit `jax.lax.psum` calls.** The
cross-chip `all-reduce` on gradients is emitted by the compiler (GSPMD) at
the `shard_map` out-specs boundary. So "fixing lever 4" means "change the
dtype of the gradient tensor immediately before it crosses that boundary."

The consequence for the debugging story: **Exp U's failure is in real tension
with the pure bf16-collective-width-4 hypothesis**, because with
`jmp.Policy(p=f32, c=f32)` the gradient reaching the shard_map boundary
should already be f32, and therefore the emitted all-reduce should already
be f32. Reconciling that tension is what Test C (below) is for.

## Why this logbook exists

[debug_accum_tpu_type.md](./debug_accum_tpu_type.md) mixes together several
different meanings of "precision":

- mixed-precision policy (`p=...`, `c=...`)
- local matmul precision on TPU
- fused-kernel internal accumulation
- cross-chip reduction precision

That made the interpretation of Exp U (`p=f32,c=f32`) too broad and the
interpretation of Z4 too aggressive. This note separates those concepts
cleanly, maps them onto the actual Marin code path, and explains exactly
where the FSDP gradient all-reduce comes from in the IR.

## Part 1: TPU hardware, from the ground up

### What the TPU hardware natively likes to do

Cloud TPU documentation describes the Matrix Multiply Unit (MXU) as taking
low-precision inputs and doing accumulation in fp32:

- the MXU uses `bfloat16` inputs
- accumulation is performed in `float32`

This is the standard TPU fast path. The practical consequence is:

- **bf16 inputs do not imply bf16 accumulation**
- but **bf16 inputs still lose mantissa bits before multiplication**

That second point matters. If a tensor is cast from fp32 to bf16 before the
matmul, the accumulator cannot recover the information lost in the cast.

### Why fp32 accumulation is not the whole story

It is tempting to reason:

> "If TPU accumulates in fp32 anyway, then `c=bf16` should already be almost as
> good as fp32."

That is incomplete for two reasons.

1. **Operand rounding still matters.** If XLA feeds bf16 operands into the
   MXU, then every multiply starts from bf16-rounded values.

2. **The dot product algorithm still matters.** XLA can choose a one-pass,
   three-pass, or six-pass implementation for higher-precision emulation.
   PyTorch/XLA exposes this as `default`, `high`, and `highest`. JAX exposes
   the same idea via `precision=` and `jax_default_matmul_precision`.

So there are two different questions:

- What dtype are the operands?
- What algorithm is used to realize the dot product?

### Matmuls and collectives are different hardware paths

This investigation cares about both:

- **MXU matmuls**
- **cross-chip collectives** such as `psum` / `all-reduce`

These are different operations and should not be conflated.

Changing matmul precision does **not** automatically change collective
precision. A run can have:

- better local dot-product numerics
- but still a bf16 cross-chip reduction

That distinction is the core reason Exp U does not close the whole question.

## Part 2: What Marin/JAX actually does in local compute

### Step 1: parameter storage dtype

The `jmp.Policy` controls how parameters are stored in the train state.

In Marin/Levanter, train-state init applies the mixed-precision policy here:

- [trainer_state.py](../../lib/levanter/src/levanter/trainer_state.py)

Specifically, `cast_params_by_trainability(model, mp, is_trainable)` is
applied at `lib/levanter/src/levanter/trainer_state.py:211` before optimizer
state is created. No LoRA-specific dtype path — LoRA weights are trainable,
so they get `mp.cast_to_param()` the same as any other trainable parameter.

So:

- `p=f32` means master params are stored as fp32
- `p=bf16` would mean params are stored as bf16

### Step 2: compute dtype at model execution time

At train-step time, Levanter casts the model to compute dtype:

- [trainer.py](../../lib/levanter/src/levanter/trainer.py)

The key line is `model = self._mp.cast_to_compute(model)`.

So:

- `c=bfloat16` means the model executes with bf16 weights/activations
- `c=f32` means the model executes with fp32 weights/activations

This is what Exp U changed.

### Step 3: linears go through Haliax dot/einsum

Standard Haliax linears eventually call `inputs.dot(...)`:

- [linear.py](../../lib/haliax/src/haliax/nn/linear.py)

That path ends up in Haliax dot/einsum wrappers:

- [dot.py](../../lib/haliax/src/haliax/_src/dot.py)

Those wrappers forward two separate precision-related arguments into JAX:

- `precision=...`
- `preferred_element_type=...`

If they are left as `None`, JAX/XLA uses the default backend behavior.

### Step 4: LoRA uses the same path, with no dtype tricks

LoRA linears in Marin are ordinary Haliax linears composed into adapters:

- `lib/levanter/src/levanter/lora.py:193` — `LowRankLinear.__call__` is
  literally `dropout → lora_A → lora_B → * scale`. Both `lora_A` and
  `lora_B` are plain `hnn.Linear`. There is **no `astype`, no
  `preferred_element_type` override, no explicit `precision=` argument** in
  the LoRA forward path.

So the standard LoRA forward/backward path is governed by:

- the `jmp` compute cast
- Haliax/JAX dot precision defaults

And nothing else. In particular, there is no hidden LoRA-specific bf16
downcast that would subvert `jmp.Policy(c=f32)`.

This fact is what makes Exp U's failure scientifically confusing — see
Part 7.

## Part 3: How FSDP actually emits the cross-chip reduction

This section covers lever D (collective reduction dtype) concretely: what
collective, over what axis, emitted by what code, running at what dtype,
and how you actually control it.

### FSDP in one paragraph

FSDP ("fully-sharded data parallelism") shards each weight matrix across
the `data` mesh axis, so every chip stores `1/|data|` of every weight. Each
chip also sees a different microbatch — this is where the "data parallel"
part comes in.

**Forward, per layer L:**

1. `all-gather` W_L along `data`: every chip temporarily reconstructs the
   full weight matrix.
2. Each chip runs forward on its local microbatch.
3. Discard the gathered copy; keep only this chip's `1/|data|` shard of W_L.

**Backward, per layer L:**

1. `all-gather` W_L again.
2. Each chip computes `∂loss/∂W_L` — the **full** gradient matrix,
   evaluated on this chip's microbatch. So chip 0 has `∂L₀/∂W`, chip 1 has
   `∂L₁/∂W`, etc.
3. **Sum those `|data|` gradients across chips.** This is where the
   batch-averaged gradient comes from — you cannot update with only one
   chip's microbatch gradient.
4. Re-shard the result back to `1/|data|` slices matching the optimizer-
   state layout on each chip.

Steps 3 and 4 together are emitted in HLO as either an `AllReduce` followed
by a scatter, or a fused `ReduceScatter`, depending on XLA's scheduling.
Either way, the logical operation is: **sum of |data| microbatch gradients
on the `data` axis.**

### The "width-4 psum" in the debug logbook, decoded

`replica_groups={4}` on v5p-8 vs `replica_groups={8}` on v5p-16 / v6e-8 is
just the number of participants in this gradient-aggregation collective.
v5p-8 has `|data|=4` (4 chips worth of microbatches to aggregate); v5p-16
and v6e-8 have `|data|=8`.

So when debug_accum_tpu_type.md says "the bf16 collective runs at width 4
on v5p-8," it is saying: "the FSDP backward gradient-aggregation all-reduce
has 4 participants on v5p-8, and the reduction region is `bf16 + bf16 →
bf16`."

### Where the collective is actually emitted in the code

**Surprise: there is no explicit `jax.lax.psum` call in Marin/Haliax for
FSDP gradient aggregation.** The collective is inserted by the compiler.

The machinery:

- `lib/haliax/src/haliax/partitioning.py:590` — the `fsdp()` wrapper.
- `lib/haliax/src/haliax/partitioning.py:610` — `_fsdp_impl`, which is
  essentially one line:

  ```python
  return named_jit(
      fn,
      in_axis_resources=parameter_mapping,    # params sharded on `data`
      out_axis_resources=parameter_mapping,   # grads must end up sharded same way
      axis_resources=compute_mapping,         # activations use compute-axis sharding
  )
  ```

- `lib/haliax/src/haliax/partitioning.py:909` — the `shard_map` boundary
  inside `named_jit`.

The rule GSPMD applies: if an output's required sharding differs from its
materialized sharding inside the jit, XLA must insert a collective to
reshard. For FSDP backward:

- *Inside* the jit, the gradient tensor naturally materializes replicated
  across `data` and sharded across `model` (same layout as activations).
- *Outside* the jit, the gradient must match `parameter_mapping` (sharded
  across `data`, matching the param layout).
- Resolving "replicated-across-data → sharded-across-data with sum" is
  exactly an `AllReduce` on the `data` axis (or a `ReduceScatter` if XLA
  chooses to fuse scatter into the same op).

**You never wrote `psum`; GSPMD wrote it on your behalf because the in/out
shardings demand it.** This is why you won't find `jax.lax.psum` by
grepping — the collective exists only in the compiled HLO, not in the
Python source.

### Why "collective dtype = tensor dtype at the boundary" is a hard rule

In HLO, `AllReduce` is a single op parameterized by its operand type. Its
reduction function is a tiny subgraph that looks like:

```
reducer (lhs: bf16, rhs: bf16) -> bf16 {
  ROOT sum = bf16 add(lhs, rhs)
}
```

The operand type of the collective and the reduction-function types must
match the input tensor type.

**There is no XLA flag, JAX argument, or GSPMD knob that says "reduce this
bf16 tensor but accumulate in f32."** The concept doesn't exist at the op
level. `preferred_element_type` controls matmul accumulation, not
collective accumulation. `--xla_allow_excess_precision` controls whether
XLA may *insert* higher-precision intermediates, not whether existing ops
get promoted.

If you want the reducer to be `f32 + f32 → f32`, **the tensor entering the
collective must itself be f32**. The only mechanism for fixing lever D is
"cast before the boundary." There is no workaround, because there is no
separate dial.

### Why that means `p=f32, c=f32` should already give f32 collectives

If `jmp.Policy(p=f32, c=f32)`:

- Params stored f32 (Step 1 of Part 2).
- Activations, weights-in-use, and backward cotangents all f32 (Step 2).
- LoRA forward/backward produces f32 grads (Step 4 — no LoRA-specific
  override).
- Gradient tensor leaves the `shard_map` body at f32.
- GSPMD sees an f32 tensor crossing the out-specs boundary.
- The emitted `AllReduce` / `ReduceScatter` operates on f32 operands with
  an `f32 + f32 → f32` reducer.

That is what the code says should happen. Whether it actually happens in
Exp U's compiled HLO is a separate empirical question — see Part 7.

## Part 4: Why FSDP hits this bug but TP doesn't

A natural question, given that Exp W (`{replica:1, data:1, model:4}` pure-TP
mesh) recovers training on the exact bad v5p-8 LoRA recipe: if TP also
shards weights 4-ways on the same topology, why doesn't it see the same
bf16-width-4 collective pathology?

This section records the intuitive mental model that prompted the question,
what it gets right, where it over-claims, and the sharp distinction that
actually explains the asymmetry.

### The intuitive claim

The first-pass answer goes roughly like this:

> In tensor parallelism each chip forward-passes on the whole batch at
> once, so you don't need to reduce gradients. Each chip has all the
> layers but only 1/N of the weights (sharded by width), every chip sees
> the whole batch, and does forward and backward fully locally — no
> cross-chip accumulation happens. That's why TP sidesteps the bug.

### What this gets right

1. **"TP doesn't shard the batch" is correct.** Every chip sees the full
   batch. There is no microbatch split along the `model` axis.
2. **"Weight gradients don't need cross-chip reduction" is correct.** This
   is the actually load-bearing observation and the real source of the
   FSDP-vs-TP asymmetry on this bug.
3. **"Each chip has 1/N of each weight matrix" is correct**, though the
   sharding axis is best named specifically (attention heads for the
   attention projections, hidden dim for MLPs) rather than "by width."

### Where it over-claims

"No cross-chip accumulation happens" is too strong. Tensor parallelism
absolutely has cross-chip collectives; they are just in a different place.

The standard Megatron-style TP pattern:

- **Column-sharded matmul** (output dim sharded across chips):
  `Y_i = X @ W_i`. Each chip computes its column-slice of Y. No collective
  yet.
- **Row-sharded matmul** (input dim sharded; typically the very next
  layer): `Y_i = X_i @ W_i`. Each chip produces a *partial sum* of the
  full output, and an `AllReduce` along the `model` axis is required to
  combine them.

So TP pairs column-sharded → row-sharded matmuls, with an `AllReduce` on
activations after every row-sharded layer and a corresponding `AllReduce`
on activation gradients in backward. Both run in bf16 on the default
recipe, both operate at `replica_groups={4}` on v5p-8 with `|model|=4`.
Same width, same dtype, same non-associativity as FSDP's gradient
reduction.

The right question is therefore not "does TP have collectives?" but "why
don't TP's collectives trigger the same bug?"

### The sharp distinction: what gets reduced

| | **TP** | **FSDP** |
|---|---|---|
| Sharding axis on weights | feature axis (heads, hidden dim) | data axis |
| Replicated across the sharded axis | batch | weights (temporarily, after all-gather) |
| What needs cross-chip reduction | **activations** (and their grads) | **weight gradients** |
| Reason | row-sharded matmul produces partial sums of the full output | different microbatches produce different copies of the full gradient matrix |
| Site of collective | middle of forward/backward, per layer | end of backward, one per weight tensor |
| Size of reduced tensor | `B × S × hidden` (millions of elements) | weight-matrix shard (LoRA: rank×dim, tiny) |
| Frequency per step | many (per layer, per direction) | one per weight tensor |

The mechanical reason weight gradients stay local in TP:

- Chip `i` holds `W_i`, a slice of `W`.
- Backward computes `∂L/∂W_i = X^T @ ∂L/∂Y_i` from locally-available
  tensors; no cross-chip contributions are needed.
- Different chips end up with *different shards* of the same full
  gradient — not different copies of the same shard. There is nothing to
  sum across chips.

Contrast FSDP: after all-gathering `W` for backward, each chip computes
the **full** `∂L/∂W` matrix on *its microbatch*. Every chip's copy is the
full shape of `W` but microbatch-specific. To get the batch-averaged
gradient you must sum across the `|data|` participants — that is the
load-bearing all-reduce.

### Why TP's activation all-reduces don't trigger the same pathology

TP's activation all-reduces also run in bf16 at width 4 on v5p-8. They
are subject to the same non-associative-addition phenomenon. Why no
crash?

Three compounding reasons:

1. **Signal-to-noise.** An activation all-reduce combines
   `B × S × hidden` values — on the order of millions of elements per
   reduction. The signal (the actual activation sum) massively dominates
   any bf16 rounding noise. An FSDP LoRA-gradient all-reduce combines
   values at the scale of a weight shard, and the step-0 LoRA gradient
   is *tiny* (rank-r projection through zero-init-B).

2. **Frequency and averaging.** Activation all-reduces happen many
   times per step (per layer, per direction). Rounding errors tend to
   average out over long chains. The FSDP weight-gradient all-reduce
   happens once per weight tensor per step — no averaging across sites.

3. **The LoRA zero-init-B trap.** This is the critical piece at step 0:
   - `∂L/∂A = 0` because `B = 0` zeros the product.
   - `∂L/∂B` is the only non-zero gradient and lives in a **rank-r
     subspace** (the projection through A).
   - The update direction is dominated by whatever signal makes it
     through that rank-r bottleneck.
   - A bf16-width-4 non-associative sum introduces a direction bias on
     the order of 1e-5 absolute (per the Z1 measurements).
   - That bias gets projected into the rank-r subspace and becomes the
     *dominant* component of the first update, because the real signal
     through the rank-r bottleneck is also tiny.
   - The model takes step 1 in a slightly wrong direction, lands in a
     basin where later gradients reinforce the wrongness, and never
     escapes.

   TP avoids this trap because LoRA's `A` and `B` are full (unsharded)
   on each chip when LoRA sits on the `model` axis — no weight-gradient
   collective touches them at all. The rank-r update is computed
   entirely locally.

### Connection to the experimental evidence

This framing cleanly predicts the Exp W / Z3 results:

- **Exp W** (pure TP, `{data:1, model:4}`) on the bad v5p-8 LoRA recipe:
  recovers. The FSDP weight-grad aggregation is absent; only
  activation-style collectives exist, and they are numerically robust.
- **Exp Z3** (mixed, `{data:2, model:2}`): recovers. The `data` axis
  shrinks from 4 to 2; at width 2 the reduction tree is a single add, so
  there is no non-associativity for that site to exploit.

Both results land exactly where the FSDP-vs-TP distinction predicts.

### Practical implication: the bug is narrow

The bug requires all four of:

- **LoRA** (not full-FT; Exp T full-FT works on the bad mesh)
- **`|data| = 4` specifically** (widths 2 and 8+ are both fine)
- **bf16 compute and bf16 collectives** (the default recipe)
- **the zero-init-B step-0 trap** (a LoRA-specific initialization
  artifact)

Users who want FSDP for memory reasons have several options without
waiting for the collective-dtype fix to be confirmed:

1. **Use a TPU where full FSDP lands on `|data| ≥ 8`** (v5p-16, v6e-8,
   v6e-16, anything larger). No code changes; canonical FSDP works.
2. **On v5p-8, use `{replica:1, data:2, model:2}` instead of full
   FSDP.** Retains 2-way data parallelism and 2-way tensor parallelism;
   per-chip memory footprint is equivalent to full FSDP on 4 chips
   (`1/4 = 1/2 × 1/2`). Z3 confirmed this mesh works. One-line config
   change.
3. **On v5p-8, use pure TP (`{replica:1, data:1, model:4}`).** Exp W
   confirmed this works. No data parallelism; forward passes on the
   full batch.
4. **Proper fix: cast LoRA gradients to f32 before the FSDP shard_map
   boundary.** Would restore full FSDP on v5p-8 LoRA. Contingent on
   Test C (Part 9) confirming that the width-4 mechanism really is
   bf16-collective-dtype.

FSDP is not broken. The bug is specific to this narrow configuration.

## Part 5: The four separate precision knobs

This is the cleanest way to think about the stack.

### A. Stored parameter dtype

Controlled by `p=...` in `jmp.Policy`.

Examples:

- `p=f32` stores master weights in fp32
- `p=bf16` stores them in bf16

This affects:

- parameter memory
- optimizer-state interactions
- checkpoint dtype

It does **not** by itself specify how matmuls or collectives are performed.

### B. Compute dtype

Controlled by `c=...` in `jmp.Policy`.

Examples:

- `c=bfloat16`
- `c=f32`

This affects the dtype seen by the model's compute graph after
`cast_to_compute`.

It changes:

- activation dtype
- the dtype of weights as used in linears
- many backward tensors — including, if unbroken, the gradient that reaches
  the `shard_map` boundary

It does **not** fully determine the TPU dot algorithm (lever C), but it
*does* determine the tensor dtype at the FSDP collective boundary (lever D),
*provided* no downstream kernel / cast narrows the gradient back to bf16.

### C. Dot/matmul precision algorithm

Controlled by:

- explicit `precision=` on `dot_general` / `einsum`
- or the global `jax_default_matmul_precision`

This is the JAX/XLA analog of PyTorch/XLA's `default` / `high` / `highest`.

Important examples:

- `default`
- `high`
- `highest`
- `BF16_BF16_F32`
- `BF16_BF16_F32_X3`
- `BF16_BF16_F32_X6`

These specify how much extra work XLA should do to emulate more accurate dot
products on TPU.

This is **separate** from `jmp`.

### D. Collective reduction dtype

Controlled by the dtype of the values entering the collective, and by
nothing else (see Part 3 for why).

This affects:

- `psum`
- `all-reduce`
- `reduce-scatter`

In Marin/Haliax FSDP, the collective for gradient aggregation is emitted
implicitly by GSPMD at the `shard_map` out-specs boundary in
`lib/haliax/src/haliax/partitioning.py:909`. To change its dtype you cast
the tensor immediately before it crosses that boundary.

Crucially:

- matmul precision knobs do **not** automatically change collective
  reduction dtype
- `preferred_element_type` on a dot does **not** change a later
  `all-reduce`
- `--xla_allow_excess_precision` does **not** promote an existing bf16
  reduction to f32

## Part 6: Interpreting the common configurations

### `p=f32,c=bfloat16`

This is the usual mixed-precision training setup in the current experiments.

Meaning:

- store master params in fp32
- cast model to bf16 for execution
- run linears with bf16 operands unless a callsite or JAX precision override
  changes the dot behavior

This is good for memory, bandwidth, and speed. But every tensor in the
compute graph, including the gradient that reaches the FSDP shard_map
boundary, is bf16. So the emitted all-reduce runs at bf16 — consistent with
the Z4 HLO evidence.

### `p=f32,c=f32`

This is what Exp U changed to in
[debug_accum_tpu_type.md](./debug_accum_tpu_type.md).

Meaning:

- store master params in fp32
- execute the model in fp32

**What this should change (per Part 3):**

- every forward/backward tensor in the graph is f32
- including the gradient crossing the shard_map boundary
- so the FSDP all-reduce's reduction region should be `f32 + f32 → f32`

**What it actually achieved in Exp U: unknown without HLO dump.** If
someone recompiles Exp U's recipe and dumps the HLO, we can check directly
whether its reduction regions are bf16 or f32. Until then, the
interpretation of Exp U's negative result is ambiguous (see Part 7).

### `p=f32,c=f32 + JAX_DEFAULT_MATMUL_PRECISION=highest`

Meaning:

- model executes in fp32
- JAX asks TPU to use the strongest float32-style default dot algorithm

This is the cleanest test of:

> "Could the problem still be in local matmul numerics?"

If this still fails, the "matmuls are the culprit" story gets much weaker.

### `p=f32,c=bfloat16 + JAX_DEFAULT_MATMUL_PRECISION=BF16_BF16_F32_X6`

Meaning:

- store params in fp32
- cast model to bf16 for compute
- keep bf16 operands
- but ask JAX/XLA to use a more accurate bf16 dot algorithm with fp32
  accumulation and a six-operation emulation path

This is the cleanest test of:

> "Can we keep the memory/bandwidth profile of bf16 compute but make the
> local matmuls numerically stronger?"

What this tests:

- local matmul precision as a cause

What this does **not** test:

- the dtype of the cross-chip all-reduce

### `preferred_element_type=jnp.float32`

This is a weaker lever than many people assume.

OpenXLA documents `preferred_element_type` as a **recommendation** for
accumulation/output type of a matmul, not a hard guarantee. It has no
effect on downstream collectives.

## Part 7: How this changes the reading of `debug_accum_tpu_type.md`

### What still looks right

Several claims in [debug_accum_tpu_type.md](./debug_accum_tpu_type.md) remain
strong:

1. **The sharding story is real.** The LoRA `PartitionSpec`s match between
   `v5p-8` and `v6e-8`, with the main difference being `data`-axis width.

2. **On the default `c=bf16` recipe, the HLO shows bf16 collective
   reduction regions.** This is consistent with Part 3: the gradient
   reaches the FSDP boundary as bf16, so GSPMD emits a bf16 `AllReduce`.

3. **The `|data|=4` path is the locus of the pathology.** Z3's mesh-
   rearrangement result is strong evidence that the problem sits
   specifically in the 4-participant collective path, not in bf16 generally
   or FSDP generally.

### What should be weakened

#### "Exp U ruled out all numeric precision"

This is too strong, but the revised reading is sharper than the earlier
version of this note suggested.

**The naive reading of Exp U:** set `c=f32`. Gradients are therefore f32
(Parts 2 and 5). The FSDP all-reduce's operand dtype is therefore f32
(Part 3). The collective reduction is therefore `f32 + f32 → f32`. If the
bug were "bf16-collective-width-4," Exp U should fix it.

**Exp U did not fix it.**

Taken at face value, that is *evidence against* the pure bf16-collective-
width-4 hypothesis, not just a gap in what was tested.

To salvage the collective-dtype hypothesis against this evidence, one of
the following must be true in Exp U:

1. **A Pallas kernel leaks bf16.** `jmp` does not reach into Pallas; the CE
   kernel at `lib/levanter/src/levanter/ops/xla.py` has its own block-size
   / dtype knobs. If the CE backward emits a bf16 cotangent, it would be
   upcast to f32 on entry to the f32 model graph — but only after some
   bf16 intermediate reduction inside the kernel.
2. **A separate optimizer-side reduction is bf16.** Distinct from the FSDP
   aggregation.
3. **Remat recomputation caches a narrower dtype.** Gradient checkpointing
   reruns forward during backward; if the rematerialized tensors are
   cached at a narrower dtype than the live graph, there is a hidden
   narrowing.
4. **`c=f32` changed enough else that the comparison is confounded.** The
   compute graph under `c=f32` differs materially from `c=bf16` in timing,
   scheduling, and memory footprint; Exp U therefore also tests "does
   anything about the compute-dtype change (not just collective dtype)
   perturb the run?" That is a weaker but nonzero confound.

Without an HLO diff on Exp U, we can't distinguish these from "the
collective really was f32 in Exp U, and collective dtype is not the
mechanism." That distinction is what Test C (below) answers directly.

#### "Z2 tested fp32 collective precision"

Still too strong. `--xla_allow_excess_precision=false` does not promote
bf16 reductions to f32; it only prevents XLA from *inserting* extra-
precision intermediates. The negative result from Z2 is useful but does
not constitute a test of fp32 collectives.

#### "Z4 nails the mechanism completely"

Z4 is strong correlative evidence but not a direct intervention. It
establishes:

- the logical sharding layouts match between v5p-8 and v6e-8
- the local shard sizes differ because `|data|=4` vs `|data|=8`
- the relevant collective reductions are bf16 (on the default `c=bf16`
  recipe)
- the corresponding LoRA-gradient all-reduce width is 4 vs 8

That is enough to make width-4 bf16 collectives the best-supported
explanation **as long as you assume** Exp U's f32 recipe actually produced
f32 collectives. Without that assumption, Z4 is consistent with both
"bf16 reduction is the mechanism" and "width-4 aggregation is the
mechanism for reasons independent of dtype" (tree topology, buffer layout,
scheduling, etc.).

## Part 8: The current best model of the failure

The current best-supported story is:

1. The bad run is not a generic `v5p-8` failure (Exp T full-FT works).
2. It is not explained by CE tiling (Exp R / R2a closed this).
3. It is not explained by the reference-model path (Exp V).
4. It is not explained by changing only `jmp` compute dtype to fp32 (Exp U)
   — but the interpretation of this is ambiguous (see Part 7).
5. It is tightly linked to the `|data|=4` distributed path (Exp W / Z3).
6. The FSDP gradient all-reduce on the default `c=bf16` recipe runs
   bf16-to-bf16 (Z4).

The most likely remaining mechanism is:

> a width-4 bf16 collective path on `v5p-8` perturbs the early LoRA update
> enough to send the run into the bad basin

That is a strong working hypothesis. It has one awkward piece of
counterevidence (Exp U) that needs to be resolved by a direct test at the
collective site, not by further compute-dtype sweeps.

## Part 9: What the next experiments should mean

Ordered by information-per-unit-effort, given the current state.

### Priority 1: Test C — cast LoRA grads to f32 at the FSDP boundary

Surgical version of lever D.

**What to change.** Inside `_fsdp_impl` at
`lib/haliax/src/haliax/partitioning.py:610`, or inside the `named_jit`
body's `shard_map` boundary at `:909`, cast LoRA gradient leaves to f32
immediately before they cross the output boundary. Optionally cast back to
bf16 afterward to preserve the rest of the memory/bandwidth profile.

Schematic:

```python
# inside the shard_map body, just before return
grads = jax.tree.map(
    lambda g: g.astype(jnp.float32) if _is_lora_grad(g) else g,
    grads,
)
return grads
# GSPMD now lowers the out_specs boundary to an f32 AllReduce
```

**Expected HLO signature after compilation.** LoRA gradient reduction
regions become `f32 + f32 → f32`. All other reductions stay bf16. This
can be verified with the same HLO-dump tooling used for Z4.

**Interpretation.**

- *Recovers training:* bf16-width-4 collective confirmed at the exact site
  of interest. Z4's correlative evidence becomes causally closed. Exp U's
  failure is explained by one of the hidden-bf16 candidates in Part 7 (or
  by compute-graph confounds independent of collective dtype).
- *Does not recover:* the width-4 story is not about dtype. Move the
  investigation to other `|data|=4`-specific differences: HLO scheduling,
  buffer layout, collective tree topology beyond dtype, all-gather vs
  reduce-scatter fusion choices.

This is the single highest-information remaining test. It should run before
Tests A, B, and the HLO-only check below.

### Priority 2: HLO dump of Exp U (no new TPU hours)

Recompile the Exp U recipe (`p=f32, c=f32`) on the bad `v5p-8` FSDP mesh
and dump the optimized HLO. Inspect the reduction regions of the LoRA
gradient `AllReduce`s.

**Interpretation.**

- *Reductions are f32:* Exp U really did produce f32 collectives, and yet
  training still went into the bad basin. This is strong evidence against
  the pure-collective-dtype mechanism and makes Test C's "does not recover"
  outcome more likely.
- *Reductions are bf16:* something downstream of `jmp c=f32` still
  narrows to bf16 before the collective (Pallas CE kernel, optimizer path,
  remat, etc.). Find and fix it; then Test C becomes a direct confirmation
  of the bf16-collective mechanism.

This is essentially free (CPU compile; no TPU time). Run it in parallel
with Test C to disambiguate faster.

### Priority 3: Test A — `p=f32, c=f32, JAX_DEFAULT_MATMUL_PRECISION=highest`

Tests lever C on top of lever B. A priori unlikely to be load-bearing: the
LoRA forward is a shallow chain (two linears plus a scale), so local
matmul-precision effects should be dominated by the weight-matrix all-
reduce.

Run if Test C and the HLO dump together still leave matmul precision
plausible; skip otherwise.

### Priority 4: Test B — `c=bf16 + BF16_BF16_F32_X6`

Tests lever C with bf16 compute. Useful mostly as a cheap sanity check if
Test A is inconclusive.

### What to skip

- Further `jmp c=f32` sweeps without HLO inspection. They do not
  distinguish between "collective is f32 but bug persists" and "something
  kept the collective at bf16." Pair any future `c=f32` run with an HLO
  check.
- Broad "just rerun with different knobs" experiments. Every further test
  should target a specific lever and be paired with a direct HLO or
  intermediate-tensor check that confirms the lever actually moved.

## Practical takeaway for readers of `debug_accum_tpu_type.md`

When reading the existing logbook, make these substitutions mentally:

- "Exp U ruled out all numeric precision"
  →
  "Exp U ruled out the simple compute-dtype explanation, and — if its HLO
  reductions actually went to f32 — provides real counterevidence to the
  pure bf16-collective story. HLO dump pending to disambiguate."

- "Z2 tested fp32 collective precision"
  →
  "Z2 tested one XLA flag that does not alter collective dtype."

- "Z4 proves the mechanism"
  →
  "Z4 makes width-4 bf16 collectives the leading explanation on the
  default `c=bf16` recipe. Direct causal closure requires Test C."

- "Just set `psum` to fp32"
  →
  "There is no such knob. Cast the tensor to f32 before it crosses the
  shard_map out-specs boundary at `partitioning.py:909`; GSPMD will then
  emit an f32 reduction automatically."

That revised framing preserves the main story — the failure is most likely
in the `data=4` LoRA-gradient aggregation path, not in generic LoRA,
generic DPO, or generic `v5p-8` compute — while honestly flagging the one
experiment (Exp U) whose result does not fit cleanly and pointing at the
single intervention (Test C) that would resolve it.

## References

### Local code

- [trainer_state.py](../../lib/levanter/src/levanter/trainer_state.py)
  (`cast_params_by_trainability` at `:211`)
- [trainer.py](../../lib/levanter/src/levanter/trainer.py)
  (`cast_to_compute` call)
- [haliax linear.py](../../lib/haliax/src/haliax/nn/linear.py)
- [haliax dot.py](../../lib/haliax/src/haliax/_src/dot.py)
- [haliax partitioning.py](../../lib/haliax/src/haliax/partitioning.py)
  (`fsdp` at `:590`, `_fsdp_impl` at `:610`, `shard_map` boundary at
  `:909`)
- [lora.py](../../lib/levanter/src/levanter/lora.py)
  (`LowRankLinear.__call__` at `:193`)
- [debug_accum_tpu_type.md](./debug_accum_tpu_type.md)

### External docs

- [JAX `jax.default_matmul_precision`](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)
- [JAX `jax.lax` and `DotAlgorithmPreset`](https://docs.jax.dev/en/latest/jax.lax.html)
- [JAX TPU/Pallas precision notes](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [JAX `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
- [PyTorch/XLA precision tutorial](https://docs.pytorch.org/xla/release/r2.8/tutorials/precision_tutorial.html)
- [OpenXLA operation semantics for `Dot` / `DotGeneral`](https://openxla.org/xla/operation_semantics)
- [OpenXLA operation semantics for `AllReduce`](https://openxla.org/xla/operation_semantics#allreduce)
- [Cloud TPU bf16 guide](https://cloud.google.com/tpu/docs/bfloat16)
- [Cloud TPU system architecture](https://cloud.google.com/tpu/docs/system-architecture)
