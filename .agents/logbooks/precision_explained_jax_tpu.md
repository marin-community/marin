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

These are four different levers.

In particular:

- `p=f32,c=f32` does **not** automatically mean "true fp32 matmul" on TPU.
- TPU MXUs natively multiply low-precision operands and accumulate in fp32, but
  XLA can still choose different precision algorithms for the dot product.
- `jax_default_matmul_precision` and explicit `precision=` on
  `dot_general`/`einsum` control matmul precision, not `jmp`.
- None of those matmul controls automatically change the dtype used by a
  cross-chip `all-reduce`.

So Exp U in [debug_accum_tpu_type.md](./debug_accum_tpu_type.md) ruled out
"just changing the model compute dtype to f32 fixes it," but it did **not**
fully rule out all numeric-precision hypotheses. It did not directly test:

- higher-precision TPU dot algorithms such as `highest` or
  `BF16_BF16_F32_X6`
- an fp32 cast on the LoRA gradient **before** the cross-chip collective

The most likely current mechanism is still the width-4 bf16 collective path on
`v5p-8`, but that should be described as the best-supported hypothesis rather
than a fully closed proof.

## Why this logbook exists

[debug_accum_tpu_type.md](./debug_accum_tpu_type.md) mixes together several
different meanings of "precision":

- mixed-precision policy (`p=...`, `c=...`)
- local matmul precision on TPU
- fused-kernel internal accumulation
- cross-chip reduction precision

That made the interpretation of Exp U (`p=f32,c=f32`) too broad. This note
separates those concepts cleanly and maps them onto the actual Marin code path.

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

1. **Operand rounding still matters.**
   If XLA feeds bf16 operands into the MXU, then every multiply starts from
   bf16-rounded values.

2. **The dot product algorithm still matters.**
   XLA can choose a one-pass, three-pass, or six-pass implementation for
   higher-precision emulation. PyTorch/XLA exposes this as `default`, `high`,
   and `highest`. JAX exposes the same idea via `precision=` and
   `jax_default_matmul_precision`.

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

## Part 2: What Marin/JAX actually does in this stack

### Step 1: parameter storage dtype

The `jmp.Policy` controls how parameters are stored in the train state.

In Marin/Levanter, train-state init applies the mixed-precision policy here:

- [trainer_state.py](../../lib/levanter/src/levanter/trainer_state.py)

Specifically, `cast_params_by_trainability(model, mp, is_trainable)` is applied
before optimizer state is created.

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

### Step 4: LoRA uses the same path

LoRA linears in Marin are ordinary Haliax linears composed into adapters:

- [lora.py](../../lib/levanter/src/levanter/lora.py)

So the standard LoRA forward/backward path is also governed by:

- the `jmp` compute cast
- Haliax/JAX dot precision defaults

That is why a global matmul-precision setting is a meaningful lever for this
investigation.

## Part 3: The four separate precision knobs

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
- many backward tensors

It does **not** fully determine the TPU dot algorithm.

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

Controlled by the dtype of the values entering the collective and by the
lowering chosen by XLA/runtime for the collective op.

This affects:

- `psum`
- `all-reduce`
- `reduce-scatter`

This is the relevant knob for the current LoRA-gradient aggregation hypothesis.

Crucially:

- matmul precision knobs do **not** automatically change collective reduction
  dtype
- `preferred_element_type` on a dot does **not** change a later `all-reduce`

## Part 4: Interpreting the common configurations

### `p=f32,c=bfloat16`

This is the usual mixed-precision training setup in the current experiments.

Meaning:

- store master params in fp32
- cast model to bf16 for execution
- run linears with bf16 operands unless a callsite or JAX precision override
  changes the dot behavior

This is good for:

- memory
- bandwidth
- speed

But it still means the model compute path is feeding bf16 operands into the dot
pipeline.

### `p=f32,c=f32`

This is what Exp U changed to in
[debug_accum_tpu_type.md](./debug_accum_tpu_type.md).

Meaning:

- store master params in fp32
- execute the model in fp32

What it **does** test:

- whether changing the model's compute tensors from bf16 to fp32 is enough to
  fix the issue

What it does **not** fully test:

- whether TPU dot products need an explicit higher-precision algorithm setting
- whether the failing piece is actually the cross-chip collective, not the
  matmuls

This is the single most important correction to the old logbook.

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

> "Can we keep the memory/bandwidth profile of bf16 compute but make the local
> matmuls numerically stronger?"

What this tests:

- local matmul precision as a cause

What this does **not** test:

- the dtype of the cross-chip all-reduce

### `preferred_element_type=jnp.float32`

This is a weaker lever than many people assume.

OpenXLA documents `preferred_element_type` as a **recommendation** for
accumulation/output type, not a hard guarantee. It is useful, but it should not
be treated as the primary "force real fp32 math" mechanism.

## Part 5: How this changes the reading of `debug_accum_tpu_type.md`

### What still looks right

Several claims in [debug_accum_tpu_type.md](./debug_accum_tpu_type.md) remain
strong:

1. **The sharding story is real.**
   The LoRA `PartitionSpec`s match between `v5p-8` and `v6e-8`, with the main
   difference being `data`-axis width.

2. **The HLO shows bf16 collective reduction regions.**
   The relevant `psum` reduction regions are bf16 adders in the optimized HLO.

3. **The width-4 collective path remains the leading mechanism.**
   The mesh experiments and the HLO evidence together keep that hypothesis
   front and center.

### What should be weakened

The following claims in [debug_accum_tpu_type.md](./debug_accum_tpu_type.md)
should be softened.

#### "Exp U ruled out all numeric precision"

This is too strong.

Exp U ruled out:

- "just changing `jmp` compute dtype to f32 fixes it"

Exp U did **not** directly rule out:

- higher-precision TPU dot algorithms (`highest`, `BF16_BF16_F32_X6`, etc.)
- fp32 cast before the LoRA-gradient collective
- collective-internal precision as the load-bearing issue

So the right interpretation is:

> Exp U ruled out the simplest local-compute-dtype explanation, not every
> precision-related mechanism.

#### "Z2 tested fp32 collective precision"

That is also too strong.

`--xla_allow_excess_precision=false` is not a "force fp32 all-reduce" flag. It
does not directly turn a bf16 collective into an fp32 collective. The negative
result from Z2 is still useful, but it should not be described as a clean test
of fp32 collectives.

#### "Z4 nails the mechanism completely"

Z4 is strong evidence, but "nailed" is too strong if it means strict causal
closure.

Z4 establishes:

- the logical sharding layouts match
- the local shard sizes differ because `data=4` vs `data=8`
- the relevant collective reductions are bf16
- the corresponding LoRA-gradient all-reduce width is 4 vs 8

That is enough to make the width-4 bf16 collective path the best explanation.
It is not yet the same thing as a direct intervention proving that changing only
the collective dtype fixes the bad run.

## Part 6: The current best model of the failure

The current best-supported story is:

1. The bad run is not a generic `v5p-8` failure.
2. It is not explained by CE tiling.
3. It is not explained by the reference-model path.
4. It is not explained by changing only `jmp` compute dtype to fp32.
5. It is tightly linked to the `data=4` distributed path.
6. The relevant LoRA-gradient aggregation path uses bf16 collective reduction.

The most likely remaining mechanism is:

> a width-4 bf16 collective path on `v5p-8` perturbs the early LoRA update
> enough to send the run into the bad basin

That is a strong working hypothesis. It is not yet a fully closed proof.

## Part 7: What the next experiments should mean

If we want to close the gap cleanly, the next experiments should target the
remaining precision levers separately.

### Next test A: true high-precision local matmuls

Run the bad recipe with:

- `p=f32,c=f32`
- `JAX_DEFAULT_MATMUL_PRECISION=highest`

Interpretation:

- If this recovers, local dot precision still mattered.
- If this fails, the local-matmul theory becomes much less likely.

### Next test B: bf16 compute, stronger bf16 dot algorithm

Run the bad recipe with:

- `p=f32,c=bfloat16`
- `JAX_DEFAULT_MATMUL_PRECISION=BF16_BF16_F32_X6`

Interpretation:

- If this recovers, the problem may still be local dot numerics rather than the
  collective.
- If this fails, the collective hypothesis strengthens again.

### Next test C: cast LoRA grads to fp32 before the collective

Patch the LoRA-gradient reduction path so the values entering the `psum` /
all-reduce are fp32.

Interpretation:

- If this recovers, the causal story is effectively closed.
- If this still fails, the collective-width hypothesis needs to be revised.

This is the most direct remaining test.

## Practical takeaway for readers of `debug_accum_tpu_type.md`

When reading the existing logbook, make these substitutions mentally:

- "Exp U ruled out all numeric precision"
  ->
  "Exp U ruled out the simple compute-dtype explanation"

- "Z2 tested fp32 collective precision"
  ->
  "Z2 tested one XLA flag that did not alter the failure"

- "Z4 proves the mechanism"
  ->
  "Z4 makes the width-4 bf16 collective path the leading explanation"

That revised framing is more technically accurate and still preserves the main
story: the failure is most likely in the `data=4` LoRA-gradient aggregation
path, not in generic LoRA, generic DPO, or generic `v5p-8` compute.

## References

### Local code

- [trainer_state.py](../../lib/levanter/src/levanter/trainer_state.py)
- [trainer.py](../../lib/levanter/src/levanter/trainer.py)
- [haliax linear.py](../../lib/haliax/src/haliax/nn/linear.py)
- [haliax dot.py](../../lib/haliax/src/haliax/_src/dot.py)
- [lora.py](../../lib/levanter/src/levanter/lora.py)
- [debug_accum_tpu_type.md](./debug_accum_tpu_type.md)

### External docs

- [JAX `jax.default_matmul_precision`](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)
- [JAX `jax.lax` and `DotAlgorithmPreset`](https://docs.jax.dev/en/latest/jax.lax.html)
- [JAX TPU/Pallas precision notes](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [PyTorch/XLA precision tutorial](https://docs.pytorch.org/xla/release/r2.8/tutorials/precision_tutorial.html)
- [OpenXLA operation semantics for `Dot` / `DotGeneral`](https://openxla.org/xla/operation_semantics)
- [Cloud TPU bf16 guide](https://cloud.google.com/tpu/docs/bfloat16)
- [Cloud TPU system architecture](https://cloud.google.com/tpu/docs/system-architecture)
