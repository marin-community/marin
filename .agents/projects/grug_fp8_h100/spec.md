# Grug FP8 on H100 Spec

This spec pins the concrete surface for a one-month work trial. The config is Grug-local; the FP8 dot helper is Haliax-level so `experiments/grug/moe` and `levanter.grug` can both use it without violating dependency direction. The implementation surface is intentionally provisional because the JAX FP8 stack has multiple overlapping paths, and MaxText/XLA appear to have moved beyond relying only on old quantize/dequantize pattern matching.

## JAX FP8 landscape

There are seemingly three relevant ways to get FP8-like GEMMs inside the JAX ecosystem today. The first phase of this project is to decide which one Grug should use, based on HLO and profiler evidence on the target H100 stack.

### A. Delayed-scaling Q/DQ around `dot_general`

This is the older path used by Flax FP8 ops and mirrored by the current Haliax FP8 implementation. The model keeps FP32 scale and amax-history metadata, quantizes operands to FP8, dequantizes or rescales around `jax.lax.dot_general`, and relies on XLA recognizing the resulting pattern or on explicit dot precision presets such as `ANY_F8_ANY_F8_F32`.

Pros:

- Closest to the existing Haliax `Fp8DotGeneralOp` and `OverwriteWithGradient` state model.
- Matches the classic Transformer Engine delayed-scaling mental model: E4M3 forward activations/weights, E5M2 output gradients, FP32 accumulation, scale histories.
- Straightforward to adapt to Grug's raw `jnp.einsum` call sites.

Risks:

- This path may be older or partially deprecated in practice. It depends on compiler pattern recognition and can silently become "FP8-looking" code that does not hit the intended H100 FP8 kernels.
- The current Haliax/Flax-style helper must be audited: using `precision=DEFAULT` on the inner dot may not be enough if XLA now expects either a specific dot algorithm or a newer scaled-dot operation.
- It does not directly solve grouped/ragged MoE; the Pallas-Triton and XLA ragged-dot backends still need separate investigation.

### B. Qwix quantized dot/ragged-dot

This is the newer MaxText `fp8_full`-style path. MaxText delegates to Qwix `dot_general_qt` for ordinary projections and Qwix also has a `ragged_dot_qt` wrapper. Qwix quantizes operands to `QArray`, calls low-precision `jax.lax.dot_general` or `jax.lax.ragged_dot_general`, and applies scales around the result.

Pros:

- Actively used by modern MaxText quantization paths, including a route that covers both dense dot-general and ragged grouped matmul.
- Its config shape already expresses E4M3FN activations/weights and E5M2 gradients, close to the target training recipe.
- It may reduce custom Marin/Haliax kernel surface if its ragged-dot route maps well to Grug expert shapes.
- Adopting Qwix is on the table if the dependency is not too heavy and its abstractions are useful for both dense and ragged Grug call sites.

Risks:

- It is a library-level quantization framework, not the existing Haliax state model. Grug would need either a thin adapter or a deliberate dependency choice.
- Qwix's dense path does not appear to call `jax.nn.scaled_dot_general`; it still relies on low-precision `jax.lax.dot_general` plus explicit scale handling.
- Its ragged-dot behavior must be verified on H100 for Grug's `[tokens, hidden] x [experts, hidden, ff]` and down-projection shapes, including backward pass and expert-parallel layouts.

### C. JAX `scaled_dot_general` / XLA `scaled-dot`

This is the newer explicit scaled-dot path. Local JAX 0.10 exposes `jax.nn.scaled_dot_general`; when configs are provided it calls a cuDNN-backed scaled-dot implementation from JAX's `cudnn.scaled_matmul_stablehlo` path. Current XLA has an explicit `scaled-dot` HLO, a GPU `ScaledDotRewriter`, and a cuDNN block-scaled-dot custom call target (`__cudnn$blockScaledDot`).

Pros:

- It is no longer just pattern matching Q/DQ arithmetic around an ordinary dot; the scale metadata is explicit in the compiler IR.
- It should be the most direct way to exercise the newer XLA/cuDNN block-scaled matmul path when the recipe and hardware match.
- The implementation can be made visible in HLO by looking for `scaled-dot` and the cuDNN block-scaled-dot custom call.

Risks:

- Local JAX's public config helper currently advertises `mxfp8` and `nvfp4`, not the month-one Hopper per-tensor delayed-scaling E4M3/E5M2 recipe. This may be more Blackwell/block-scaled oriented than the H100 target.
- The `implementation="cudnn"` selector is deprecated/ignored; backend selection is automatic, so the project must verify what actually lowers on the target machine.
- There is no obvious ragged/grouped-MoE analogue to `jax.nn.scaled_dot_general`, so even if C wins for dense projections, MoE may still require A or B.

## Phase 1: lowering-path decision

Do not assume A, B, or C is the right implementation target. Before implementing the `Fp8Dot` helper, run a short research spike that compiles and profiles tiny dense and ragged FP8 dots on the target JAX/XLA stack.

The spike must compare at least these paths:

- A: Haliax/Flax-style delayed scaling that emits Q/DQ around `dot_general`.
- B: Qwix-style quantized `dot_general_qt` and `ragged_dot_qt`.
- C: `jax.nn.scaled_dot_general`, which local JAX 0.10 exposes and current XLA lowers as a distinct `scaled-dot` HLO / cuDNN block-scaled-dot path.

The result should pick the implementation surface for the rest of the trial and record the evidence in the benchmark report: relevant jaxpr/HLO snippets, whether XLA emits Q/DQ plus `__cublas$lt$matmul$f8`, `scaled-dot`, `__cudnn$blockScaledDot`, or another custom call, and whether profiler counters show H100 FP8 tensor-core use. If the best path is still A, explain why it wins over the newer B/C options for this worktree.

Phase 1 deliverables:

- A table comparing A/B/C for dense projections: API fit, state/checkpoint fit, HLO shape, profiler evidence, numerical recipe, and implementation cost.
- A separate table for routed MoE: Pallas-Triton ragged-dot, XLA `ragged_dot_general`, and Qwix `ragged_dot_qt`.
- A written decision: one primary dense path, one primary MoE path, and any fallback. It is acceptable for dense and MoE to choose different approaches.
- A dependency decision on Qwix: whether to adopt it directly, wrap only its relevant primitives, borrow its design without adding the dependency, or avoid it. Adoption is acceptable if it is lightweight enough for Marin/Haliax and materially improves the quantized-dot abstraction.
- A state-shape decision for routed MoE: whether the dense-projection state object can also represent grouped/ragged matmuls, or whether MoE needs a separate `QuantizedRaggedDot`/`RaggedFp8Dot` state object. Do not assume `Fp8Dot` is correct for ragged matmul just because both operations are ultimately GEMMs.
- An architecture decision on whether Grug should keep raw weight arrays plus an `Fp8Dot`/quantized-dot helper, or move selected projections toward a `haliax.nn.Linear`-like module surface. The `Linear`-like option is probably less Gruggy because Grug's experiment template intentionally keeps weights and einsums explicit, but it may reuse more existing Haliax quantization machinery and reduce one-off FP8 state handling.
- A forward-compatibility note for TPU int8. Even though the month-one target is H100 FP8, the selected abstraction should not make future int8-on-TPU work awkward. Prefer names and boundaries like "quantized dot" or backend-specific lowering choices over APIs that permanently encode Hopper FP8 assumptions in all dense projection code.
- A decision on whether the month-one target remains per-tensor delayed scaling on H100, or whether the evidence points to block-scaled MXFP8 as a follow-up instead of this trial's core target.
- A performance target proposal derived from measurement, not a hard-coded guess. Use the BF16 profile to estimate the fraction of step time spent in FP8-eligible GEMMs, compare expected kernel-level speedup against the H100 roofline/profiler counters, and set separate end-to-end and expert-kernel targets before the implementation phase.

## New config surface

File: `experiments/grug/moe/model.py`

```python
@dataclass(frozen=True)
class GrugFp8Config:
    """FP8 training options for Grug MoE GEMMs.

    When disabled, model behavior is unchanged. When enabled, selected dense
    and routed-expert GEMMs quantize forward activations and weights to E4M3,
    quantize output gradients to E5M2, and accumulate in FP32. Router logits,
    normalization, softmax, QB beta computation, attention score softmax, and
    loss computation remain outside this FP8 scope.
    """

    enabled: bool = False
    dense: bool = True
    moe: bool = True
    amax_history_length: int = 1024
    dot_precision: str = "ANY_F8_ANY_F8_F32"
    fast_accum: bool = False
    checkpoint_metadata: bool = True
```

`GrugModelConfig` adds:

```python
fp8: GrugFp8Config = dataclasses.field(default_factory=GrugFp8Config)
```

Validation:

- `amax_history_length > 0`.
- `fast_accum=True` maps to `ANY_F8_ANY_F8_F32_FAST_ACCUM` unless `dot_precision` is explicitly set.
- FP8 helpers raise `ValueError` when enabled on unsupported dtypes or unsupported backends rather than silently falling back to BF16.

## FP8 dot helper

File: `lib/haliax/src/haliax/fp8_dot.py`

This API is provisional until the lowering-path spike above completes. It should hide the selected implementation from Grug call sites, but it should not bake in a Q/DQ-only assumption.

Phase 1 must decide whether this helper remains a raw-array companion object or whether the implementation should instead introduce a `haliax.nn.Linear`-like projection module for Grug dense weights. The tradeoff is:

- Raw arrays plus `Fp8Dot` are more Gruggy: they preserve the template-first style, keep sharding and einsums visible, and avoid turning the experiment into a Haliax module rewrite.
- A `Linear`-like surface is less Gruggy, but may be the cleaner abstraction if the winning path is shared with existing Haliax quantization conversion, Qwix-style module interception, or future TPU int8. It may also make checkpoint/state ownership simpler by colocating weights and quantization metadata.

Whichever surface wins, the public call sites should be phrased around "quantized projection/dot" rather than "FP8-only dot" when practical, so int8 TPU lowering can reuse the same boundary later without a broad model rewrite.

```python
class Fp8Dot(eqx.Module):
    """Stateful delayed-scaling FP8 dot helper for raw JAX arrays.

    The helper stores one input scale/history, one kernel scale/history, and one
    output-gradient scale/history. Calling it returns the dequantized result in
    the requested output dtype while the custom VJP emits overwrite metadata for
    the next scale/history state. The helper is intended for a single logical
    projection; callers should not share it across unrelated weights.
    """

    input_scale: jax.Array
    output_grad_scale: jax.Array
    kernel_scale: jax.Array
    input_amax_history: jax.Array
    output_grad_amax_history: jax.Array
    kernel_amax_history: jax.Array
    lowering: str = eqx.field(static=True)
    precision: str | None = eqx.field(static=True)

    @staticmethod
    def init(
        amax_history_length: int,
        *,
        lowering: str,
        precision: str | None = None,
    ) -> "Fp8Dot": ...

    def __call__(
        self,
        lhs: jax.Array,
        rhs: jax.Array,
        dimension_numbers: jax.lax.DotDimensionNumbers,
        *,
        out_sharding: jax.sharding.PartitionSpec | None = None,
        output_dtype: jnp.dtype | None = None,
    ) -> jax.Array: ...
```

Helper functions:

```python
def fp8_einsum(
    spec: str,
    lhs: jax.Array,
    rhs: jax.Array,
    fp8_dot: Fp8Dot | None,
    *,
    out_sharding: jax.sharding.PartitionSpec | None = None,
) -> jax.Array:
    """Run an einsum through FP8 when `fp8_dot` is present, otherwise use `jnp.einsum`."""
```

The helper must preserve existing output sharding semantics at the call sites.

## Model fields

File: `experiments/grug/moe/model.py`

`CausalSelfAttention` adds one optional `Fp8Dot` per projection:

```python
fp8_q: Fp8Dot | None
fp8_k: Fp8Dot | None
fp8_v: Fp8Dot | None
fp8_o: Fp8Dot | None
```

`DenseMLP` adds:

```python
fp8_gate: Fp8Dot | None
fp8_up: Fp8Dot | None
fp8_down: Fp8Dot | None
```

`MoEExpertMlp` in `lib/levanter/src/levanter/grug/grug_moe.py` adds optional quantized state for the two expert GEMMs. The exact type is a Phase 1 output:

```python
quantized_w13: QuantizedRaggedDot | Fp8Dot | None
quantized_w2: QuantizedRaggedDot | Fp8Dot | None
```

`Fp8Dot` is only acceptable here if Phase 1 proves its scale/history semantics match grouped matmul. Ragged MoE may need a different state shape because scales may be per expert, per expert-output axis, per block, or backend-owned by Qwix-style `QArray`s rather than one input scale, one kernel scale, and one output-gradient scale per logical dense projection.

## Ragged MoE surface

File: `lib/haliax/src/haliax/nn/ragged_dot.py`

```python
def ragged_dot(
    lhs_: jax.Array,
    rhs_: jax.Array,
    group_sizes_: jax.Array,
    ar: bool = False,
    implementation: Implementation = "auto",
    *,
    precision: PrecisionLike | None = None,
) -> jax.Array:
    """Grouped matrix multiply.

    When `precision` is provided, supported backends must use it for the
    underlying dot. Backends that cannot honor the precision must raise
    `NotImplementedError` so callers can fail fast or select another backend.
    """
```

If a stateful quantized wrapper is needed around ragged-dot, add a ragged-specific state object unless Phase 1 proves the dense `Fp8Dot` state is sufficient:

```python
class QuantizedRaggedDot(eqx.Module):
    """Quantized grouped-matmul state.

    The state layout is intentionally not specified before Phase 1 because
    grouped MoE may require per-expert, per-axis, block-scaled, or backend-owned
    scaling metadata rather than dense-projection delayed-scaling metadata.
    """

    lowering: str = eqx.field(static=True)


def fp8_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    quantized_dot: QuantizedRaggedDot | Fp8Dot,
    *,
    implementation: Implementation = "auto",
) -> jax.Array:
    """Apply the selected quantized lowering to a grouped matrix multiply."""
```

The Pallas-Triton backend must either honor `precision` in `pl.dot` or raise `NotImplementedError`. The XLA backend must pass the equivalent precision if `jax.lax.ragged_dot_general` accepts it in the local JAX version; otherwise it must raise. The lowering-path spike should also evaluate Qwix's `ragged_dot_qt` shape: it has a quantized ragged-dot wrapper, but the spec must verify whether its fast path actually maps to efficient H100 FP8 work for Grug's grouped-MoE shapes.

The Phase 1 report must explicitly answer whether `Fp8Dot` is the right abstraction for `fp8_ragged_dot`. The default assumption should be "no" until proven otherwise, because the dense helper's state model is tied to one logical projection and ordinary dot-general backward passes, while routed MoE has group sizes, expert axes, backend dispatch, and custom VJPs.

## Train-step contract

File: `experiments/grug/moe/train.py`

Add a local helper:

```python
def apply_grug_updates(
    params: Transformer,
    updates: optax.Updates,
    overwrites: Transformer | None,
) -> Transformer:
    """Apply optimizer updates plus direct FP8 state overwrites to Grug params."""
```

The train step changes from direct `optax.apply_updates(qb_params, updates)` to:

```python
overwrites, trainable_grads = partition_for_grad_overwrite(grads)
updates, opt_state = optimizer.update(trainable_grads, state.opt_state, qb_params)
params = apply_grug_updates(qb_params, updates, overwrites)
```

Optimizer initialization must exclude overwrite-only FP8 metadata. EMA must not average FP8 scale/history arrays unless a follow-up explicitly opts into that behavior.

## Acceptance criteria

- `./infra/pre-commit.py --all-files` passes.
- `uv run pytest lib/haliax/tests/test_fp8.py lib/levanter/tests/grug/test_grugformer_moe.py tests/test_grug_variant_contracts.py` passes, with new FP8 coverage included in existing files where practical.
- The Phase 1 report documents which FP8 lowering path was selected and why. It must explicitly state whether the implementation relies on A/QDQ pattern matching, B/Qwix-style low-precision `dot_general`, C/XLA `scaled-dot` and cuDNN block-scaled dot, or a hybrid.
- The Phase 1 report states whether routed MoE reuses `Fp8Dot` or introduces a ragged-specific quantized state object, with the scale shapes and backward-pass state updates spelled out.
- The Phase 1 report documents the architecture choice between raw arrays plus helper state and a `haliax.nn.Linear`-like projection surface, including why the choice is acceptable for Grug maintainability.
- The Phase 1 report includes a TPU int8 forward-compatibility check: which API boundaries, dtype names, and tests would survive a future int8 backend, and which would need refactoring.
- A tiny FP8-enabled Grug MoE train step under JIT updates FP8 scale histories and keeps optimizer state free of overwrite-only metadata.
- An H100 synthetic-data profile shows FP8 matmul use for attention/shared dense projections.
- An H100 synthetic-data profile shows FP8 matmul use for the routed expert up/gate and down GEMMs, or the final report explains precisely which backend prevents it.
- A short real-data Grug MoE run finishes without NaN loss and keeps router metrics finite.
- The benchmark report compares BF16 and FP8 on tokens/sec, step time breakdown, MFU or analytic-FLOP throughput, loss movement, compile time, peak HBM, and profiler evidence.
- The benchmark report evaluates against the Phase 1 performance targets. Those targets must state their derivation; if end-to-end speedup is communication-bound, report kernel-level throughput separately instead of substituting an unsourced fixed threshold.

## Out of scope

- Blackwell MXFP8 or blockwise scaling. Hopper blockwise FP8 for grouped MoE is a stretch goal, not a required month-one deliverable.
- FP8 optimizer states.
- FP8 router logits, top-k, QB beta, attention softmax, loss, or normalization.
- Portable FP8 checkpoint/export format for inference.
- Rewriting Grug into Haliax modules.
- Matching Transformer Engine's full API surface.
