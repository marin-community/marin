# Grug FP8 on H100

Add FP8 training support to the Grug MoE experiment so H100 runs can use FP8 for the expensive transformer GEMMs while keeping numerically sensitive routing, normalization, attention softmax, and loss paths in higher precision. The target is not a general-purpose quantization framework; it is a one-month work trial with a concrete outcome: an FP8 Grug MoE path that can be benchmarked against BF16 and compared to the shape of Megatron/Transformer Engine and MaxText FP8 training.

## Challenges

Grug's model is intentionally template-first and uses raw JAX arrays rather than Haliax `Linear` modules. The existing Haliax FP8 path rewrites `haliax.nn.Linear` and stores delayed-scaling state inside `Fp8DotGeneralOp`, but Grug's dense projections are `jnp.einsum` sites in `experiments/grug/moe/model.py`.

The larger challenge is MoE. Routed expert matmuls are not ordinary dense matmuls: `MoEExpertMlp` calls `moe_mlp`, and the expert-parallel backends call `haliax.nn.ragged_dot` for the up/gate and down projections. That function currently dispatches to Pallas-Triton, TPU megablox, or XLA ragged-dot without a quantized-dot contract. Matching Megatron/MaxText-style FP8 therefore means building both ordinary dense FP8 and a credible FP8 grouped-matmul path.

## Costs / Risks

- This adds state to the Grug model for amax histories and scales. The custom Grug train loop must learn how to apply FP8 state overwrites, otherwise scale histories will silently stop updating.
- FP8 can look "enabled" while failing to use H100 FP8 tensor cores. The trial must inspect lowering/profiles, not just dtypes.
- Routed MoE capacity and dispatch shape changes may hide the FP8 benefit behind ragged kernel overhead or collectives.
- Delayed scaling is stable for mainstream transformer FP8 training, but Grug's QB routing and GatedNorm recipe may have different activation distributions.
- A full Transformer Engine clone is too large for a month. The trial should accept a narrow, measurable Grug path and explicitly leave Blackwell MXFP8, inference checkpoint formats, and generic Haliax conversion out of scope.

## Design

Introduce a Grug-local FP8 config and a reusable Haliax FP8 dot helper rather than rewriting the model into Haliax modules. The model config gets an optional FP8 field, defaulting to disabled. When enabled, the large dense GEMM sites call the FP8-aware dot helper, which owns delayed-scaling state and hides the selected lowering path from Grug call sites. The first dense scope is attention Q/K/V/O and shared expert MLP; GatedNorm projections are lower priority because their intermediate rank is small. Router projection stays fp32, as the current code already comments at `experiments/grug/moe/model.py:371`, because top-k, QB beta, softmax, and z-loss are more sensitive than the large projection GEMMs.

The FP8 recipe should start with Megatron/Transformer Engine's common H100 shape: E4M3 for forward activations and weights, E5M2 for output gradients, FP32 accumulation, per-tensor delayed scaling, amax history length 1024, and an opt-in fast-accumulation mode for profiling. This mostly matches the existing Haliax implementation in `lib/haliax/src/haliax/_src/fp8.py`, but the helper must be explicit about the lowering path because local JAX exposes both `ANY_F8_ANY_F8_F32` precision presets and `jax.nn.scaled_dot_general`, while current XLA has an explicit `scaled-dot` HLO. Megatron's production MoE path and MaxText's DeepSeek-style recipe both point beyond this toward blockwise/per-axis grouped-GEMM FP8. That should be treated as a stretch goal or follow-up unless per-tensor delayed scaling fails to show useful H100 kernels.

Before week-1 implementation, run a focused lowering spike. Compile tiny dense and ragged FP8 examples using the existing Haliax/Flax Q/DQ style, a Qwix-style quantized dot/ragged-dot shape, and `jax.nn.scaled_dot_general` for dense projections. Pick the helper implementation from observed HLO/profiler behavior, not API familiarity. The final implementation may still use Q/DQ if that is what XLA optimizes best on this stack, but the report must say so explicitly.

For MoE, add an FP8 ragged-dot contract instead of baking quantization into every backend. The interface should look like `ragged_dot(..., dot_general=None, precision=None)` or a more specific `ragged_fp8_dot(...)` if the backend differences make a generic interface misleading. The minimum accepted MoE path is expert-parallel H100 with the active Grug backend. If Pallas-Triton can use FP8 through `pl.dot` after quantize/dequantize casts and profiles as FP8 tensor-core work, keep the implementation there. If not, evaluate whether `jax.lax.ragged_dot_general` with FP8 inputs and dot precision lowers adequately, and compare that with Qwix's `ragged_dot_qt` approach. If none works, the trial should report this as a blocking systems finding rather than pretending the feature landed.

The train loop must integrate FP8 state updates. `experiments/grug/moe/train.py` currently calls `jax.value_and_grad`, `optimizer.update`, and `optax.apply_updates` directly. FP8 state should use the existing Haliax overwrite pattern: partition overwrite leaves out of the gradient, update optimizer state only for real trainable gradients, and apply overwrites directly to the model. EMA should either include FP8 state from current params or explicitly exclude FP8 metadata; the first version should prefer excluding metadata from EMA if doing so avoids confusing scale-history averaging.

The work-trial evaluation should be staged:

Week 1: dense FP8 path. Add config, Haliax helper, train-step overwrite handling, CPU/GPU unit tests for scale updates, and a small H100 smoke run proving no NaNs. Include the LM head only if the fused loss path can use the same dot contract without turning week 1 into kernel work.

Week 2: MoE FP8 path. Add FP8 state to `MoEExpertMlp`, wire expert up/gate/down ragged matmuls, and add tests comparing FP8 output shape/finite behavior and BF16 closeness on small problems.

Week 3: profiling and correctness gates. Run BF16 vs FP8 synthetic-data benchmarks on H100, inspect XLA/HLO/profiler evidence of FP8 kernels, and run a short real-data Grug MoE training job through at least the current eval cadence.

Week 4: polish and handoff. Fix checkpoint/state serialization issues, document enabled/disabled scopes, produce a benchmark report, and leave a clear follow-up list for deeper kernel work if the ragged path is the bottleneck.

## Testing

Unit tests should extend `lib/haliax/tests/test_fp8.py` for any reusable helper and `lib/levanter/tests/grug/test_grugformer_moe.py` for MoE behavior. Tests should assert finite outputs, stable state updates, and numerical closeness to BF16 within FP8-appropriate tolerances on small shapes. They should not assert incidental command strings or profiler labels.

Integration tests should include a tiny Grug MoE train step under JIT with FP8 enabled, confirming that scale histories change after the step and optimizer state initialization ignores overwrite-only metadata. On H100, acceptance requires a profiler or HLO artifact showing FP8 matmul use for dense projections and the MoE expert path, plus a synthetic-data throughput comparison against the BF16 Grug MoE baseline.

The run gate for the trial is a short real-data training run that does not NaN, keeps router metrics sane, and achieves either a meaningful end-to-end speedup or a well-explained bottleneck profile. "Meaningful" must be set during Phase 1 from the BF16 profile: estimate how much step time is in FP8-eligible GEMMs, compare expected kernel gains against profiler counters or roofline estimates, and set separate end-to-end and expert-kernel targets before implementation. Do not use unsourced fixed thresholds. Short-run loss and compile-time tolerances should be set the same way from the baseline run and risk tolerance.

## Open Questions

- Should the first trial include the output projection / fused cross entropy matmul, or should "dense projections" mean only transformer block projections?
- Should GatedNorm's small dense projections be included for completeness, or excluded until the high-FLOP projections are proven?
- Which Grug MoE backend is the real target on H100 today: `ring`, `ragged_all_to_all`, or whichever `moe_implementation=None` resolves to in the launch config?
- Is the acceptance bar "match Megatron/MaxText numerics shape" or "match their speedup envelope"? The former is feasible in a month; the latter may depend on custom grouped-matmul kernel work.
- Should FP8 metadata be checkpointed as part of Grug state, or should scales be allowed to re-warm after restore for the first trial?
