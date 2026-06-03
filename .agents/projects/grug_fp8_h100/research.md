# Grug FP8 on H100 Research

## Framing

Add FP8 training support to the Grug MoE variant on H100, with the specific goal of matching the shape of FP8 support in Megatron/Transformer Engine and MaxText: FP8 for large GEMMs, high precision for routing/norm/loss numerics, and measurable throughput improvement without loss instability. This is scoped as a roughly month-long work trial, so the proposal emphasizes concrete checkpoints and acceptance criteria over a fully generalized quantization framework.

## In-repo findings

- `experiments/grug/moe/model.py:65` defines the MoE model config. It currently has no FP8 or quantization field.
- `experiments/grug/moe/model.py:128` stores attention projection weights as raw JAX arrays, and `experiments/grug/moe/model.py:155` through `experiments/grug/moe/model.py:174` applies Q/K/V/O via `jnp.einsum`.
- `experiments/grug/moe/model.py:195` stores GatedNorm down/up projections as raw arrays. These are dense matmuls, but the rank is only 128, so they are a lower-priority FP8 target than attention, shared MLP, and routed experts.
- `experiments/grug/moe/model.py:220` stores the shared dense MLP weights as raw JAX arrays, and `experiments/grug/moe/model.py:248` through `experiments/grug/moe/model.py:250` applies gate/up/down via `jnp.einsum`.
- `experiments/grug/moe/model.py:483` stores the LM head/output projection. The normal training loss path calls fused linear softmax cross entropy, so including the output projection is a separate kernel question from ordinary dense block projections.
- `experiments/grug/moe/model.py:329` implements QB-routed MoE. Router logits are explicitly kept in fp32 at `experiments/grug/moe/model.py:371`, which should remain true for the FP8 trial.
- `experiments/grug/moe/model.py:412` calls `MoEExpertMlp`, which dispatches into `levanter.grug.grug_moe.moe_mlp`.
- `lib/levanter/src/levanter/grug/grug_moe.py:55` defines `MoEExpertMlp` with raw expert weights `w_gate_up` and `w_down`; `lib/levanter/src/levanter/grug/grug_moe.py:100` forwards them into `moe_mlp`.
- MoE expert matmuls are all `haliax.nn.ragged_dot` calls: local scatter in `lib/levanter/src/levanter/grug/_moe/scatter.py:44` and `lib/levanter/src/levanter/grug/_moe/scatter.py:48`, local sonic in `lib/levanter/src/levanter/grug/_moe/sonic.py:332` and `lib/levanter/src/levanter/grug/_moe/sonic.py:336`, EP ring in `lib/levanter/src/levanter/grug/_moe/ep_ring.py:101` and `lib/levanter/src/levanter/grug/_moe/ep_ring.py:105`, and EP ragged all-to-all in `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py:92` and `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py:95`.
- `lib/haliax/src/haliax/nn/ragged_dot.py:339` exposes `ragged_dot(lhs, rhs, group_sizes, ar=False, implementation="auto")`; it does not expose precision, preferred element type, or FP8 scaling state.
- `lib/haliax/src/haliax/nn/ragged_dot.py:82` implements a Pallas-Triton GPU grouped matmul. It casts operands to `jnp.result_type(a, b)` and calls `pl.dot` at `lib/haliax/src/haliax/nn/ragged_dot.py:106`.
- `lib/haliax/src/haliax/nn/ragged_dot.py:296` has an XLA fallback via `jax.lax.ragged_dot_general`.
- `lib/haliax/src/haliax/nn/linear.py:188` defines `MoELinear`; `lib/haliax/src/haliax/nn/linear.py:198` has an explicit TODO for ragged-dot quantization and no `dot_general` field.
- `lib/haliax/src/haliax/quantization.py:154` defines `Fp8DotGeneralOp`, a delayed-scaling dot-general wrapper with separate input, kernel, and output-gradient amax histories.
- `lib/haliax/src/haliax/_src/fp8.py:83` quantizes forward inputs/weights to `jnp.float8_e4m3fn`; `lib/haliax/src/haliax/_src/fp8.py:103` quantizes output gradients to `jnp.float8_e5m2`.
- `lib/haliax/src/haliax/quantization.py:246` can rewrite `haliax.nn.Linear` modules, but Grug MoE uses raw arrays/einsums/ragged-dot, so that rewrite does not apply directly.
- `experiments/grug/moe/train.py:254` initializes a custom `GrugTrainState`; `experiments/grug/moe/train.py:312` computes gradients and `experiments/grug/moe/train.py:314` applies optax updates directly. If FP8 state uses `OverwriteWithGradient`, this train step must partition overwrites like Levanter's generic trainer does.
- `lib/levanter/src/levanter/grad_accum.py:145` and `lib/levanter/src/levanter/trainer_state.py:101` show existing support for Haliax quantization state in generic training paths.
- `lib/levanter/tests/grug/test_grugformer_moe.py:119` and later tests already cover local and EP MoE backend semantics; these are the natural test extension points for FP8 output sanity and lowering.
- Local JAX is `0.10.0`. `jax.lax.dot_general` accepts FP8 dot presets through the `precision` argument; observed presets include `ANY_F8_ANY_F8_F32` and `ANY_F8_ANY_F8_F32_FAST_ACCUM`. Local JAX also exposes `jax.nn.scaled_dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32, configs=None, implementation=None)`, while `jax.lax.scaled_dot_general` is not present.

## Prior-art shape

- NVIDIA Megatron Core exposes FP8 as a Transformer Engine-backed mode. Its documented `fp8` choices include `e4m3` and `hybrid`; `hybrid` uses E4M3 for activation and weight tensors and E5M2 for output activation gradients. See the Megatron Core transformer config docs: <https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html>.
- Megatron Core also documents scaling recipes including tensorwise current scaling, delayed scaling, MXFP8, blockwise, and custom. Megatron-LM's MoE README now points production MoE training on Hopper toward blockwise FP8, with E4M3 by default and block shapes such as 1x128 activation blocks and 128x128 weight blocks: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md>. That is an important stretch target, but per-tensor delayed scaling is the more realistic one-month starting point.
- NVIDIA Transformer Engine's delayed-scaling docs describe one FP32 scale per tensor and historical amax-based scaling. Delayed scaling avoids an extra current-tensor read before each quantization, which is the right first target for an H100 training trial: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html>.
- Transformer Engine's FP8 primer distinguishes E4M3 and E5M2 as the Hopper FP8 formats and uses a delayed-scaling recipe as the common training recipe: <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.9/user-guide/examples/fp8_primer.html>.
- MaxText's performance docs describe quantization recipes including `fp8`, `fp8_full`, `fp8_gpu`, and `fp8_nanoo`, with benchmarking guidance based on synthetic data to remove input stalls: <https://maxtext.readthedocs.io/en/latest/guides/optimization/benchmark_and_performance.html>.
- MaxText quantization docs describe a DeepSeek-style `w8a8g8` recipe that targets attention projections, weight all-gathers, and MoE grouped matmuls (`gmm`/`tgmm`). The documented format mix is E4M3FN for activations/weights and E5M2 for gradients, with per-axis/static scaling for weights/activations and dynamic scaling for gradients: <https://maxtext.readthedocs.io/en/latest/reference/core_concepts/quantization.html>. This is more ambitious than the proposed first pass but useful as a comparison point.
- Current JAX docs expose FP8-relevant dot algorithm selection through `jax.lax.dot_general` precision and newer scaled-dot APIs. Local Marin/JAX has both the dot precision presets and `jax.nn.scaled_dot_general`.

## Upstream lowering check

- Checked upstream MaxText at commit `56db4d5c5066e7a0b40638bcd7b86b1aad1315f1`. MaxText still has multiple FP8 routes:
  - `quantization=fp8` uses Flax `nn.Fp8DirectDotGeneralOp` / `nn.Fp8Einsum` wrappers in `src/maxtext/layers/quantizations.py`.
  - `quantization=fp8_full` uses Qwix `dot_general_qt.DotGeneralQtConfig` with E4M3FN forward activations/weights and E5M2 gradients.
  - MaxText's Qwix wrapper calls `dot_general_qt.dot_general_qt`; it does not directly call `jax.nn.scaled_dot_general`.
- Checked upstream Qwix at commit `50e951133cf77786b913a37c62dc6f673725c274`. Qwix's ordinary QT path quantizes operands to `QArray`, calls `jax.lax.dot_general` on quantized values, and applies scales around the result. Qwix also has `ragged_dot_qt`, which quantizes `[M, K]` activations and `[G, K, N]` expert weights before calling its quantized `ragged_dot` wrapper.
- Checked upstream XLA at commit `089e80dd2a5d0f39fa0d4efd68a24c9e73371f4c`. XLA now has an explicit `HloOpcode::kScaledDot` with `XlaBuilder::ScaledDot`. The GPU optimization pipeline runs `ScaledDotRewriter` unless `xla_gpu_experimental_scaled_dot_with_triton` is enabled, and the post-layout pipeline runs an FP8-only `GemmRewriter`. That means current XLA is no longer only a Q/DQ pattern-matching story.

## Surprises

- The easiest existing FP8 path is not wired into Grug because Grug intentionally avoids Haliax modules for its template-first experiment style.
- The hardest part is not attention or shared dense MLP; it is routed expert FP8 because the MoE hot path is grouped/ragged matmul with custom VJP and backend dispatch.
- The Grug MoE train loop does not use the generic `TrainerState`, so FP8 state overwrite handling must be added explicitly.
- Any helper shared by Grug dense projections and `levanter.grug` MoE must live in Haliax or Levanter, not under `experiments/grug`, because Levanter cannot import from Marin experiments.
- The implementation surface should not be fixed before a lowering spike. The plausible paths are Flax/Haliax delayed-scaling Q/DQ, Qwix-style quantized dot/ragged-dot, or JAX `scaled_dot_general` / XLA `scaled-dot`.

## Unknowns

- Whether XLA `ragged_dot_general` can lower FP8 efficiently enough on H100, or whether the Pallas-Triton ragged kernel must be taught FP8 `pl.dot`/accumulation behavior.
- Whether the Haliax delayed-scaling implementation is sufficient for Megatron-like performance on H100, since `dot_general_with_precision` currently calls `lax.dot_general(..., precision=lax.Precision.DEFAULT)` rather than an explicit FP8 algorithm preset or `jax.nn.scaled_dot_general`.
- Whether `jax.nn.scaled_dot_general` can express the desired per-tensor delayed-scaling recipe cleanly enough for dense Grug projections, and whether it has any credible ragged/grouped-MoE analogue.
- Whether output projection / fused cross entropy should be included in the first trial's "dense projections" scope.
