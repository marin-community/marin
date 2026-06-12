# Splash Prefix-LM Block Skipping: Research Logbook

## Scope
- Goal: Extend Splash Attention support from static causal/local masks plus runtime segment IDs to prefix-LM masks with data-dependent prefix lengths under packing, while preserving or improving block skipping on TPU.
- Primary metrics: Correctness versus dense/reference attention; steady-state attention time on v4-8 and v5p-8 for sequence lengths 8192 and 16384; compile-including time for first-call regressions.
- Constraints: Keep existing causal, full, sliding-window, and segment-ID Splash paths working. Avoid materializing large dense masks on the common static path. Start with v4-8 if v5p availability is poor.

## Baseline
- Date: 2026-06-11
- Code refs:
  - `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`
  - `lib/levanter/src/levanter/layers/attention.py`
  - `lib/levanter/src/levanter/layers/attention_mask.py`
  - `lib/levanter/src/levanter/grug/attention/_core.py`
- Baseline numbers: TBD. PR #6314 merged the shared Splash mask lowering helper and passed a v4-8 Iris smoke test with existing causal/sliding/segment-ID masks.

## Experiment Log
### 2026-06-11 00:00 - Kickoff on merged Splash helper
- Hypothesis: Prefix-LM can be added as a structured `AttentionMask` field and lowered to Splash without a major speed hit by keeping static mask lowering for uniform prefix lengths, then adding a dynamic per-example path for packed/data-dependent prefix lengths.
- Command: `git switch -c codex/splash-prefixlm-block-skipping origin/main`
- Config: Local code inspection on merged `origin/main`.
- Result: Current JAX Splash dynamic-mask path accepts `[heads, q, kv]` masks and still computes `block_mask`/`data_next` to skip fully empty blocks. It does not directly accept a batch axis. Existing Levanter Splash invocation vmaps over batch and shares one kernel object/mask-info across examples.
- Interpretation: Uniform prefix-LM can use a static/sliceable Splash mask. Packed data-dependent prefix-LM needs either per-example kernel/mask-info vmapping or a batch-to-head folding strategy; using only `segment_ids` cannot skip segment-empty blocks because Splash applies `segment_ids` inside visited blocks.
- Next action: Prototype a structured prefix-LM mask surface and static lowering tests first, then test whether vmapping a dynamic-mask `SplashAttentionKernel` over per-example mask-info is accepted by `shard_map`/Pallas.

### 2026-06-11 00:30 - Batched dynamic Splash kernel probe
- Hypothesis: A per-example `SplashAttentionKernel` pytree can be batched over the Levanter batch axis and passed through the existing `shard_map` boundary, making per-example dynamic prefix metadata possible without changing the Pallas kernel interface.
- Command: Local CPU interpret-mode probes using `splash_attention_kernel.make_splash_mha(..., interpret=True)` with dynamic masks of shape `[B, H, 128, 128]`, then `jax.vmap(make_kernel)` and a `shard_map` call with leading-batch partition specs for every kernel metadata leaf.
- Config: `B=2`, `H=2`, `S=128`, `D=32`, block sizes all 128, single-device CPU mesh with axis `data`.
- Result: `jax.vmap(make_splash_mha)` produced a batched `SplashAttentionKernel` pytree. `jax.vmap` over per-example kernel calls succeeded. A `shard_map` wrapper with leading `P("data", ...)` specs for the batched kernel leaves also succeeded and returned `(2, 2, 128, 32)`.
- Interpretation: The Pallas/JAX interface does not appear to block batched per-example mask metadata. The remaining performance issue is how to construct compact dynamic prefix/packed metadata without dense `[B, H, S, S]` masks at 8192/16384.
- Next action: Build compact prefix-length `MaskInfo` directly from block predicates; avoid `process_dynamic_mask` for long sequence lengths because it materializes full per-block mask payloads.

### 2026-06-11 01:00 - Dynamic prefix length reference mask
- Hypothesis: Add a batch-dependent `prefix_lengths` field to `AttentionMask` as the public structured representation for data-dependent prefix lengths, while keeping Splash lowering explicit about unsupported dynamic prefix metadata.
- Command:
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or sliding_window_mask'`
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py`
- Config: Local CPU tests.
- Result: Reference mask tests passed with `4 passed`; Splash helper tests passed with `6 passed`.
- Interpretation: `AttentionMask.prefix_lm(prefix_lengths=...)` now gives correct dense/reference semantics for per-batch prefix lengths. Splash lowering rejects dynamic prefix lengths instead of silently dropping them until the compact metadata path is implemented.
- Next action: Implement and test a compact forward `MaskInfo` builder for prefix lengths, then extend it to backward and segment-aware packed-doc masks.
