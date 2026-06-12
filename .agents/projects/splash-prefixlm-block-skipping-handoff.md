# Splash Prefix-LM Block Skipping Handoff

## Current Branch

- Branch: `codex/splash-prefixlm-block-skipping`
- Base: `origin/main` as of 2026-06-11, including merged PR #6314.
- Research logbook: `.agents/logbooks/splash-prefixlm-block-skipping.md`

## Goal

Build on the shared Splash Attention mask lowering from PR #6314 to support:

- Prefix-LM masks in Splash Attention.
- Data-dependent prefix lengths under packing.
- Block skipping when static or structured mask information makes blocks empty.
- Later: block skipping for segment-ID masks, moving toward THD-style packed attention without slowdown.

Target hardware for validation and tuning:

- First: v5p-8 and v4-8, with v4-8 acceptable if v5p availability is poor.
- Then: v5e and v6e.
- Shapes where skipping should matter: sequence lengths 8192 and 16384, usually with 512-token blocks.

## What Changed So Far

This checkpoint implements the first slices: static prefix-LM masks, reference semantics for batch-dependent prefix lengths, and compact Splash metadata for dynamic prefix lengths.

Files changed:

- `lib/levanter/src/levanter/layers/attention_mask.py`
  - Adds `AttentionMask.prefix_lm(...)`.
  - Adds `prefix_length`, `prefix_lengths`, and `prefix_mask` fields.
  - Materializes prefix-LM as `causal_or_prefix`, then applies explicit and segment masks.
  - Sliding-window prefix-LM semantics are: prefix keys remain visible to every query, while non-prefix keys are causal and local-window constrained.

- `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`
  - Adds `prefix_length`, `has_prefix_lengths`, and `has_prefix_mask` to `SplashAttentionMaskSpec`.
  - Adds `PrefixMask`, a sliceable Splash mask for static prefix keys.
  - Lowers static prefix-LM to `LogicalOr(causal_or_local_mask, PrefixMask)`, preserving Splash static mask processing and block skipping.
  - Adds compact `prefix_lm_mask_infos(...)` metadata for dynamic prefix lengths. The representation stores a full block grid but only `2 * q_blocks` partial blocks per example.
  - Still rejects dynamic `prefix_mask`.

- `lib/levanter/src/levanter/layers/attention.py`
  - Passes `prefix_length` / `has_prefix_mask` from `AttentionMask` into the Splash mask spec.
  - Builds one dynamic-prefix `SplashAttentionKernel` per batch example with `jax.vmap`, then passes the batched kernel pytree through `shard_map`.

- `lib/levanter/tests/test_attention.py`
  - Adds reference materialization tests for prefix-LM, prefix-LM plus sliding window, and per-batch dynamic prefix lengths.

- `lib/levanter/tests/kernels/test_splash_attention.py`
  - Adds static Splash prefix-LM lowering tests.
  - Adds compact forward/dKV `MaskInfo` reconstruction tests for dynamic prefix lengths.
  - Extends test mask materialization to handle `LogicalOr`.

## Validation Already Run

Both commands passed locally:

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py
```

Result after compact metadata: `14 passed`.

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or sliding_window_mask'
```

Result after adding dynamic prefix lengths: `4 passed`.

Additional local probe:

- CPU interpret-mode `jax.vmap(make_splash_mha)` over `[B, H, 128, 128]` dynamic masks succeeded.
- Passing the batched `SplashAttentionKernel` pytree through a one-device `shard_map` also succeeded when every kernel metadata leaf used a leading batch partition spec.
- This suggests the remaining hard part is compact metadata construction, not the Pallas call interface.

## Important Design State

JAX Splash has two mask processing paths:

- Static masks: `MultiHeadMask` / `Mask` objects processed by `process_mask`.
- Dynamic masks: `jax.Array` of shape `[heads, q, kv]` processed by `process_dynamic_mask`.

`process_dynamic_mask` still computes `block_mask` and `data_next`, so it can skip fully empty blocks. However, it has no batch axis. Levanter’s current Splash path creates one `SplashAttentionKernel` outside `shard_map`, then vmaps the kernel call over batch, only varying `q`, `k`, `v`, `segment_ids`, and sinks.

Dynamic prefix lengths are now represented by a batched `SplashAttentionKernel` pytree. A packed batch can use per-example prefix lengths and segment IDs for correctness. Segment IDs are still applied inside visited blocks by Splash, so this does not yet provide segment-ID block skipping.

## Open Technical Question

Need decide the dynamic packed-prefix representation.

Main options:

1. Per-example dynamic mask metadata:
   - Build per-example `[heads, q, kv]` masks or metadata.
   - Try `jax.vmap(make_splash_mha)(mask_bhsk)` to produce a batched `SplashAttentionKernel` pytree.
   - Then call the per-example kernel inside the existing batch vmap.
   - Local probe showed this works in CPU interpret mode, including through `shard_map` with leading batch specs.
   - Current branch uses compact metadata instead of dense `[heads, q, kv]` masks.

2. Structured metadata without dense `[B,H,S,S]` masks:
   - Derive `MaskInfo` directly from per-example prefix lengths and packed segment metadata.
   - This is probably the right performance direction if dynamic-mask dense materialization is too expensive at 8192/16384.
   - It likely requires bypassing or extending JAX Splash internals rather than only using public mask objects.

3. Fold batch into heads:
   - Represent per-example masks as extra heads and avoid a batch axis in the dynamic mask.
   - Likely awkward because the actual Q/K/V batch dimension remains separate and Splash expects `[H, Q, KV]` mask metadata, not `[B*H, Q, KV]` over a separate batch call.

## Key Source Files

- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py`
- `lib/levanter/src/levanter/layers/attention.py`

## Recommended Next Steps

1. Run a TPU smoke test for `AttentionMask.prefix_lm(prefix_lengths=...)` through `_tpu_splash_attention`.
2. Add correctness tests comparing:
   - static prefix length against reference attention,
   - per-example prefix masks,
   - prefix masks plus segment IDs for packed docs.
3. Add a benchmark harness for 8192/16384 that reports compile-including and steady-state timing on v4-8/v5p-8.
4. Implement segment-ID block skipping, probably by deriving block metadata from packed segment runs or THD metadata.

## Caveats

- Dynamic prefix-length support is implemented but not yet TPU-smoke-tested in this branch.
- Current Splash segment IDs mask inside visited blocks; they do not provide block skipping for packed segment masks by themselves.
- Dynamic `prefix_mask` remains unsupported in Splash; use structured `prefix_lengths` for the compact path.
