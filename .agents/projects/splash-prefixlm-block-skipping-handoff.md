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

This checkpoint implements the first, deliberately small slice: static prefix-LM masks.

Files changed:

- `lib/levanter/src/levanter/layers/attention_mask.py`
  - Adds `AttentionMask.prefix_lm(...)`.
  - Adds `prefix_length` and `prefix_mask` fields.
  - Materializes prefix-LM as `causal_or_prefix`, then applies explicit and segment masks.
  - Sliding-window prefix-LM semantics are: prefix keys remain visible to every query, while non-prefix keys are causal and local-window constrained.

- `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`
  - Adds `prefix_length` and `has_prefix_mask` to `SplashAttentionMaskSpec`.
  - Adds `PrefixMask`, a sliceable Splash mask for static prefix keys.
  - Lowers static prefix-LM to `LogicalOr(causal_or_local_mask, PrefixMask)`, preserving Splash static mask processing and block skipping.
  - Explicitly rejects dynamic `prefix_mask` in Splash for now.

- `lib/levanter/src/levanter/layers/attention.py`
  - Passes `prefix_length` / `has_prefix_mask` from `AttentionMask` into the Splash mask spec.

- `lib/levanter/tests/test_attention.py`
  - Adds reference materialization tests for prefix-LM and prefix-LM plus sliding window.

- `lib/levanter/tests/kernels/test_splash_attention.py`
  - Adds static Splash prefix-LM lowering tests.
  - Extends test mask materialization to handle `LogicalOr`.

## Validation Already Run

Both commands passed locally:

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py
```

Result: `6 passed`.

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or sliding_window_mask'
```

Result: `3 passed`.

## Important Design State

JAX Splash has two mask processing paths:

- Static masks: `MultiHeadMask` / `Mask` objects processed by `process_mask`.
- Dynamic masks: `jax.Array` of shape `[heads, q, kv]` processed by `process_dynamic_mask`.

`process_dynamic_mask` still computes `block_mask` and `data_next`, so it can skip fully empty blocks. However, it has no batch axis. Levanter’s current Splash path creates one `SplashAttentionKernel` outside `shard_map`, then vmaps the kernel call over batch, only varying `q`, `k`, `v`, `segment_ids`, and sinks.

This means data-dependent prefix-LM under packing is not solved by the static checkpoint. A packed batch has per-example prefix masks and segment boundaries, so one shared static `MultiHeadMask` is not enough.

## Open Technical Question

Need decide the dynamic packed-prefix representation.

Main options:

1. Per-example dynamic mask metadata:
   - Build per-example `[heads, q, kv]` masks or metadata.
   - Try `jax.vmap(make_splash_mha)(mask_bhsk)` to produce a batched `SplashAttentionKernel` pytree.
   - Then call the per-example kernel inside the existing batch vmap.
   - Local probe showed `jax.vmap(make_splash_mha)` can produce a batched kernel pytree in eager Python, but it is not yet proven inside `shard_map`/Pallas execution.

2. Structured metadata without dense `[B,H,S,S]` masks:
   - Derive `MaskInfo` directly from per-example prefix lengths and packed segment metadata.
   - This is probably the right performance direction if dynamic-mask dense materialization is too expensive at 8192/16384.
   - It likely requires bypassing or extending JAX Splash internals rather than only using public mask objects.

3. Fold batch into heads:
   - Represent per-example masks as extra heads and avoid a batch axis in the dynamic mask.
   - Likely awkward because the actual Q/K/V batch dimension remains separate and Splash expects `[H, Q, KV]` mask metadata, not `[B*H, Q, KV]` over a separate batch call.

## Subagent State

Subagent Sagan was spawned for a narrow feasibility investigation:

- Agent id: `019eb96b-ea68-7bb3-8c33-178cfd28796d`
- Question: can a batched `SplashAttentionKernel` pytree from `jax.vmap(make_splash_mha)` be practically passed through the current `shard_map` and per-batch vmap, or is there a Pallas/JAX blocker?
- It had not completed by the handoff checkpoint.

If the next thread cannot access that subagent, redo the investigation locally. The key source files are:

- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py`
- `lib/levanter/src/levanter/layers/attention.py`

## Recommended Next Steps

1. Run `./infra/pre-commit.py --changed-files --fix` after this checkpoint and inspect any edits.
2. Prove or reject the batched-kernel approach with a small `shard_map` reproducer.
3. If batched kernels work, add a `prefix_mask` Splash path that builds dynamic masks only for the packed/data-dependent case.
4. If batched kernels do not work, implement a helper that constructs `MaskInfo` from prefix lengths and possibly segment metadata without dense full masks.
5. Add correctness tests comparing:
   - static prefix length against reference attention,
   - per-example prefix masks,
   - prefix masks plus segment IDs for packed docs.
6. Add a benchmark harness for 8192/16384 that reports compile-including and steady-state timing on v4-8/v5p-8.

## Caveats

- Current static prefix-LM support is useful but not the final requested end state.
- Current Splash segment IDs mask inside visited blocks; they do not provide block skipping for packed segment masks by themselves.
- Dynamic prefix masks are intentionally rejected in Splash lowering for now, so standard reference attention remains the fallback for `prefix_mask`.
