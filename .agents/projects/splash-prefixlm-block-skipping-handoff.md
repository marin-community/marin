# Splash Prefix-LM Block Skipping Handoff

## Current Branch

- Branch: `codex/splash-prefixlm-block-skipping`
- Base: current `origin/main` with the shared Splash Attention mask lowering in tree.
- Research logbook: `.agents/logbooks/splash-prefixlm-block-skipping.md`
- Experiment issue: https://github.com/marin-community/marin/issues/6332

## Goal

Build on the shared Splash Attention mask lowering to support:

- Prefix-LM masks in Splash Attention.
- Data-dependent prefix lengths under packing.
- Block skipping when static or structured mask information makes blocks empty.
- Later: block skipping for segment-ID masks, moving toward THD-style packed attention without slowdown.

Target hardware for validation and tuning:

- First: v5p-8 and v4-8, with v4-8 acceptable if v5p availability is poor.
- Then: v5e and v6e.
- Shapes where skipping should matter: sequence lengths 8192 and 16384, usually with 512-token blocks.

## What Changed So Far

This checkpoint implements the first slices: static prefix-LM masks, reference semantics for batch-dependent prefix lengths, compact Splash metadata for dynamic prefix lengths, and a packed prefix-mask Splash path for `prefix_mask + segment_ids`.

Files changed:

- `lib/levanter/src/levanter/layers/attention_mask.py`
  - Adds `AttentionMask.prefix_lm(...)`.
  - Adds `prefix_length`, `prefix_lengths`, and `prefix_mask` fields.
  - Materializes prefix-LM as `causal_or_prefix`, then applies explicit and segment masks.
  - Sliding-window prefix-LM semantics are: prefix keys remain visible to every query, while non-prefix keys are causal and local-window constrained.

- `lib/levanter/src/levanter/kernels/pallas/splash_attention.py`
  - Adds `prefix_length` and `dynamic_prefix` to `SplashAttentionMaskSpec`.
  - Adds `PrefixMask`, a sliceable Splash mask for static prefix keys.
  - Lowers static prefix-LM to `LogicalOr(causal_or_local_mask, PrefixMask)`, preserving Splash static mask processing and block skipping.
  - Adds compact `prefix_lm_mask_infos(...)` metadata for dynamic prefix lengths. The representation stores a full block grid but only `2 * q_blocks` partial blocks per example.
  - Adds `packed_prefix_lm_mask_infos(...)` for packed prefix masks represented as `same_segment(q, kv) & (kv <= q | prefix_mask[kv])`.
  - Uses a mask-info head dimension of size 1 for packed prefix masks so Splash broadcasts the metadata across heads instead of storing one dynamic partial-mask payload per head.

- `lib/levanter/src/levanter/layers/attention.py`
  - Passes `prefix_length` / `has_prefix_mask` from `AttentionMask` into the Splash mask spec.
  - Builds one dynamic-prefix `SplashAttentionKernel` per batch example with `jax.vmap`, then passes the batched kernel pytree through `shard_map`.
  - For `prefix_mask`, builds packed prefix metadata from the prefix mask and segment IDs, then omits the redundant runtime `segment_ids` argument because the block metadata already encodes same-segment filtering.

- `lib/levanter/tests/test_attention.py`
  - Adds reference materialization tests for prefix-LM, prefix-LM plus sliding window, and per-batch dynamic prefix lengths.
  - Adds reference materialization and TPU-gated parity tests for packed prefix masks with segment IDs.

- `lib/levanter/tests/kernels/test_splash_attention.py`
  - Adds static Splash prefix-LM lowering tests.
  - Adds compact forward/dKV `MaskInfo` reconstruction tests for dynamic prefix lengths.
  - Adds packed prefix-mask metadata reconstruction tests, including empty/full/partial block assertions.
  - Extends test mask materialization to handle `LogicalOr`.

## Validation Already Run

Both commands passed locally:

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py
```

Result after packed prefix metadata: `15 passed`.

```bash
uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or sliding_window_mask'
```

Result after packed prefix mask materialization: focused subset `3 passed, 1 skipped`.

Additional local probe:

- CPU interpret-mode `jax.vmap(make_splash_mha)` over `[B, H, 128, 128]` dynamic masks succeeded.
- Passing the batched `SplashAttentionKernel` pytree through a one-device `shard_map` also succeeded when every kernel metadata leaf used a leading batch partition spec.
- This suggests the remaining hard part is compact metadata construction, not the Pallas call interface.

TPU validation on Iris v4-8:

- Worker: `marin-tpu-v4-reserved-8-us-central2-b-20260611-0351-9ea73cc7`.
- Dynamic prefix-LM smoke passed on TPU for `B=2`, `S=512`, `H=8`, `D=128`, block size `256`, prefix lengths `[0, 300]`, compared against materialized Haliax attention.
- Dynamic prefix-LM plus batched segment IDs smoke passed on TPU for `B=2`, `S=512`, `H=8`, `D=128`, block size `256`, prefix lengths `[128, 300]`, compared against materialized Haliax attention.
- Jitted forward microbench on v4-8, `B=1`, `S=8192`, `H=8`, `D=128`, block size `512`: static causal median warm `2.15 ms`; dynamic prefix-LM with prefix length `2048` median warm `2.18 ms`.
- Jitted forward microbench on v4-8, `B=2`, `S=8192`, `H=8`, `D=128`, block size `512`: static causal median warm `3.67 ms`; dynamic prefix-LM with prefix lengths `[0, 2048]` median warm `4.11 ms`.
- The two focused smokes are now permanent TPU-gated tests in `lib/levanter/tests/test_attention.py`. Run them on TPU with pytest xdist disabled: `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'test_tpu_splash_attention_dynamic_prefix_lm' -q -n0`.
- Packed prefix-mask TPU smoke passed on v4-8 for `B=2`, `S=512`, `H=8`, `D=128`, block size `256`, two packed layouts, compared against materialized Haliax attention. Run it with: `uv run --project lib/levanter --group test python -m pytest -n0 lib/levanter/tests/test_attention.py -k 'tpu_splash_attention_packed_prefix_lm'`.
- Packed prefix-mask v4-8 microbench:
  - `B=1`, `S=8192`, `H=8`, `D=128`, four 2048-token packed docs, 256-token prefix per doc, block size `512`: Splash compile-including `1.024 s`, steady median `1.395 ms`; dense materialized Haliax reference compile-including `10.246 s`, steady median `3.605 ms`; max abs diff `3.05e-05`.
  - `B=1`, `S=16384`, `H=8`, `D=128`, four 4096-token packed docs, 512-token prefix per doc, block size `512`: Splash compile-including `1.315 s`, steady median `4.49 ms` (Splash-only timing).

## Important Design State

JAX Splash has two mask processing paths:

- Static masks: `MultiHeadMask` / `Mask` objects processed by `process_mask`.
- Dynamic masks: `jax.Array` of shape `[heads, q, kv]` processed by `process_dynamic_mask`.

`process_dynamic_mask` still computes `block_mask` and `data_next`, so it can skip fully empty blocks. However, it has no batch axis. Levanter’s current Splash path creates one `SplashAttentionKernel` outside `shard_map`, then vmaps the kernel call over batch, only varying `q`, `k`, `v`, `segment_ids`, and sinks.

Dynamic prefix lengths are represented by a batched `SplashAttentionKernel` pytree. Packed prefix masks are represented by `prefix_mask + segment_ids` and now get Splash block metadata that encodes same-segment filtering, so off-document/off-causal blocks can be skipped. Pure causal packed segment-ID masks also have a first block-skipping path for batched segment IDs, still using dynamic dense-mask metadata rather than a final compact segment-run representation.

## Open Technical Question

Need decide the compact THD/segment-run representation for pure segment-ID masks and for improving packed prefix masks further.

Main options:

1. Structured metadata without dense `[B,H,S,S]` masks:
   - Derive `MaskInfo` directly from per-example prefix lengths and packed segment metadata.
   - This is probably the right performance direction if dynamic packed-mask partial payloads become too expensive at larger packed layouts.
   - It likely requires bypassing or extending JAX Splash internals rather than only using public mask objects.

2. Fold batch into heads:
   - Represent per-example masks as extra heads and avoid a batch axis in the dynamic mask.
   - Likely awkward because the actual Q/K/V batch dimension remains separate and Splash expects `[H, Q, KV]` mask metadata, not `[B*H, Q, KV]` over a separate batch call.

## Key Source Files

- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
- `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py`
- `lib/levanter/src/levanter/layers/attention.py`

## Recommended Next Steps

1. Add a checked-in benchmark harness for 8192/16384 that reports compile-including and steady-state timing on v4-8/v5p-8.
2. Implement pure segment-ID block skipping, probably by deriving block metadata from packed segment runs or THD metadata.
3. Replace packed prefix-mask dynamic partial payloads with a compact segment-run representation if v5p/v5e/v6e timing or memory pressure warrants it.
4. Broaden TPU validation to v5p-8, then v5e/v6e.

## Caveats

- Dynamic prefix-length support passed focused v4-8 smokes and now has permanent TPU-gated tests, but those tests must run single-process (`-n0`) because libtpu cannot be initialized by pytest-xdist workers on one host.
- Current Splash segment IDs mask inside visited blocks; they do not provide block skipping for packed segment masks by themselves.
- Packed `prefix_mask + segment_ids` uses Splash dynamic mask metadata and skips empty blocks, but its partial-mask payload still grows with the block grid. It is not yet the final compact THD-style representation.
