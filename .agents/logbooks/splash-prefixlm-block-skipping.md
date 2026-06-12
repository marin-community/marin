# Splash Prefix-LM Block Skipping: Research Logbook

## Scope
- Goal: Extend Splash Attention support from static causal/local masks plus runtime segment IDs to prefix-LM masks with data-dependent prefix lengths under packing, while preserving or improving block skipping on TPU.
- Primary metrics: Correctness versus dense/reference attention; steady-state attention time on v4-8 and v5p-8 for sequence lengths 8192 and 16384; compile-including time for first-call regressions.
- Constraints: Keep existing causal, full, sliding-window, and segment-ID Splash paths working. Avoid materializing large dense masks on the common static path. Start with v4-8 if v5p availability is poor.
- Experiment issue: https://github.com/marin-community/marin/issues/6332

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

### 2026-06-11 01:45 - Compact prefix-length Splash metadata
- Hypothesis: Prefix-LM block skipping can avoid dense `[B, H, S, S]` masks by constructing Splash `MaskInfo` directly from each example's prefix length. For equal Q/KV block sizes, each query block needs at most two partial mask blocks: the causal boundary and the prefix boundary.
- Command:
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py`
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or sliding_window_mask'`
- Config: Local CPU tests. Compact metadata tested at `S=256`, `block=64`, prefix lengths `0`, `64`, `130`, and `256`.
- Result: Pre-commit passed. Splash helper tests passed with `14 passed`. Attention-mask tests passed with `4 passed`.
- Interpretation: `prefix_lm_forward_mask_info` and `prefix_lm_dkv_mask_info` reconstruct the dense prefix-LM mask while storing only `2 * q_blocks` partial blocks. `_tpu_splash_attention` now has a dynamic-prefix branch that builds a batched `SplashAttentionKernel` from `prefix_lm_mask_infos` and passes it through `shard_map` with leading batch specs.
- Next action: Run a TPU smoke test on v4-8 for dynamic `prefix_lengths`, then benchmark 8192/16384 against dense dynamic masks/reference fallback.

### 2026-06-11 22:10 - v4-8 TPU smoke and forward microbench
- Hypothesis: The compact dynamic prefix-LM metadata path should execute through Pallas on TPU, preserve correctness with segment IDs, and avoid a major steady-state speed regression after the outer training/inference step is jitted.
- Command:
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-prefixlm-v4 allocate --tpu-type v4-8`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-prefixlm-v4 execute --no-sync -- env PYTHONPATH=lib/levanter/tests uv run --project lib/levanter --group test python -c '...'`
- Config:
  - Worker: `marin-tpu-v4-reserved-8-us-central2-b-20260611-0351-9ea73cc7`.
  - Correctness smoke 1: `B=2`, `S=512`, `H=8`, `D=128`, block size `256`, prefix lengths `[0, 300]`.
  - Correctness smoke 2: same shape with batched segment IDs and prefix lengths `[128, 300]`.
  - Jitted microbench: `S=8192`, `H=8`, `D=128`, block size `512`, warm iterations only.
- Result:
  - Dynamic prefix-LM TPU smoke passed against materialized Haliax attention.
  - Dynamic prefix-LM plus batched segment IDs TPU smoke passed against materialized Haliax attention.
  - `B=1` jitted forward median warm: static causal `2.15 ms`; dynamic prefix-LM, prefix length `2048`, `2.18 ms`.
  - `B=2` jitted forward median warm: static causal `3.67 ms`; dynamic prefix-LM, prefix lengths `[0, 2048]`, `4.11 ms`.
  - Non-jitted calls were much slower because they include Python-side metadata construction and dispatch; they are not representative of a jitted Levanter step.
- Interpretation: The batched dynamic-prefix `SplashAttentionKernel` pytree works on v4-8, including with segment IDs for correctness. The jitted steady-state overhead is small for `B=1` and about 12% in the `B=2` smoke. Segment IDs still do not contribute block skipping; this only confirms they remain correct inside visited blocks.
- Next action: Promote the two smokes into TPU-gated tests, add a repeatable 8192/16384 benchmark harness, and then implement segment-ID block skipping from packed segment metadata.

### 2026-06-11 22:20 - Permanent TPU-gated tests
- Hypothesis: The ad hoc v4-8 smokes should be cheap enough to keep as TPU-gated regression tests in `test_attention.py`.
- Command:
  - Local: `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm or tpu_splash_attention_dynamic_prefix_lm'`
  - TPU: `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-prefixlm-v4-tests execute --no-sync -- env PYTHONPATH=lib/levanter/tests uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'test_tpu_splash_attention_dynamic_prefix_lm' -q -n0`
- Config: Same v4-8 worker and correctness shapes as the previous smoke. Pytest xdist was disabled on TPU with `-n0` because libtpu only supports one initializing process per host.
- Result: Local subset passed with `3 passed, 2 skipped`. TPU subset passed with `2 passed, 74 deselected`.
- Interpretation: The dynamic prefix-LM and dynamic prefix-LM plus segment-ID paths are now covered by permanent TPU-gated tests. Default xdist is unsafe for TPU tests in this file; use `-n0` for these targeted runs.
- Next action: Add a repeatable benchmark harness for 8192/16384 and proceed to segment-ID block skipping.

### 2026-06-11 23:40 - Packed prefix-mask Splash path on v4-8
- Hypothesis: Packed prefix-LM can use `AttentionMask.prefix_lm(prefix_mask=..., segment_ids=...)` as the public representation: the per-token prefix mask marks each document's prefix keys after packing, while segment IDs keep prefix visibility local to each packed document. Splash can lower this by building dynamic block metadata for `same_segment(q, kv) & (kv <= q | prefix_mask[kv])`.
- Command:
  - Local metadata tests: `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py`
  - Local public-mask tests: `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'prefix_lm_mask_materializes or tpu_splash_attention_packed_prefix_lm'`
  - TPU smoke: `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-prefixlm-v4 execute -- uv run --project lib/levanter --group test python -m pytest -n0 lib/levanter/tests/test_attention.py -k 'tpu_splash_attention_packed_prefix_lm'`
  - TPU microbench: ad hoc `uv run --project lib/levanter --group test python -c ...` through the same dev TPU session.
- Config:
  - Worker: `marin-tpu-v4-reserved-8-us-central2-b-20260529-1645-faa7f405`, v4-8, `us-central2-b`.
  - Correctness smoke: `B=2`, `S=512`, `H=8`, `D=128`, block size `256`, two packed layouts, per-token prefix masks, compared against materialized Haliax attention.
  - Microbench 8192: `B=1`, `S=8192`, `H=8`, `D=128`, four 2048-token packed docs, 256-token prefix per doc, block size `512`.
  - Microbench 16384: `B=1`, `S=16384`, `H=8`, `D=128`, four 4096-token packed docs, 512-token prefix per doc, block size `512`.
- Result:
  - Kernel metadata reconstruction tests passed locally with `15 passed`.
  - Public mask subset passed locally with `3 passed, 1 skipped`.
  - Packed prefix-mask TPU smoke passed with `1 passed, 77 deselected`.
  - The dynamic packed metadata is shared across heads using a mask-info head dimension of size 1, which Splash broadcasts to all heads. This avoids an H-fold partial-mask payload.
  - v4-8 8192 jitted forward: packed-prefix Splash compile-including `1.024 s`, steady median `1.395 ms`; dense materialized Haliax reference compile-including `10.246 s`, steady median `3.605 ms`; max absolute diff `3.05e-05`.
  - v4-8 16384 Splash-only jitted forward: compile-including `1.315 s`, steady median `4.49 ms`.
- Interpretation: Packed prefix-LM now has a working Splash path with block skipping for off-segment/off-causal blocks and long-shape v4 evidence. It still uses Splash's dynamic-mask representation, so partial-mask payload size grows with the block grid (`q_blocks * kv_blocks`) rather than with only segment boundaries. The shared-head optimization keeps it viable for the tested 8192/16384 cases, but a more compact THD/segment-run representation is still the next performance target.
- Next action: Land this packed-prefix checkpoint, then replace the dynamic dense block payload with segment-run-derived compact partial blocks and extend the same machinery to pure causal segment-ID block skipping.
