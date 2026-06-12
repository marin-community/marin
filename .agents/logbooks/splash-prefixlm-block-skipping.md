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
- Baseline numbers: TBD. The starting point had shared Splash mask lowering helpers and passed a v4-8 Iris smoke test with existing causal/sliding/segment-ID masks.

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

### 2026-06-12 - Pure causal packed segment-ID block skipping
- Hypothesis: The packed prefix-mask lowering can be reused for a first pure causal packed-doc path by generating dynamic Splash metadata for `same_segment(q, kv) & (kv <= q)`, then clearing runtime segment IDs so Splash can skip off-segment blocks directly.
- Command:
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py`
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_attention.py -k 'packed_causal_segment or prefix_lm_mask_materializes or tpu_splash_attention_packed'`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: Local CPU metadata reconstruction at `S=128`, block size `16`, two packed segments; TPU integration smoke added for `B=2`, `S=512`, `H=8`, `D=128`, block size `256`.
- Result: Kernel metadata tests passed with `16 passed`. Attention subset passed locally with `3 passed, 2 skipped`; the new packed causal segment-ID TPU test skipped locally. Changed-file precommit passed.
- Interpretation: Plain causal packed segment IDs now have a block-skipping Splash lowering for batched segment IDs. This is still the dynamic dense-mask metadata representation, so it is a correctness and first-performance path rather than the final compact THD-style segment-run representation.
- Next action: Run the new TPU-gated test on v4-8/v5p when available and benchmark 8192/16384 packed causal layouts against the runtime-segment-ID path.

### 2026-06-12 - Repeatable v4-8 packed mask benchmark harness
- Hypothesis: A checked-in microbench for structured Splash masks will give a stable gate for deciding when to replace dynamic dense-mask metadata with compact segment-run metadata.
- Command:
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-bench allocate --tpu-type v4-8 --timeout 900`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-bench execute -- uv run --project lib/levanter python lib/levanter/scripts/bench/bench_splash_attention_masks.py --seq-len 8192 --block-size 512 --docs-per-sequence 4 --prefix-tokens-per-doc 256 --iterations 5 --warmup 2 --include-dense`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-bench execute --no-sync -- uv run --project lib/levanter python lib/levanter/scripts/bench/bench_splash_attention_masks.py --seq-len 16384 --block-size 512 --docs-per-sequence 4 --prefix-tokens-per-doc 512 --iterations 5 --warmup 2`
- Config:
  - Worker: `marin-tpu-v4-reserved-8-us-central2-b-20260612-0605-e5d92216`, v4-8, `us-central2-b`.
  - `B=1`, `H=8`, `D=128`, BF16 inputs, block size `512`, four equal-length packed docs per sequence.
  - The benchmark records compile-including time and five steady-state forward timings for static causal Splash, packed causal segment Splash, packed prefix-LM Splash, and optional dense materialized Haliax references.
- Result:
  - Added `lib/levanter/scripts/bench/bench_splash_attention_masks.py`.
  - 8192 steady medians: static causal Splash `1.837 ms`; packed causal segment Splash `1.547 ms`; packed prefix-LM Splash `1.510 ms`; packed causal dense reference `3.769 ms`; packed prefix dense reference `3.745 ms`.
  - 8192 compile-including: static causal Splash `0.959 s`; packed causal segment Splash `0.829 s`; packed prefix-LM Splash `0.851 s`; packed causal dense reference `12.542 s`; packed prefix dense reference `9.921 s`.
  - 16384 Splash-only steady medians: static causal `5.809 ms`; packed causal segment `4.480 ms`; packed prefix-LM `4.462 ms`.
  - 16384 Splash-only compile-including: static causal `2.334 s`; packed causal segment `1.167 s`; packed prefix-LM `1.247 s`.
- Interpretation: The dynamic packed metadata path is already a useful long-sequence win on v4-8 for both packed prefix and pure causal segment IDs: about `2.4-2.5x` faster than dense materialized references at 8192, and faster than static causal Splash because cross-document blocks are skipped. The remaining compact segment-run work should target metadata construction/payload scalability rather than proving basic block-skipping value.
- Next action: Run the same harness on v5p-8 when available, then design a compact segment-run `MaskInfo` builder with a fixed partial-block capacity derived from contiguous packed document boundaries.

### 2026-06-12 - Direct blocked metadata for packed masks
- Hypothesis: The first packed prefix/segment implementation can avoid the intermediate dense `[S, S]` dynamic mask by constructing the blocked mask payload directly from blocked segment IDs and prefix masks, while preserving the same Splash `MaskInfo` contract.
- Command:
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/kernels/test_splash_attention.py`
  - `./infra/pre-commit.py --changed-files --fix`
  - `timeout 900 ./infra/pre-commit.py --review --agent-command='env RUST_LOG=error codex exec --ignore-user-config --ephemeral --dangerously-bypass-approvals-and-sandbox'`
- Config: Local CPU metadata tests at `S=128`, block size `16`, with both block-aligned two-document packing and ragged four-document packing (`19/23/41/45` tokens).
- Result:
  - `packed_prefix_lm_mask_infos` and `packed_causal_segment_mask_infos` now build `[q_blocks, kv_blocks, block_q, block_kv]` mask blocks directly from packed segment IDs.
  - Removed the private dense dynamic-mask metadata path for packed masks.
  - Splash helper tests passed with `18 passed`.
  - Changed-file precommit passed.
  - Lint review reported no findings.
- Interpretation: This is not full THD-style segment-run metadata yet, because dynamic partial-mask blocks are still allocated for every block coordinate. It does remove the avoidable dense mask materialization step and adds ragged packed-doc coverage, which is a concrete step toward compact segment-run metadata.
- Next action: Re-benchmark on TPU after this patch, then replace per-block partial payloads with segment-run-derived partial blocks where possible.

### 2026-06-12 - v5p-8 packed mask benchmark harness
- Hypothesis: The packed mask Splash wins observed on v4-8 should hold on v5p-8 with the scoped VMEM setting used for v5-class TPU microbenches.
- Command:
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-v5p-bench allocate --tpu-type v5p-8 --timeout 900`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-v5p-bench execute --env 'LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000' -- uv run --project lib/levanter python lib/levanter/scripts/bench/bench_splash_attention_masks.py --seq-len 8192 --block-size 512 --docs-per-sequence 4 --prefix-tokens-per-doc 256 --iterations 5 --warmup 2 --include-dense`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-v5p-bench execute --no-sync --env 'LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000' -- uv run --project lib/levanter python lib/levanter/scripts/bench/bench_splash_attention_masks.py --seq-len 16384 --block-size 512 --docs-per-sequence 4 --prefix-tokens-per-doc 512 --iterations 5 --warmup 2`
  - `uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name dlwh-splash-mask-v5p-bench release`
- Config:
  - Worker: `marin-tpu-v5p-preemptible-8-us-central1-20260612-0636-2fe5e706`, IP `10.128.0.130`, zone `us-central1-a`.
  - Env: `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`.
  - Benchmark-reported device: `TPU v5`, `num_devices=4`.
  - `B=1`, `H=8`, `D=128`, BF16 inputs, block size `512`, four equal-length packed docs per sequence.
- Result:
  - 8192 steady medians: static causal Splash `1.204 ms`; packed causal segment Splash `1.051 ms`; packed prefix-LM Splash `1.079 ms`; packed causal dense reference `1.850 ms`; packed prefix dense reference `1.861 ms`.
  - 8192 compile-including: static causal Splash `1.155 s`; packed causal segment Splash `0.834 s`; packed prefix-LM Splash `0.837 s`; packed causal dense reference `15.056 s`; packed prefix dense reference `8.947 s`.
  - 16384 Splash-only steady medians: static causal `3.297 ms`; packed causal segment `2.683 ms`; packed prefix-LM `2.699 ms`.
  - 16384 Splash-only compile-including: static causal `2.298 s`; packed causal segment `0.961 s`; packed prefix-LM `1.009 s`.
  - Release verified: no active local dev TPU session after release; Iris holder job ended `JOB_STATE_KILLED` with `Terminated by user`.
- Interpretation: The packed block-skipping path also wins on v5p-8. At 8192 it is about `1.7x` faster than dense materialized references and modestly faster than static causal Splash because cross-document blocks are skipped. At 16384 the same trend holds against static causal Splash.
- Next action: Re-run v5p after the direct blocked metadata patch is pushed, then check v5e/v6e compatibility and start a bounded segment-run metadata design.
