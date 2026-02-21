# Fused CE GPU Pallas kernel – GPU branch log (2026-02-18)

## What I changed
- Added a true GPU Pallas forward path:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - Implements batched `x @ W` streaming in-logit space over V-blocks and writes per-row loss/LSE.
- Updated tuned block-size lookup for NVIDIA GPUs:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  - Added `NVIDIA`, `NVIDIA A100`, and `NVIDIA H100` keys and a `tiny` bucket.
- Added non-multiple-shape coverage for GPU in tests:
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
- Added GPU-only block-size sweep bench:
  - `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_gpu_block_sweep.py`

## Notes
- Current kernel assumes one batch block per program and streams over V-tile(s) internally; there is no GPU-specific backward custom VJP yet, so gradients flow through forward autodiff.
- Hidden dimension is loaded as one full `H` block per program in this first pass; `h_block_size` is currently accepted for API compatibility but is not a tunable axis yet.
- Inputs are padded on `B` and `V` to block multiples; out-of-bounds labels are padded with `-1` and masked out of the per-row label extraction path.

## Pending / next
- Run the new sweep script on GB10 GPU and update tuned block sizes with measured best `(b_block_size, v_block_size)` pairs.
- Capture speedup vs `xla` for the target shape(s):
  - `batch=64,pos=2048,embed=1024,vocab=128256` (and any real production shape).
- Compare with TPU/CUDA roofline baselines once measured to decide whether backward/precision tuning is needed.

## 2026-02-18 safety updates
- Added GB10-focused launch guards in `pallas_gpu.py` to avoid machine-locking failure modes:
  - require `h_block_size` and `v_block_size` to be power-of-two,
  - reject weight tiles that exceed GB10 shared-memory budget (`101,376` bytes),
  - reject very large static dot-tile counts (`num_h_blocks * num_v_blocks > 512`) that trigger pathological compile time on current lowering.
- Updated API default dispatch on GB10 (`api.py`) to prioritize `xla` first, with `pallas_gpu` still available explicitly.
- Added per-implementation precision resolution in `api.py`:
  - default/API precision remains `None`,
  - `pallas_gpu` gets `Precision.HIGHEST` when precision is unspecified.
- Updated GPU sweep script to support `--h-block-sizes` and `--max-dot-tiles` so tuning runs can stay bounded and interactive on fragile GB10 nodes.
- Updated XLA path in `xla.py` to use an exact dense fallback when `v_block_size >= vocab` to reduce tiny-shape gradient drift in default tests.
- Added GB10-specific tuned buckets in `tuned_block_sizes.py`:
  - device key: `NVIDIA GB10`
  - new bucket: `small-h-small-vocab` for `B in [2048,8192], H in [1,256], V in [4096,16384]`.

## 2026-02-18 measured GB10 results (bounded sweep)
Shape:
- `batch=16,pos=256,embed=128,vocab=8192,input=bfloat16,accum=float32`
- `steps=1,warmup=0`

Baseline:
- `xla`: `steady_s=0.001345`, `tokens_per_s=3,045,706`

Pallas candidates:
- `b=16,h=64,v=128`: `steady_s=0.000791`, `+69.99% vs xla`
- `b=16,h=64,v=256`: `steady_s=0.000622`, `+116.32% vs xla`
- `b=16,h=128,v=128`: `steady_s=0.000843`, `+59.59% vs xla`
- `b=16,h=128,v=256`: `steady_s=0.000984`, `+36.69% vs xla`
- `b=32,h=64,v=128`: `steady_s=0.000589`, `+128.14% vs xla` (best in this sweep)
- `b=32,h=64,v=256`: `steady_s=0.000645`, `+108.56% vs xla`
- `b=32,h=128,v=128`: `steady_s=0.000662`, `+103.13% vs xla`
- `b=32,h=128,v=256`: `steady_s=0.000818`, `+64.41% vs xla`

Full target sanity check:
- `batch=64,pos=2048,embed=1024,vocab=128256`
- `xla`: `steady_s=1.019020`, `tokens_per_s=128,625.543`
- `pallas_gpu b=16,h=64,v=512`: fails fast by guard with
  - `dot_tiles=4016 > 512` (avoids long compile/machine lock).

## Suggested command
```bash
python -m lib/levanter.scripts.bench.bench_fused_cross_entropy_loss_gpu_block_sweep \
  --batch 64 --pos 2048 --embed 1024 --vocab 128256 \
  --b-block-sizes 128,256,512 --v-block-sizes 256,512,1024,2048 --steps 5 --warmup 2
```

## 2026-02-19 continued tuning log (GB10)

### Scope and objective
- Objective for this pass: make `implementation="pallas_gpu"` faster than `xla` with acceptable numerics on GB10, then improve `fwd+bwd` throughput.
- Priority was empirical throughput and numerical behavior, not final kernel cleanliness.

### Key code changes made in this pass
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  - Removed the implicit `precision=HIGHEST` override for `pallas_gpu` dispatch when precision is unspecified.
  - Reason: this hurt GB10 fallback performance in practice.
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - Added GB10 BF16 routing logic:
    - For moderate output sizes, route to full-matmul/reference path (fast and numerically tight on GB10).
    - For larger shapes, route to XLA streaming path with GB10-specific `v_block` tuning.
  - Added helper policy for fallback block sizes by batch regime.
  - Final policy in this pass for GB10 BF16 and `V >= 65536`:
    - `B >= 8192` -> `v_block=3584`
    - `B >= 4096` -> `v_block=3072`
    - `B >= 1024` -> `v_block=2048`
  - Kept Pallas tiled path for non-fallback regimes; this log’s speed wins below are primarily from better GB10 dispatch/routing.
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  - Added GB10 fallback entry so unsupported default blocks are avoided when no tuned bucket matches.

### Blackwell/Pallas exploration notes
- Attempted to follow the Blackwell Pallas matmul path (`jax.experimental.pallas.mosaic_gpu` / `blackwell_matmul_mgpu`) for phase-1 kernel intuition.
- Findings on this machine:
  - `jax.experimental.pallas.gpu` tutorial path is not available in this environment; available path is `mosaic_gpu`.
  - Running shipped Blackwell `tcgen05` kernel path fails on this GB10 target (`sm_121a`) with unsupported instruction/features (`tcgen05.*`, `.cta_group::1`, etc.) in current toolchain stack.
  - Practical conclusion for now: Blackwell tutorial concepts are still useful for intuition, but not directly deployable here as-is.

### Numerical debugging conclusions
- Large earlier Pallas-vs-XLA diffs were real in BF16 custom tile path.
- Reduction method (associative scan vs sequential merge) was not the root cause; phase-1 matmul tile behavior dominated error.
- For fallback-routed paths used here, numerics are tight vs XLA/reference (typical max diff in the `1e-5` to `1e-4` range depending on shape/run).

### Benchmarks captured in this pass

#### Forward-only, target-ish large shape
- Shape: `B=8192, H=1024, V=128256` (GB10, BF16 input, `dtype=float32`)
- After GB10 fallback tuning:
  - `xla`: `~73.24 ms`
  - `pallas_gpu`: `~49.21 ms`
  - `xla/pallas`: `~1.49x`
  - observed diff stats: mean `~2.93e-6`, max `~6.10e-5`

#### Effective large batch without full logits materialization
- Effective shape: `B=131072, H=1024, V=128256`
- Method: chunked run as `16 x B=8192` microbatches to avoid direct `B x V` allocation.

Forward-only aggregate:
- `xla_total_ms`: `~1299.39`
- `pallas_total_ms`: `~962.69`
- `xla/pallas`: `~1.35x`

Forward+backward aggregate (loss mean, grads wrt `x` and `w`):
- Earlier tuned run: `xla/pallas ~1.075x`
- After `v_block=3072` tuning: `xla/pallas ~1.10x`
- Final controlled run with piecewise fallback policy: `xla/pallas ~1.11x`
- Example controlled result (same chunk set):
  - `xla_ms`: `5192.09`
  - `pallas_ms`: `4678.45`
  - `xla/pallas`: `1.1098x`
  - aggregate loss/checksum remained closely aligned (`loss_abs_diff ~3.05e-5`).

#### Fwd+bwd speedups by chunk batch with final piecewise policy
- Shape family: `H=1024, V=128256`
- `B=2048`: `xla/pallas ~1.163x`
- `B=4096`: `xla/pallas ~1.080x`
- `B=8192`: `xla/pallas ~1.096x`

### Commits made in this pass
- `45db4a475` `Tune GB10 fused cross-entropy routing and fallback blocks`
- `05d9325c5` `Tune GB10 large-batch XLA fallback for backward throughput`
- `252e32560` `Refine GB10 fallback v-blocks for backward throughput`

### Current status
- `pre-commit` passing after each tuning step.
- On GB10 for the measured regimes, current `pallas_gpu` route is faster than `xla` with acceptable numerics, including `fwd+bwd`.
- Remaining gap: speedup in `fwd+bwd` is real but smaller than forward-only; further gains likely require deeper backward-path work (or more direct fused backward kernels) rather than only fallback block retuning.

## 2026-02-19 custom backward implementation update

### What was implemented
- Added a custom VJP path for:
  - `linear_softmax_cross_entropy_loss_pallas_gpu` in
    `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`.
- Structure:
  - Forward path remains routed by existing GB10 policy (reference / XLA-streaming / tiled Pallas).
  - Backward now has a GB10-specialized streaming kernel (`_backward_streaming_from_lse`) for the large-batch BF16 fallback regime.
  - For other regimes/devices, backward falls back to `jax.vjp` of the implementation path.
- New backward math:
  - Uses saved per-row `lse` from forward.
  - Streams over vocab blocks, computes `probs = exp(logits - lse)`, applies cotangents for `(loss, lse)`, subtracts label contribution, applies optional soft-cap derivative, accumulates `grad_x` and per-block `grad_w`.
  - Accumulates in `float32`, returns gradients in primal dtypes (`x.dtype`, `w.dtype`) to match existing behavior.

### Why this aligns with Blackwell-inspired learnings (within current constraints)
- Could not use direct `tcgen05` Blackwell path on this GB10 stack due toolchain instruction support limits.
- Applied the transferable principles anyway:
  - streaming/reduction over V (avoid giant intermediates),
  - stable normalization with saved `lse`,
  - block-wise accumulation and explicit epilogue-style gradient updates,
  - route-specific tuning (`v_block`) by workload regime.

### Benchmarks after custom backward

#### Single chunk `fwd+bwd`
- Shape: `B=8192, H=1024, V=128256`
- `xla`: `~319.17 ms`
- `pallas_gpu`: `~226.27 ms`
- speedup: `xla/pallas ~1.41x`
- `loss_diff`: `0.0` in sampled run

#### Effective large batch `fwd+bwd` (chunked)
- Effective `B=131072` as `16 x B=8192`, `H=1024`, `V=128256`
- `xla_ms`: `~5180.49`
- `pallas_ms`: `~3663.89`
- speedup: `xla/pallas ~1.414x`
- aggregate diffs stayed small:
  - `loss_abs_diff ~3.05e-05`
  - checksum diff in low `1e-06` range

#### Additional sampled numerical checks (smaller shape)
- Shape: `B=256, H=1024, V=65536` (`fwd+bwd`)
- `loss_diff`: `0.0`
- `gx` diff: mean `~3.8e-06`, max `~1.22e-04`
- sampled `gw` diff: `0.0` in tested slice

### Notes
- This pass materially improves backward throughput for the target GB10 large-batch regime.
- Next optimization frontier is likely deeper kernel-level scheduling/fusion (once toolchain supports the relevant Blackwell instruction path), not just fallback block retuning.

## 2026-02-19 follow-up: decoupled backward block tuning

### Change summary
- Refined custom backward tuning in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`:
  - Decoupled backward `v_block` from forward fallback `v_block`.
  - Forward fallback (XLA streaming) remains:
    - `B >= 8192` -> `3584`
    - `B >= 4096` -> `3072`
    - `B >= 1024` -> `2048`
  - Custom backward streaming now uses:
    - `B >= 8192` -> `8192`
    - `B >= 1024` -> `6144`
  - (for GB10 BF16 with `V >= 65536`, outside full-matmul fallback regime).

### Why
- Direct backward-kernel sweeps showed larger vocab tiles are better for custom backward than for forward fallback.
- Using one shared block-size policy left backward performance on the table.

### Backward-only sweep highlights (`H=1024, V=128256`)
- `B=8192`: best around `8192` (and close `7168/6144`), notably better than `3584`.
- `B=4096`: best around `6144` (better than `3072/3584/4096`).
- `B=2048`: `8192` slightly best in sampled runs; `6144` close.
- `B=1024`: `6144` best in sampled runs.

### End-to-end after decoupling (`fwd+bwd`)
- `B=2048, H=1024, V=128256`:
  - `xla`: `~96.65 ms`
  - `pallas_gpu`: `~65.95 ms`
  - `xla/pallas`: `~1.47x`
- `B=4096, H=1024, V=128256`:
  - `xla`: `~169.40 ms`
  - `pallas_gpu`: `~119.01 ms`
  - `xla/pallas`: `~1.42x`
- `B=8192, H=1024, V=128256`:
  - `xla`: `~320.70 ms`
  - `pallas_gpu`: `~226.83 ms`
  - `xla/pallas`: `~1.41x`

Effective large batch (`16 x 8192` chunks, `B=131072`, `H=1024`, `V=128256`):
- `xla_ms`: `~5204.35`
- `pallas_ms`: `~3499.87`
- `xla/pallas`: `~1.49x`
- aggregate loss/checksum diffs remained small (loss diff on order `1e-5`).

Additional validation at larger chunk size:
- Single chunk `B=16384, H=1024, V=128256` (`fwd+bwd`):
  - `xla`: `~615.30 ms`
  - `pallas_gpu`: `~419.70 ms`
  - `xla/pallas`: `~1.47x`
  - sampled loss/checksum diffs: `0.0`
- Effective `B=131072` as `8 x 16384` chunks:
  - `xla_ms`: `~4953.00`
  - `pallas_ms`: `~3367.07`
  - `xla/pallas`: `~1.47x`
- aggregate diffs remained small (`loss_diff ~3.05e-5`).

Custom-backward `logit_soft_cap` check:
- Shape: `B=512, H=1024, V=65536`, `logit_soft_cap=2.0`, `fwd+bwd`
- `loss_diff`: `0.0`
- `gx` diff: mean `~3.66e-08`, max `~3.05e-05`
- sampled `gw` diff: mean `~9.33e-14`, max `~1.86e-09`

### Forward-only check (unchanged objective sanity)
- `B=8192, H=1024, V=128256`:
  - `xla`: `~71.88 ms`
  - `pallas_gpu`: `~46.71 ms`
  - `xla/pallas`: `~1.54x`

## 2026-02-19 follow-up: BF16 tensor-core backward matmul path

### Change
- Updated custom GB10 streaming backward in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`:
  - Keep probability/softmax math in `float32`.
  - Cast `dlogits` to activation dtype (`bf16` on target path) before the two backward GEMMs:
    - `grad_x`: `dlogits @ W^T`
    - `grad_w`: `X^T @ dlogits`
- Rationale:
  - Prior path used mixed-input GEMMs (`bf16 x float32`), which underutilized fast tensor-core kernels on GB10.
  - New path restores tensor-core-friendly GEMMs while preserving stable softmax math in `float32`.

### Target performance impact (`precision=None`)
- Single chunk `fwd+bwd` (`B=8192, H=1024, V=128256`):
  - `xla`: `~320.91 ms`
  - `pallas_gpu`: `~188.22 ms`
  - `xla/pallas`: `~1.70x`
- Effective large batch (`16 x 8192` chunks, `B=131072`):
  - `xla_ms`: `~5225.48`
  - `pallas_ms`: `~3036.75`
  - `xla/pallas`: `~1.72x`
  - `loss_diff`: `0.0`
  - checksum diff remained small (`~7.30e-2` aggregate over all chunks).

### Backward kernel microbenchmark (`B=8192, H=1024, V=128256, v_block=7168`)
- Old custom backward: `~179.81 ms`
- New custom backward: `~149.93 ms`
- Relative: `~1.20x` faster (`~16.6%` lower time).

### Post-change backward block retune check
- `B=8192`: `v_block=7168` remains best among `{6144, 7168, 8192}`.
- `B=2048`: `v_block=7168` slightly better than `{6144, 8192}`.
- `B=1024`: `v_block=6144` remains marginally best.
- Kept current piecewise policy (`>=8192 -> 7168`, `>=1024 -> 6144`) to preserve best behavior across sampled regimes.

### Numerical parity checks after change
- Shape: `B=512, H=1024, V=65536`, `fwd+bwd`, `precision=None`
- `logit_soft_cap=None`:
  - `loss_diff`: `~1.53e-5`
  - `gx` diff: mean `~1.90e-6`, max `~6.10e-5`
  - `gw` diff: mean `~1.82e-14`, max `~3.73e-9`
- `logit_soft_cap=2.0`:
  - `loss_diff`: `~9.54e-7`
  - `gx` diff: mean `~4.01e-8`, max `~3.05e-5`
  - `gw` diff: mean `~1.16e-13`, max `~7.45e-9`
- Target shape parity sample (`B=8192, H=1024, V=128256`, `logit_soft_cap=None`):
  - `loss_diff`: `0.0`
  - `gx` diff: mean `~1.54e-7`, max `~7.63e-6`
  - sampled `gw` diff (`64x64` random indices): `0.0`

### Note on precision mode
- With `precision=jax.lax.Precision.HIGHEST`, this workload can still be slower than XLA.
- Current tuning target and wins are for the default dispatch regime (`precision=None`), which is the practical training path used in these benchmarks.

## 2026-02-19 follow-up: replace scatter label update with dense one-hot path

### Change
- Updated the custom backward label subtraction in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`:
  - Old: per-row indexed scatter (`dlogits.at[row_ids, safe_idx].add(...)`).
  - New: dense one-hot mask (`jax.nn.one_hot`) with masked subtraction.
- Kept all softmax/logsumexp math and gradient flow unchanged; only the label update primitive changed.

### Why this helps on GB10
- The scatter path was introducing a major non-matmul hotspot in each vocab tile.
- The dense one-hot formulation maps better to vectorized fused kernels on this workload, and avoids indexed update overhead.

### Backward-only microbenchmark impact (`logit_soft_cap=None`, `precision=None`)
- `B=8192, H=1024, V=128256, v_block=7168`:
  - scatter path: `~146.97 ms`
  - one-hot path: `~111.62 ms`
  - relative: `~1.32x` faster.
- Additional shapes (same `H,V`, tuned `v_block` per shape) showed consistent wins:
  - `B=1024`: `~1.20x`
  - `B=2048`: `~1.19x`
  - `B=4096`: `~1.24x`
  - `B=8192`: `~1.34x`

### End-to-end `fwd+bwd` impact after one-hot change
- Single-chunk:
  - `B=2048`: `xla ~95.52 ms`, `pallas ~51.48 ms`, `~1.86x`
  - `B=4096`: `xla ~170.12 ms`, `pallas ~84.52 ms`, `~2.01x`
  - `B=8192`: `xla ~322.52 ms`, `pallas ~153.31 ms`, `~2.10x`
- Effective large batch (`16 x 8192` chunks, `B=131072`):
  - `xla_ms`: `~5214.08`
  - `pallas_ms`: `~2634.95`
  - `xla/pallas`: `~1.98x`
  - `loss_diff`: `0.0`

### Tuning update
- Added an explicit mid-batch backward tile tier:
  - `B >= 8192 -> 7168`
  - `B >= 2048 -> 7168`
  - `B >= 1024 -> 6144`
- This preserves `B=1024` behavior while improving `B=2048/4096` in sampled runs.

## 2026-02-19 final retune pass (current best)

### Parameter adjustments
- Forward fallback retuned for large batch:
  - `GB10_XLA_STREAMING_V_BLOCK_BATCH_8K`: `3584 -> 3072`.
- Backward policy now includes explicit `>=2048` tier (same tile as `>=8192`):
  - `B >= 8192 -> 7168`
  - `B >= 2048 -> 7168`
  - `B >= 1024 -> 6144`

### End-to-end results (`fwd+bwd`, `precision=None`, current code)
- `B=2048, H=1024, V=128256`:
  - `xla`: `~96.54 ms`
  - `pallas_gpu`: `~49.15 ms`
  - `xla/pallas`: `~1.96x`
- `B=4096, H=1024, V=128256`:
  - `xla`: `~170.62 ms`
  - `pallas_gpu`: `~86.13 ms`
  - `xla/pallas`: `~1.98x`
- `B=8192, H=1024, V=128256`:
  - `xla`: `~321.67 ms`
  - `pallas_gpu`: `~156.03 ms`
  - `xla/pallas`: `~2.06x`
- Effective `B=131072` as `16 x 8192` chunks:
  - `xla_ms`: `~5223.43`
  - `pallas_ms`: `~2519.79`
  - `xla/pallas`: `~2.07x`
  - `loss_diff`: `0.0`
  - Note: `loss_diff=0.0` is expected in this GB10 BF16 configuration because
    the pallas-gpu route intentionally reuses the same XLA forward implementation
    and only overrides backward via `custom_vjp`.

## 2026-02-19 opt-in experiment: native GB10 Pallas forward

### Code switch
- Added a GB10-native forward opt-in in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`:
  - env var: `LEVANTER_PALLAS_GPU_GB10_NATIVE_FORWARD=1`
  - requires explicit `block_sizes=BlockSizes(...)` (fails fast otherwise)
- Default behavior is unchanged (hybrid path remains default).

### Bounded exercise (no large tile sweeps)
- Shape: `B in {2048,4096}`, `H=1024`, `V=65536`, `fwd+bwd`
- Native tiles tried:
  - `b=128,h=128,v=256`
  - `b=128,h=64,v=512`

Results:
- `B=2048`:
  - `xla`: `~48.06 ms`
  - `pallas native (128,128,256)`: `~358.20 ms` (`~0.13x` vs xla)
  - `pallas native (128,64,512)`: `~446.53 ms` (`~0.11x` vs xla)
- `B=4096`:
  - `xla`: `~86.35 ms`
  - `pallas native (128,128,256)`: `~698.65 ms` (`~0.12x` vs xla)
  - `pallas native (128,64,512)`: `~886.46 ms` (`~0.10x` vs xla)

Numerics on this native-forward config (`B=2048,H=1024,V=65536`, `b=128,h=128,v=256`):
- `loss_diff`: `~1.23e-2`
- `gx` diff: mean `~8.68e-5`, max `~1.68e-3`
- sampled `gw` diff: mean `~5.65e-6`, max `~5.19e-4`

Conclusion:
- For target-like large-hidden/vocab regimes, native GB10 Pallas forward is currently
  both slower and less numerically aligned than the hybrid path.
- Keep native-forward as opt-in-only for experimentation.
- Keep hybrid default (XLA forward + custom backward), which remains the best measured path.

### Small-shape sanity
- Shape: `B=4096,H=128,V=8192`
- Forward-only:
  - `xla`: `~1.301 ms`
  - `native`: `~1.137 ms`
  - `~1.15x` speedup
- Full `fwd+bwd` at same shape:
  - near parity (`~1.00x`).

## 2026-02-20 backward rewrite + FA retune (GB10)

### Key implementation change
- File: `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
- Rewrote custom backward accumulation in `_backward_streaming_from_lse`:
  - old: `lax.fori_loop` carrying full `grad_x` and full `grad_w` with per-iteration `dynamic_update_slice`
  - new: `lax.scan` carrying only `grad_x`, emitting `grad_w_block` per vocab tile, then one final reshape/transpose to materialize `grad_w`
- The attempted scatter-based label update in backward regressed throughput and was reverted.

### Why this helped
- Removing full `grad_w` from loop carry significantly reduced backward update overhead at large `V`.
- Backward still dominates total step time at target shape, so this structural change produced a measurable end-to-end gain.

### Target shape benchmark (fwd+bwd)
Shape:
- `B=8192, H=1024, V=128256`, BF16 inputs, `dtype=float32`

Measured after rewrite:
- `xla`: `~321.13 ms`
- `pallas_gpu hybrid` (XLA fwd + custom bwd): `~144.97 ms`
- `pallas_gpu FA-optin` (`b=1024,h=32,v=1024`): `~143.15 ms`

Speedups:
- `xla / hybrid`: `~2.22x`
- `xla / FA-optin`: `~2.24x`

### Effective large batch benchmark (fwd+bwd)
Method:
- emulate `B=131072` as `16 x B=8192` microbatches (to avoid direct `B x V` materialization)

Measured totals:
- `xla`: `~5151.71 ms`
- `pallas_gpu FA-optin (b=1024,h=32,v=1024)`: `~2301.77 ms`
- speedup: `xla / pallas ~2.24x`

### Numerics checks
At target shape (`B=8192,H=1024,V=128256`), versus XLA:
- `loss_diff`: `0.0` (float32 scalar equality in sampled run)
- `gx`: mean abs diff `~1.54e-7`, max abs diff `~7.63e-6`
- sampled `gw`: mean abs diff `~5.53e-15`, max abs diff `~5.82e-11`

Interpretation note:
- A literal `loss_diff=0.0` here means equal rounded float32 scalar in this run, not proof of bitwise equivalence in all regimes.

### Additional tuning notes
- Backward vocab-tile sweep with rewritten kernel kept `v_block=7168` as best for `B>=8192` among tested values.
- FA forward limited sweep found `b=1024,h=32,v=1024` best among valid shared-memory-safe candidates.
- Invalid combinations encountered as expected:
  - `h=64,v=1024` exceeds GB10 shared-memory tile budget,
  - non-power-of-two `v_block` rejected by current lowering constraints.

### 2026-02-20 follow-up: tuned block-size inference update
- Updated `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`:
  - Added GB10 bucket `gb10-large-vocab-mid-batch` (`B in [2048,32768], H in [512,1536], V in [65536,262144]`).
  - Mapped GB10 BF16/FP32 tuned block sizes for that bucket to `BlockSizes(b=1024,h=32,v=1024)`.
- Validation:
  - `infer_block_sizes(8192,1024,128256,dtype=float32,device_kind='NVIDIA GB10')` now returns `BlockSizes(1024,32,1024)`.
  - With `LEVANTER_PALLAS_GPU_GB10_NATIVE_FORWARD=1` and inferred block sizes (`block_sizes=None`), measured `fwd+bwd` median: `~143.88 ms` on `B=8192,H=1024,V=128256`.

### 2026-02-20 micro-opt update (backward label mask dtype)
- In `_backward_streaming_from_lse`, changed label one-hot temporary from `float32` to `x.dtype` (BF16 on target runs), while subtraction remains against `float32` `dlogits`.
- Rationale: reduce one-hot temporary bandwidth/footprint with exact 0/1 representability in BF16.
- Measured at `B=8192,H=1024,V=128256` (`fwd+bwd`):
  - `xla`: `~322.71 ms`
  - `pallas_hybrid` (inferred blocks): `~143.42 ms`
  - `pallas_fa` (inferred blocks + opt-in): `~143.94 ms`
- Numerics vs XLA in sampled run remained tight (`loss_diff=0.0`, `gx_max ~7.63e-6`, sampled `gw_max=0.0`).
