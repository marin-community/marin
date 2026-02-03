# Linear CE Loss Pallas Kernel – Status (2026-01-31)

## Snapshot
We now have a **new Pallas TPU backward kernel** for `fused_cross_entropy_loss_and_logsumexp_penalty` that uses a **parallel core grid axis** (no `core_map`) with per-core `w_grad` partials. This targets v5p megacore and uses **per-core batch splitting** plus explicit HBM↔VMEM staging.
**New:** a **split backward strategy** is implemented: separate Pallas calls compute `x_grad` and `w_grad` (each remats logits). This is exposed via `BlockSizes(bwd_strategy="split")` for profiling/experiments.

## Where it lives
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
  - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` (HBM-ref, scratch-first `dw`, core-parallel when available)
  - `_make_custom_vjp` always uses this single backward implementation (split path removed)

## Current shape/alignment constraints
The parallel combined kernel enforces:
- `B % num_cores == 0`
- `b_dim >= num_cores * b_block_size`
- `v_block_size % 128 == 0`
- On TPU v4/v5/v7 with large batch (`B >= 1024`), `b_block_size % 1024 == 0` to satisfy label layout constraints.

We are using `pl.multiple_of` on b/h/v start indices to satisfy tiling constraints.

## Kernel structure (bwd combined, core-parallel)
Per-core kernel (leading grid axis) now:
1. For each `v_block` and `h_block`: zero `dw_accum` in VMEM.
2. Loop over all `b_block`s for that core:
   - Build logits over **all h_blocks** (remat) for the current `b_block, v_block`.
   - Compute `delta` (softmax + labels + lse + soft-cap).
   - Update `x_grad` in HBM (RMW) for this `b_block, h_block`.
   - Accumulate `dw_accum += x_tile^T @ delta` (scratch-first).
3. After the `b_block` loop: write `dw_accum` **once** to `w_grad_partial[core, h_slice, v_slice]`.
4. Host-side: `w_grad = sum(w_grad_partial, axis=0)`.

### Parallel combined pseudocode (current implementation)
```text
grid = (num_cores, num_b_blocks_per_core, num_v_blocks, num_stages, num_h_blocks)
dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary", "arbitrary")

for core_idx in parallel:
  core_idx = axis_index("core")
  b_per_core = B / num_cores
  b_start = core_idx * b_per_core

  for b_block in range(0, b_per_core, b_block_size):
    b_slice = [b_start + b_block : b_start + b_block + b_block_size]
    for v_block in range(0, V, v_block_size):
      v_slice = [v_block : v_block + v_block_size]

      # Load tiles into VMEM
      x_tile = x[b_slice, :]
      w_tile = w[:, v_slice]
      lse_tile = lse[b_slice]
      y_tile = labels[b_slice]
      dloss_tile = dout_loss[b_slice]
      dlse_tile = dout_lse[b_slice]

      # Remat logits and softmax delta
      logits = x_tile @ w_tile
      if softcap: logits = softcap(logits)
      delta = softmax(logits - lse_tile) * dloss_tile
      if logit_soft_cap: delta *= cap_deriv(logits)
      delta[range(b_block_size), y_tile - v_block] -= dloss_tile
      delta += 2 * logsumexp_weight * lse_tile * dlse_tile

      # Accumulate gradients
      dx_tile = delta @ w_tile.T
      dw_tile = x_tile.T @ delta

      x_grad[b_slice, :] += dx_tile
      w_grad_partial[core_idx, :, v_slice] += dw_tile

  w_grad = sum(w_grad_partial, axis=0)
```

### Alternative cooperative megacore sketch (not implemented)
```text
cooperative megacore:
  split each b_block across cores:
    core0 handles b_sub0, core1 handles b_sub1
    each core computes logits for its b_sub
  all-gather logits across cores to form full (b_block, v_block) on both cores
  core0 computes dx for full b_block
  core1 computes dw for full b_block
```

## Known issues
- We are likely close to the practical limit of this remat design:
  - reference fwd ~31 ms
  - reference bwd ~109 ms
  - ideal remat floor ~= 109 + 31 = 140 ms
  - current pallas bwd ~= 180 ms
- Remaining gap (~40 ms vs floor) is plausibly explained by tiling + DMA/semaphore overhead + per-core `w_grad_partial` reduction.

## Test status
- `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
  - TPU bwd comparison now targets the single backward path (split removed).
  - Some failures were previously due to XLA layout / tile misalignment; now handled by constraints.

## Perf notes
- Forward+backward test (B=2048, H=128, V=4096) compiled, perf ~3.4M tok/s (fwd+bwd), OK.
- Realistic profile regime now at roughly:
  - fwd ~31 ms
  - bwd ~180 ms
- This is a major improvement from earlier ~300+ ms bwd, and likely leaves only incremental tuning headroom.

## Open TODOs
- If needed, squeeze incremental wins via block-size retuning and pipeline scheduling.
- If we need another large step, it probably requires algorithm/storage changes rather than micro-tuning.

## V4 Work (separate thread / subagent candidate)
### Goal
We need a **minimal Pallas kernel** that reliably reproduces the **TPU v4 sublane crash**
triggered by `jax.nn.one_hot` / dynamic gather lowering inside a Pallas kernel, and then
we need a **v4-safe one-hot emulation** to avoid that path.

### Why
On v4, `jax.nn.one_hot(labels, ...)` inside a Pallas kernel lowers to
`tpu.dynamic_gather` on the **sublane** path, which is unsupported and crashes the
kernel. This blocks the fused CE Pallas path on v4 entirely.

### What we want
1. **Repro kernel**: a tiny Pallas kernel that calls `jax.nn.one_hot` (or the specific
   gather primitive) and triggers the v4 sublane error. This gives us a stable minimal
   repro for upstream or future debugging.
2. **Workaround kernel**: emulate one-hot **without** `dynamic_gather` on v4, e.g.:
   - Compute a block-local index vector `idx = arange(v_block)` in SMEM/VMEM
   - For each label in the `b_block`, compute `eq = (idx == label_offset)` using
     broadcast compare, yielding a one-hot row
   - Use that boolean mask (cast to dtype) to implement the same label subtraction
     in the softmax delta path

The workaround should be isolated to v4 (or a backend capability check) so v5+ can
keep using the faster native lowering.

### Files already created
- `lib/levanter/scripts/debug/fused_ce_v4_probe.py`
  - Tries the Pallas path on v4 and prints the failure.
- `lib/levanter/scripts/debug/fused_ce_v4_workaround.py`
  - Forces XLA path; useful for baseline comparisons.
- `lib/levanter/scripts/debug/splash_attention_v4_repro.py`
  - Uses splash attention with `v=I` to recover softmax; works on v4.
- `lib/levanter/scripts/debug/splash_like_logsumexp_v4_repro.py`
  - Minimal splash-style logsumexp + in-kernel label logits; works on v4.

### V4 work log (2026-02-02)
- Built a splash-style logsumexp Pallas kernel that runs on TPU v4 with
  `dimension_semantics=("parallel","arbitrary","arbitrary")` and streaming `m/l`
  accumulation (no gather). Matches reference logsumexp within ~2e-6.
- Extended it to compute cross-entropy in-kernel by accumulating label logits
  via a one-hot compare within each V tile (no dynamic_gather). Also matches
  reference CE within ~2e-6.
- Added a v4-safe fused CE forward kernel (streaming logsumexp + label logits, no gather)
  plus emulated one-hot for backward; auto-selected on v4 via `_is_tpu_v4()`.
- Added v4 smoke scripts (`v4_ce_smoke.py`, `v4_ce_smoke_bwd.py`) to validate fwd/bwd
  correctness and bypass sublane gather issues.
- Tune results (value+grad) on v4:
  - B=65536, H=512, V=128256: only `v_block_size=1024` fits; best `b=1024,h=512,v=1024`
    at ~293k tokens/s; larger v_block sizes hit VMEM OOM.
  - B=16384, H=2048, V=128256: only `v_block_size=1024` fits; best `b=1024,h=512,v=1024`
    at ~89k tokens/s; larger v_block sizes hit VMEM OOM.
- Registered v4 tuned block sizes in `tuned_block_sizes.py` (including new
  `medium-batch-medium-h` bucket for H≈2048).

### Next steps
- Consider adding a tiny standalone kernel that **only** does one-hot to isolate
  the crash (even smaller than the fused CE kernel), then verify it fails on v4.
- Add a regression test that exercises the emulated one-hot path on v4.

## How to run TPU tests
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml execute .venv/bin/python -m pytest lib/levanter/tests/kernels
```

## Perf bench
- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
- Added roofline estimate and grug-wrapped bench variant.

### V4 bench snapshots (pallas_tpu, infer block sizes)
- B=65536, H=512, V=128256 (batch=64, pos=1024):
  - roofline tokens/s ~8.38M (4 chips)
  - fwd tokens/s ~1.17M
  - bwd tokens/s ~0.293M
- B=16384, H=2048, V=128256 (batch=16, pos=1024):
  - roofline tokens/s ~2.09M (4 chips)
  - fwd tokens/s ~0.410M
  - bwd tokens/s ~0.089M
- XLA reference path OOMs on bwd for B=65536 due to full BxV temporary (>56GB HBM).

### V5p forced-streaming kernel tune (LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000)
- Forced the streaming (v4-safe) kernel on v5p via `_use_v4_kernel()` and tuned.
- B=65536, H=512, V=128256: best `b=1024,h=512,v=1024` (~576k tok/s).
- B=16384, H=2048, V=128256: best `b=1024,h=256,v=2048` (~160k tok/s).
- v_block_size=4096 configs still hit scoped vmem OOM (~48.8MiB limit even with increased scoped vmem).

### 2026-02-03: Streaming kernel becomes default
- Removed the legacy Pallas forward kernel and selection logic; the streaming kernel is now the only path.
- Backward defaults to the emulated one-hot path.
- Updated v5p/v5 tuned block sizes to match the new default.

### V4 head-to-head (fits reference/XLA)
Shape: B=32768, H=512, V=128256 (batch=32, pos=1024), v4-8 (4 chips)
- pallas_tpu:
  - fwd ~1.17M tokens/s (0.0281s)
  - bwd ~0.292M tokens/s (0.112s)
- xla:
  - fwd ~0.787M tokens/s (0.0417s)
  - bwd ~0.107M tokens/s (0.307s)
- reference:
  - fwd ~0.938M tokens/s (0.0349s)
  - bwd ~0.354M tokens/s (0.0926s)
Notes:
- pallas is now faster than reference on forward, and ~2.7× faster than xla on bwd at this shape.

### 2026-02-03: Cleanup follow-up
- Dropped the `use_emulated_one_hot` flag; backward always uses the emulated one-hot path now.
- Streaming Pallas kernel is the only forward implementation (no legacy selection logic left in code).
- Updated `BlockSizes` defaults to `b=1024, h=512, v=1024` to match the v4 tuning baseline.
