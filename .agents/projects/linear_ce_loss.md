# Linear CE Loss Pallas Kernel – Status (2026-01-31)

## Snapshot
We now have a **new Pallas TPU backward kernel** for `fused_cross_entropy_loss_and_logsumexp_penalty` that uses a **parallel core grid axis** (no `core_map`) with per-core `w_grad` partials. This targets v5p megacore and uses **per-core batch splitting** plus explicit HBM↔VMEM staging.

Forward already uses megacore via `dimension_semantics`. Backward now uses a leading grid axis of size `num_cores`, accumulates `w_grad_partial[core]`, then reduces across cores.

## Where it lives
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
  - `linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel`
  - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` (single backward path now)
  - `_make_custom_vjp` always uses the same backward implementation (split path removed)

## Current shape/alignment constraints
The parallel combined kernel enforces:
- `B % num_cores == 0`
- `b_dim >= num_cores * b_block_size`
- `v_block_size % 128 == 0`
- On TPU v4/v5/v7 with large batch (`B >= 1024`), `b_block_size % 1024 == 0` to satisfy label layout constraints.

We are using `pl.multiple_of` on b/h/v start indices to satisfy tiling constraints.

## Kernel structure (bwd combined, core-parallel)
Per-core kernel (via a leading grid axis) currently does:
1. For each `(b_block, v_block)` tile:
   - Loads `x` and `w` tiles into VMEM.
   - Recomputes logits for that tile (remat).
   - Computes softmax delta for labels and LSE, produces `delta`.
2. Accumulates:
   - `dx` into HBM (`x_grad_hbm_ref`) using `pltpu.sync_copy` read/modify/write.
   - `dw` into per-core HBM buffer `w_grad_partial_ref[core_idx]`.
3. Host-side: `w_grad = sum(w_grad_partial_ref, axis=0)`

This means **dw is computed per-core** to avoid cross-core atomics, and reduced afterward.

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

## How to run TPU tests
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml execute .venv/bin/python -m pytest lib/levanter/tests/kernels
```

## Perf bench
- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
- Added roofline estimate and grug-wrapped bench variant.
