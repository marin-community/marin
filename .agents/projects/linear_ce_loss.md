# Linear CE Loss Pallas Kernel – Status (2026-01-31)

## Snapshot
We now have a **new Pallas TPU backward kernel** for `fused_cross_entropy_loss_and_logsumexp_penalty` that uses a **parallel core grid axis** (no `core_map`) with per-core `w_grad` partials. This targets v5p megacore and uses **per-core batch splitting** plus explicit HBM↔VMEM staging.
**Note (historical):** we previously experimented with a **split backward strategy** (separate Pallas calls for `x_grad` and `w_grad`, each rematting logits), but this path and its `BlockSizes(..., bwd_strategy="split")` hook have been removed; the current implementation only exposes block-size parameters (`b_block_size`, `h_block_size`, `v_block_size`) for the single combined backward kernel.

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
- Removed the split backward path and unused legacy backward kernels.

### 2026-02-17: Ferry low-MFU block-size sweep (v5p, llama-150m-ish shape)
- Shape target for ferry daily (`train_batch_size=512`, `seq_len=1024`, `data_shards=4`):
  - per-shard kernel shape `B=131072, H=512, V=128256`.
- Added sweep controls to `lib/levanter/scripts/tune/tune_fused_cross_entropy_loss_block_sizes.py`:
  - configurable `implementation`, `steps`, and CSV block-size grids,
  - optional `--include-infer`,
  - machine-readable `result_json` output per config.
- Large-shape sweep (`ray-run-dlwh-tune_fused_cross_entropy_loss_block_sizes-20260217-221606`, east5-a):
  - tested `b in {1024,2048}`, `h in {256,512}`, `v in {512,768,1024,1280,1536,2048}` + `infer`.
  - 25 configs total, 9 succeeded, 16 failed `RESOURCE_EXHAUSTED` (scoped VMEM 16 MiB limit).
  - best successful config: `BlockSizes(b=1024, h=512, v=1024)` at ~2.43M tokens/s (global-token estimate).
  - `infer` picked nearly identical performance (~2.43M tokens/s), consistent with tuned table selecting `1024/512/1024`.
  - most configs with larger `v` (>=1536) or many `b=2048` combinations failed VMEM.
- Large-shape XLA baseline (`ray-run-dlwh-tune_fused_cross_entropy_loss_block_sizes-20260217-221618`, east5-a):
  - failed with HBM OOM in backward (`RESOURCE_EXHAUSTED`, ~112G program HBM requirement).
- Smaller-shape A/B (for backend comparison):
  - xla job `ray-run-dlwh-tune_fused_cross_entropy_loss_block_sizes-20260217-221946` at `B=32768,H=512,V=128256` succeeded ~0.96M tokens/s.
  - pallas job `ray-run-dlwh-tune_fused_cross_entropy_loss_block_sizes-20260217-221959` at same shape:
    - best (`infer`) ~2.39M tokens/s, ~2.48x faster than xla.
    - `v=1280` failed VMEM; `v<=1024` succeeded.
- Conclusion from this sweep:
  - For ferry-scale shape, the immediate issue is configuration/memory envelope (`v_block_size` too large) rather than a universally bad kernel.
  - Recommended ferry setting on v5p for this shape remains `v_block_size=1024` (or use infer/default table).

### 2026-02-17: v5p llama3-ish (`H=8192,V=128256`) parity investigation
- Target shape (kernel batch): `B in {8192,16384,32768}`, `H=8192`, `V=128256`, `dtype=bf16/fp32acc`.
- Root cause for earlier gap:
  - forward pallas path did not use explicit core-grid parallelization while backward already did.
- Code change:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
  - forward `pallas_call` grid changed from `(num_b_blocks, num_v_blocks, num_h_blocks)` to
    `(num_cores, num_b_blocks_per_core, num_v_blocks, num_h_blocks)` using `_infer_core_grid(...)`.
  - forward kernel program IDs updated accordingly.

#### Post-change forward comparison (bench_fused_cross_entropy_loss_pallas.py, block-sizes=infer)
- `B=8192`: pallas `198,494.70 tok/s` vs xla `176,862.09 tok/s` (`+12.2%` pallas)
- `B=16384`: pallas `198,815.25 tok/s` vs xla `181,035.99 tok/s` (`+9.8%` pallas)
- `B=32768`: pallas `198,135.38 tok/s` vs xla `170,873.36 tok/s` (`+15.9%` pallas)

#### Backward status (same runs)
- `B=8192`: pallas bwd `43,580.19 tok/s` vs xla bwd `49,992.28 tok/s`
- `B=16384`: pallas bwd `43,742.56 tok/s` vs xla bwd `51,350.72 tok/s`
- `B=32768`: pallas bwd `43,151.46 tok/s` vs xla bwd `49,678.96 tok/s`
- Net: forward parity is achieved/beaten; backward remains slower.

#### Additional investigations
- Block-size sweep at `B=8192,H=8192,V=128256` (value+grad benchmark):
  - best runnable config remained `b1024_h512_v1024` (~43.6k tok/s).
  - most larger `v_block_size` configs (`>=2048`) failed scoped VMEM (`RESOURCE_EXHAUSTED`).
- Tried replacing one-hot paths with gather/scatter indexing:
  - failed in Mosaic lowering (`_gather_lowering_rule` assertion), reverted.
- Tried forcing more core partitions (`num_cores=4`) for testing:
  - no gain; slightly worse backward, reverted.
- Added optional one-hot backend toggle (default remains emulated):
  - env var `LEVANTER_FUSED_CE_NATIVE_ONE_HOT=1` opts into `jax.nn.one_hot` on non-v4.
  - v4 still forces emulated one-hot.
- This is for fast A/B testing of backward one-hot overhead without changing default behavior.
- Benchmark attempts for this variant on `marin-us-east5-a` and `marin-us-central2-staging`
  were queued but remained `PENDING` due TPU capacity during this session.

### 2026-02-18: Backward x_grad VMEM accumulation experiment (v5p, central1)
- Kernel change in `pallas_tpu.py`:
  - Backward path now accumulates `x_grad_tile_ref` across `v_index` entirely in VMEM and performs a single async VMEM->HBM write at `v_index == num_v_blocks - 1`.
  - Removed per-`v` HBM read/modify/write for `x_grad` (kept `w_grad` HBM staging path unchanged).
- Motivation: reduce backward HBM traffic and close residual pallas vs xla gap at `H=8192, V=128256`.

#### Central1 benchmark results (`bench_fused_cross_entropy_loss_pallas.py`, `block-sizes=infer`)
- `B=8192`:
  - pallas (new): fwd `197,998.77 tok/s`, bwd `46,279.66 tok/s`, combined ~`37,505.58 tok/s`
  - pallas (old emu-onehot baseline): fwd `198,557.80`, bwd `43,555.29`, combined ~`35,725.95`
  - xla (refresh): fwd `176,208.00`, bwd `49,891.52`, combined ~`38,887.40`
  - Net: pallas bwd +6.3%, combined gap to xla shrank from ~8.9% to ~3.6%.
- `B=16384`:
  - pallas (new): fwd `199,319.35`, bwd `46,645.61`, combined ~`37,802.43`
  - xla (refresh): fwd `180,909.38`, bwd `51,340.12`, combined ~`39,991.08`
  - Net: pallas still ~5.5% behind xla on combined at this batch.
- `B=32768`:
  - pallas (new): fwd `195,244.70`, bwd `46,548.56`, combined ~`37,587.32`
  - xla (refresh): fwd `159,674.22`, bwd `48,702.89`, combined ~`37,319.82`
  - Net: near parity / slight pallas lead (~0.7%) on combined.

#### Infrastructure notes
- Several central1 jobs failed due node termination / supervisor actor loss (not kernel exceptions).
- Needed retries for `B=16384` pallas to get a clean measurement.

#### In-flight
- Submitted fresh block-size sweeps for patched kernel at:
  - `B=8192`: `ray-run-dlwh-ce-sweep-xgradvmem2-b8192-h8192-v128256-central1-20260218`
  - `B=16384`: `ray-run-dlwh-ce-sweep-xgradvmem2-b16384-h8192-v128256-central1-20260218`
  - `B=32768`: `ray-run-dlwh-ce-sweep-xgradvmem2-b32768-h8192-v128256-central1-20260218`
- Sweep grid: `b in {1024,2048}`, `h in {256,512}`, `v in {1024,1536,2048}`, plus `infer`.

### 2026-02-18: Central1 sweep completion (`B=32768` retry)
- Job: `ray-run-dlwh-ce-sweep-xgradvmem2-retry-b32768-h8192-v128256-central1-20260218`
- Status: `SUCCEEDED`
- Top configs (value+grad throughput):
  - `infer`: `46846.51 tok/s` (`steady_time_s=0.699476`)
  - `b1024_h512_v1024`: `46840.14 tok/s` (`steady_time_s=0.699571`)
  - `b1024_h256_v1024`: `43627.88 tok/s` (`steady_time_s=0.751079`)
- Conclusion remains consistent with B=8192 and B=16384: `infer` ~= `b1024_h512_v1024`, and larger `v` settings in this sweep grid do not beat the 1024-v block family.

### 2026-02-18: H=4096/V=128256 parity tracking (central1/east5/east1)
- Primary goal reset: beat XLA on **combined fwd+bwd** for `H=4096, V=128256`.
- Infra reality during this pass:
  - many jobs failed with `JOB_SUPERVISOR_ACTOR_START_TIMEOUT` at 900s before entrypoint start,
  - `dev_tpu.py allocate` on central1 and east5 both timed out waiting for `host_info` (after autoscaler expansion messages),
  - east1 submissions initially requested `v6e-8`; autoscaler reported unsatisfiable request (`TPU-v6e-8-head`), so v6e-4 retries were queued.

#### One completed central1 measurement so far
- Job: `ray-run-dlwh-ce-pallas-h4096-b32768-v128256-central1-longwait2-20260218`
- Status: `SUCCEEDED`
- Command shape/backend:
  - `bench_fused_cross_entropy_loss_pallas.py --batch 32768 --pos 1 --embed 4096 --vocab 128256 --implementation pallas_tpu --block-sizes infer`
- Extracted metrics:
  - `steady_time_s`: `0.0852505916` (fwd), `tokens_per_s`: `384372.70`
  - `bwd_steady_time_s`: `0.3557810612`, `bwd_tokens_per_s`: `92101.59`
  - combined fwd+bwd throughput (computed): `32768 / (0.0852505916 + 0.3557810612) = 74298.52 tok/s`

#### Comparator status
- Matching xla at `B=32768`:
  - `ray-run-dlwh-ce-xla-h4096-b32768-v128256-central1-longwait2-20260218` -> `FAILED` (start-timeout).
  - retried as `ray-run-dlwh-ce-xla-h4096-b32768-v128256-central1-retry3-20260218`; currently `PENDING` at time of note update.
- East5 and east1 queues for additional pallas/xla pairs are still mostly `PENDING` and being pruned/retried.

### 2026-02-18: East1 v6e fallback datapoint (`H=4096,V=128256,B=8192`)
- Region/device fallback used while central1/east5 v5p jobs were capacity-constrained.
- pallas run:
  - job: `ray-run-dlwh-ce-pallas-h4096-b8192-v128256-east1v6e4-20260218`
  - status: `SUCCEEDED`
  - metrics:
    - `steady_time_s`: `0.0173003344` (fwd), `tokens_per_s`: `473516.86`
    - `bwd_steady_time_s`: `0.06815923`, `bwd_tokens_per_s`: `120189.15`
    - combined fwd+bwd throughput: `8192 / (0.0173003344 + 0.06815923) = 95858.20 tok/s`
- xla peer run:
  - job: `ray-run-dlwh-ce-xla-h4096-b8192-v128256-east1v6e4-20260218`
  - status: `FAILED` (`JOB_SUPERVISOR_ACTOR_DIED`)
  - failure detail from Ray metadata: actor node terminated (`received SIGTERM`) before benchmark printed timing lines.

Notes:
- This is not a v5p apples-to-apples comparison, but it confirms the pallas kernel is operational at `H=4096,V=128256` on fallback hardware while infra is unstable.

### 2026-02-18: Backward `w_grad` VMEM accumulation experiment (v5p, central1, `H=4096`)
- Kernel change in `pallas_tpu.py` (backward path):
  - Removed per-`b` `w_grad` HBM read/modify/write staging.
  - Now accumulate `w_grad_tile_ref` across `b_index` in VMEM and perform a single async VMEM->HBM write at `b_index == num_b_blocks_per_core - 1`.
  - Also removed the now-unused `w_read_sem` scratch semaphore.
- Motivation: the residual gap looked HBM-traffic bound in backward; this mirrors the earlier `x_grad` VMEM accumulation win.

#### Central1 head-to-head (`B=32768,H=4096,V=128256`, `block-sizes=infer`)
- pallas job:
  - `ray-run-dlwh-ce-pallas-wgradvmem-tpu-b32768-h4096-v128256-central1-20260218`
  - status: `SUCCEEDED`
  - fwd: `steady_time_s=0.0854430582`, `tokens_per_s=383506.87`
  - bwd: `bwd_steady_time_s=0.3358259614`, `bwd_tokens_per_s=97574.35`
  - combined: `32768 / (0.0854430582 + 0.3358259614) = 77784.03 tok/s`
- xla job:
  - `ray-run-dlwh-ce-xla-wgradvmem-tpu-b32768-h4096-v128256-central1-20260218`
  - status: `SUCCEEDED`
  - fwd: `steady_time_s=0.1011706892`, `tokens_per_s=323888.27`
  - bwd: `bwd_steady_time_s=0.3804668188`, `bwd_tokens_per_s=86125.78`
  - combined: `32768 / (0.1011706892 + 0.3804668188) = 68034.57 tok/s`

#### Net effect
- pallas now beats xla on the target `H=4096,V=128256,B=32768` shape by `~14.3%` on combined fwd+bwd in this run.
- Versus the earlier pallas datapoint at the same shape (`74298.52 tok/s`), combined improved by `~4.7%`.

#### Sanity check on prior gap shape (`H=8192,V=128256,B=8192`, central1)
- pallas:
  - `ray-run-dlwh-ce-pallas-wgradvmem-tpu-b8192-h8192-v128256-central1-20260218`
  - fwd `197510.83 tok/s` (`steady_time_s=0.0414762058`)
  - bwd `48418.32 tok/s` (`bwd_steady_time_s=0.1691921660`)
  - combined `38885.76 tok/s`
- xla:
  - `ray-run-dlwh-ce-xla-wgradvmem-tpu-b8192-h8192-v128256-central1-20260218`
  - fwd `177068.56 tok/s` (`steady_time_s=0.0462645660`)
  - bwd `49999.21 tok/s` (`bwd_steady_time_s=0.1638425898`)
  - combined `38989.63 tok/s`
- Result:
  - combined gap shrank from about `-3.55%` (older xgrad-only patch) to about `-0.27%` (near parity) at this shape.

### 2026-02-20: v4-8 retune on us-central2 (dev_tpu)
- Environment:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central2.yaml --tpu-type v4-8`
  - TPU: `TPU v4` (4 local devices on the `v4-8` slice)
- Tuned with:
  - `lib/levanter/scripts/tune/tune_fused_cross_entropy_loss_block_sizes.py`
  - candidate grid from script (`b in {1024,2048}`, `h in {128,256,512}`, `v in {1024,2048,4096}`)
- Result summary (all non-`v=1024` candidates OOM on v4-8 in these runs):
  - `small-vocab` (`B=2048,H=512,V=8192`): best `b=1024,h=256,v=1024` at ~11.04M tokens/s.
  - `llama3-ish` (`B=8192,H=4096,V=128256`): best `b=1024,h=512,v=1024` at ~180k tokens/s.
  - `large-batch-small-h` (`B=65536,H=512,V=128256`): best `b=1024,h=512,v=1024` at ~1.17M tokens/s.
  - `medium-batch-medium-h` (`B=8192,H=2048,V=128256`): best `b=1024,h=512,v=1024` at ~349k tokens/s.
- Code update:
  - Added missing TPU v4 table entries in `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py` for:
    - `small-vocab` (`h=256,v=1024`)
    - `llama3-ish` (`h=512,v=1024`)
  - Kept existing TPU v4 entries for `large-batch-small-h` and `medium-batch-medium-h` (already `h=512,v=1024`).
- Tokamax note:
  - `lib/levanter/scripts/bench/bench_tokamax_linear_softmax_ce.py` currently fails in this dev TPU env with
    `ModuleNotFoundError: No module named 'tokamax'`.

### 2026-02-20: v4-8 pallas vs xla backend comparison (same bench script)
- Command template:
  - `uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py --input-dtype bfloat16 --accum-dtype float32 --block-sizes infer --implementation <pallas_tpu|xla> ...`
- Environment:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central2.yaml` on `v4-8` (`TPU v4`, 4 local devices)
- Shape A (`B=32, P=1024, H=512, V=128256`):
  - `pallas_tpu`: fwd `0.0281s` (`~1.17M tok/s`), bwd `0.112s` (`~292k tok/s`)
  - `xla`: fwd `0.0417s` (`~786k tok/s`), bwd `0.307s` (`~107k tok/s`)
  - Relative: pallas is `~1.48x` faster on fwd and `~2.74x` faster on bwd.
- Shape B (`B=8, P=1024, H=2048, V=128256`):
  - `pallas_tpu`: fwd `0.0201s` (`~408k tok/s`), bwd `0.0939s` (`~87.2k tok/s`)
  - `xla`: fwd `0.0235s` (`~349k tok/s`), bwd `0.106s` (`~77.2k tok/s`)
  - Relative: pallas is `~1.17x` faster on fwd and `~1.13x` faster on bwd.
- Note:
  - Loss values matched closely between backends in these runs.

### 2026-02-21: v4-8 VMEM pressure triage + XLA custom VJP direction
- Environment:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central2.yaml --tpu-type v4-8`
  - TPU: `TPU v4` (4 local devices)
- Core finding on VMEM failures:
  - At large failing configs (e.g. `B=65536,H=512,V=128256`, `b/h/v=1024/128/1024`), `fwd_only` already fails with scoped VMEM OOM (`~41.11M / 16M` bytes), so forward is the first cliff.
  - Backward-only can fail at even higher VMEM (`~47.05M / 16M`), but this is secondary at the tested boundaries.
  - Boundary probing:
    - `B=512,H=512,v_block=768`: forward fails while direct backward can pass.
    - `B=512,H=512,v_block=640`: both pass.
- Pallas speed check versus XLA on v4:
  - For `B=512,H=512,V=128256` forward-only, best Pallas config observed was around `~110k tok/s`.
  - XLA forward-only on same shape was around `~572k tok/s` (about `5x` faster).
  - For value+grad on same shape, Pallas `~49k tok/s` vs XLA streaming `~122k tok/s`.
- Conclusion:
  - For v4, Pallas path is constrained by VMEM and is not competitive at these tested settings.
  - We should favor an XLA streaming path with a custom backward on v4.

#### XLA streaming custom-VJP prototype result (v4-8)
- Prototype behavior:
  - forward uses existing streaming CE (`linear_softmax_cross_entropy_loss_streaming`)
  - backward manually streams over vocab blocks to avoid full autodiff materialization.
- Measured on `B=512,H=512,V=128256`:
  - builtin `xla` (`v_block=32768`) value+grad: `~121.6k tok/s` (`~0.00421s`)
  - custom streaming VJP (`v_block=32768`): `~212.5k tok/s` (`~0.00241s`)
  - custom streaming VJP (`v_block=8192`): `~161k tok/s`
- Prototype correctness spot-check (same env):
  - `loss_builtin == loss_custom` exactly in the sampled run.
  - gradient deltas:
    - `gx_max_abs = 4.8828125e-04`
    - `gw_max_abs = 5.9604645e-08`
    - `gx_rel = 2.2317406e-03`
    - `gw_rel = 9.1245504e-08`

#### Repo changes (this branch)
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py`
  - Added `_use_v4_custom_xla_vjp()` gate: enable custom VJP only for TPU v4.
  - Added `_linear_softmax_cross_entropy_loss_streaming_custom_vjp(...)` with manual streaming backward:
    - computes blockwise `delta = (dL + dLSE)*prob - dL*one_hot`
    - applies soft-cap derivative when enabled
    - accumulates `dx` and writes `dw` blockwise.
  - `linear_softmax_cross_entropy_loss_xla(...)` now dispatches to custom VJP on v4; other backends keep existing behavior.
- `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
  - Added `test_v4_custom_xla_vjp_gate`
  - Added `test_xla_streaming_custom_vjp_grad_matches_streaming_autodiff`
  - Local test run (`-k 'xla or custom_vjp or gate'`): `3 passed, 1 skipped`.

#### In-repo TPU validation after patch (v4-8)
- API bench (`linear_softmax_cross_entropy_loss_xla` through fused loss API):
  - shape `B=512,H=512,V=128256` (`batch=1,pos=512` in bench script)
  - fwd `steady_time_s=0.00090334` (`~566.8k tok/s`)
  - bwd `bwd_steady_time_s=0.00243601` (`~210.2k tok/s`)
- Direct head-to-head on the same TPU run (`v_block=32768`, value+grad):
  - API path (now v4 custom VJP): `0.002495s` (`~205.2k tok/s`)
  - baseline streaming autodiff (`linear_softmax_cross_entropy_loss_streaming`): `0.004175s` (`~122.6k tok/s`)
  - speedup: `~1.67x` for backward-inclusive step time.
- Correctness spot-check on TPU after integration:
  - `_use_v4_custom_xla_vjp()` returned `True` on `TPU v4`.
  - max abs diff versus streaming-autodiff gradient at sample shape (`B=128,H=128,V=4096`):
    - `gx_max_abs = 1.220703125e-04`
    - `gw_max_abs = 1.220703125e-04`

### 2026-02-21: v5p-8 sanity run (us-central1 dev TPU)
- Environment:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-type v5p-8`
  - TPU: `tpu v5` (`4` local devices on this slice)
- API bench (`implementation=xla`, same shape as v4 check):
  - shape `B=512,H=512,V=128256` (`batch=1,pos=512`)
  - fwd `steady_time_s=0.000687716` (`~744.5k tok/s`)
  - bwd `bwd_steady_time_s=0.00283721` (`~180.5k tok/s`)
- Direct API-vs-baseline check (`v_block=32768`, value+grad):
  - `_use_v4_custom_xla_vjp()` returned `False` (expected on v5p/v5).
  - API (`linear_softmax_cross_entropy_loss_xla`): `0.00290257s` (`~176.4k tok/s`)
  - baseline (`linear_softmax_cross_entropy_loss_streaming` autodiff): `0.00289034s` (`~177.1k tok/s`)
  - delta is negligible (`~0.4%`), confirming the v4-only gate preserves v5 behavior.

### 2026-02-21: What XLA is actually implementing (and why tile size is hard to extract)
- HLO inspection on v4 (`linear_softmax_cross_entropy_loss_xla`, `B=512,H=512,V=128256`) shows:
  - explicit while-loop over vocab blocks with trip count `4` (`131072 padded vocab / 32768 block`).
  - per-iteration `dynamic-slice` of `w` with `dynamic_slice_sizes={512,32768}`.
  - one block GEMM-equivalent op per iteration in unoptimized HLO:
    - `dot(Arg_1.9, dynamic_slice.1)` in `closed_call.7`.
  - masked logits (`where` with `-inf`), per-block `reduce_max` / `exp` / `reduce_sum`, `logaddexp` accumulation, and label-logit gather.
- In optimized TPU HLO dump, that dot is canonicalized into a convolution-form op:
  - `convolution(...), dim_labels=bf_io->bf` with metadata tracing back to `dot_general`.
  - This is why searching for `dot(` in late dumps is often misleading.
- What is visible as “tiling”:
  - layout annotations such as `bf16[512,131072]{1,0:T(8,128)(2,1)}` and `f32[512,32768]{1,0:T(8,128)}`.
  - These are layout/packing tiles (memory layout tiling), not a direct “MXU kernel tile size” parameter.
- What is **not** directly exposed:
  - the backend-selected microkernel tile/schedule/unrolling used by TPU codegen/libtpu for the matmul-like op.
  - There is no stable single field in emitted HLO that says “the matmul tile size is X by Y”.
- Practical conclusion:
  - We can reliably recover **algorithmic blocking** (`v_block_size=32768`) and loop structure from HLO.
  - We can see **layout tile annotations** (`T(8,128)` etc.).
  - We generally cannot recover a single definitive backend GEMM micro-tile from user-facing HLO text alone.

### 2026-02-21: Forced custom-VJP trial on v5p-8
- Goal:
  - Evaluate enabling the new custom VJP on v5p (currently gated off in code) by directly calling
    `_linear_softmax_cross_entropy_loss_streaming_custom_vjp(...)`.
- Environment:
  - `marin-us-central1` dev TPU `v5p-8`, device kind reported as `TPU v5`.
  - Gate check: `_use_v4_custom_xla_vjp() == False` (expected).
- Comparison setup:
  - same value+grad benchmark, `dtype=float32`, `v_block_size=32768`.
  - compared:
    - `api_xla` (`linear_softmax_cross_entropy_loss_xla`)
    - `custom_vjp` (forced private custom-vjp call)
    - `stream_autodiff` (`linear_softmax_cross_entropy_loss_streaming` with AD)
- Results:
  - Shape `B=512,H=512,V=128256`:
    - `api_xla`: `0.00290897s` (`~176.0k tok/s`)
    - `custom_vjp`: `0.00208500s` (`~245.6k tok/s`)  **(+39.5% vs api_xla)**
    - `stream_autodiff`: `0.00315458s` (`~162.3k tok/s`)
  - Shape `B=8192,H=4096,V=128256`:
    - `api_xla`: `0.09183749s` (`~89.2k tok/s`)
    - `custom_vjp`: `0.09866696s` (`~83.0k tok/s`)  **(-6.9% vs api_xla)**
    - `stream_autodiff`: `0.09211473s` (`~88.9k tok/s`)
- Interpretation:
  - For v5p, forced custom VJP is **shape-dependent**: faster at smaller shape, slower at larger `H=4096` shape.
  - This supports keeping the default v4-only gate for now unless we add shape-based gating/autotune.
- Correctness spot-check on v5p (small shape `B=512,H=512,V=128256`):
  - max abs grad diff (api vs forced custom):
    - `gx_max_abs = 6.103515625e-05` (`gx_rel = 5.78e-03`)
    - `gw_max_abs = 3.0517578125e-05` (`gw_rel = 3.18e-03`)

### 2026-02-21: v5p question - streaming custom VJP vs pallas
- Direct value+grad head-to-head on `v5p-8`:
  - shape `B=512,H=512,V=128256`:
    - `pallas_tpu` (infer): `0.00337185s` (`~151.8k tok/s`)
    - `streaming_custom_vjp`: `0.00169963s` (`~301.2k tok/s`)
    - result: streaming custom VJP is about `1.98x` faster.
  - shape `B=8192,H=4096,V=128256`:
    - `pallas_tpu` failed scoped VMEM OOM in JVP path (`39.04M / 16.00M`).
    - `streaming_custom_vjp` succeeded at `0.09830s` (`~83.3k tok/s`).

### 2026-02-21: XLA default switched to custom VJP
- Code change:
  - `linear_softmax_cross_entropy_loss_xla(...)` now unconditionally dispatches to
    `_linear_softmax_cross_entropy_loss_streaming_custom_vjp(...)`.
  - Removed the v4-only gate from active dispatch.
- Tests:
  - Removed gate-specific test and kept custom-VJP grad parity test.
  - `pytest -k 'xla or custom_vjp'` in `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`:
    - `2 passed, 1 skipped`.
- v5p sanity after change (`B=512,H=512,V=128256`):
  - `api_xla`: `0.00168887s` (`~303.2k tok/s`)
  - forced custom-vjp call: `0.00169305s` (`~302.4k tok/s`)
  - confirms API now uses the same path.

### 2026-02-21: Default backend policy update
- Changed fused CE API default implementation order to always prefer `xla`, even when `pallas_tpu` is importable.
  - file: `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  - `pallas_tpu` remains available when explicitly requested via `implementation='pallas_tpu'`.
- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
  - result: `14 passed, 3 skipped`.

### 2026-02-21: Follow-up v5p check (where can pallas still win?)
- Additional large-shape head-to-head (`v5p-8`, value+grad):
  - shape `B=32768,H=4096,V=128256`, `v_block=32768`
  - `pallas_tpu`: failed scoped VMEM OOM (`39.04M / 16.00M`)
  - `streaming_custom_vjp`: succeeded at `0.39032s` (`~83.95k tok/s`)
- Combined with earlier same-session results:
  - `B=512,H=512,V=128256`: custom-vjp `~301k tok/s` vs pallas `~152k tok/s`
  - `B=8192,H=4096,V=128256`: pallas OOM, custom-vjp `~83k tok/s`
- Practical takeaway on this env:
  - With current scoped VMEM limits, pallas is not competitive for these tested v5p backward-inclusive workloads.

### 2026-02-21: v5p rerun with higher scoped VMEM limit
- Reran with:
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
  - same value+grad benchmark, `v_block=32768`.
- Results:
  - `B=512,H=512,V=128256`:
    - `pallas_tpu`: `0.00336478s` (`~152.2k tok/s`)
    - `streaming_custom_vjp`: `0.00189534s` (`~270.1k tok/s`)
  - `B=8192,H=4096,V=128256`:
    - `pallas_tpu`: `0.116103s` (`~70.6k tok/s`)
    - `streaming_custom_vjp`: `0.0923914s` (`~88.7k tok/s`)
  - `B=32768,H=4096,V=128256`:
    - `pallas_tpu`: `0.436443s` (`~75.1k tok/s`)
    - `streaming_custom_vjp`: `0.365443s` (`~89.7k tok/s`)
- Conclusion:
  - Raising scoped VMEM lets pallas run large shapes again, but streaming custom VJP remains faster on all tested v5p backward-inclusive shapes.

### 2026-02-21: Tokamax vs xla(custom-vjp) vs pallas on v5e-8/v6e-8 (eu-west4)
- Request:
  - compare Tokamax kernel vs our new default `xla` path (streaming custom VJP) and our `pallas_tpu` kernel.
  - target TPUs:
    - `v5e-8` in `europe-west4-b` (`infra/marin-eu-west4.yaml`)
    - `v6e-8` in `europe-west4-a` (`infra/marin-eu-west4-a.yaml`)
  - shape used for all runs: `B=8192, H=4096, V=128256`.

#### Infra notes
- `v5e-8` allocation had intermittent autoscaler/preemption churn:
  - initial attempts timed out waiting for actor start.
  - one successful allocation was later terminated (`ActorDiedError`, node SIGTERM) and had to be reacquired.
- `v6e-8` allocation was stable in this session.

#### Tokamax install/runtime notes
- A dedicated Tokamax env was used on each TPU VM:
  - `uv venv .venv_tokamax --python 3.11`
  - `uv pip install tokamax`
  - `uv pip install 'jax[tpu]==0.9.0' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
- This produced:
  - `jax==0.9.0`, `jaxlib==0.9.0`, `libtpu==0.0.34` for Tokamax runs.
- Levanter xla/pallas runs stayed on project-locked env (`jax==0.8.0`, `jaxlib==0.8.0`, `libtpu==0.0.24`).

#### Dtype compatibility findings (Tokamax `mosaic_tpu`)
- `bf16` failed on both `v5e` and `v6e` with Pallas verifier error:
  - `'tpu.matmul' op Expected matmul acc to be 32-bit`
- `float32` runs were successful for Tokamax on both `v5e` and `v6e`.
- Per follow-up request, comparison was done in a shared working dtype (`float32`).

#### Float32 comparison (value+grad, `B=8192,H=4096,V=128256`)
- v5e-8 (`europe-west4-b`):
  - `xla` (custom-vjp default):
    - fwd: `128,056 tok/s`
    - bwd: `36,612 tok/s`
    - combined (harmonic): `28,472 tok/s`
  - `pallas_tpu` (`block-sizes=infer`):
    - fwd: `128,223 tok/s`
    - bwd: `25,737 tok/s`
    - combined: `21,435 tok/s`
  - Tokamax `mosaic_tpu`:
    - fwd: `11,036 tok/s`
    - bwd: `22,999 tok/s`
    - combined: `7,458 tok/s`
- v6e-8 (`europe-west4-a`):
  - `xla` (custom-vjp default):
    - fwd: `259,456 tok/s`
    - bwd: `86,501 tok/s`
    - combined: `64,873 tok/s`
  - `pallas_tpu` (`block-sizes=infer`):
    - fwd: `243,238 tok/s`
    - bwd: `53,753 tok/s`
    - combined: `44,024 tok/s`
  - Tokamax `mosaic_tpu`:
    - fwd: `11,451 tok/s`
    - bwd: `76,094 tok/s`
    - combined: `9,953 tok/s`

#### Extra bf16 context (our kernels)
- On both TPUs, our bf16 `xla`/`pallas` runs completed; `xla` remained ahead on combined throughput.
- Tokamax bf16 remained blocked by the verifier error above.

#### Bottom line
- In the only shared working dtype (`float32`), our `xla` custom-vjp path is clearly fastest on combined throughput on both `v5e-8` and `v6e-8`.
- `pallas_tpu` remains competitive on forward but trails on backward, so combined is below `xla`.
- Tokamax `mosaic_tpu` is not competitive in this setup and cannot currently run bf16 on these TPUs due the matmul-accumulator verification failure.

### 2026-02-21: Tokamax config-sensitivity follow-up on v5p (extra fairness sweep)
- Motivation:
  - The earlier Tokamax comparison used `linear_softmax_cross_entropy_loss(..., implementation="mosaic_tpu")`.
  - On v5p, that API path appeared to pick a very poor forward configuration for some shapes.
- Environment:
  - TPU: `v5p-8` (`us-central1`)
  - dtype: `float32`
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
  - shape family: `B=8192, H=4096, V in {32768, 65536, 128256}`
- Tokamax implementation variants:
  - API default: `linear_softmax_cross_entropy_loss(..., implementation="mosaic_tpu")`
  - Explicit config: `PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(config=Config(b=1024,h=512,v=<sweep>))`

#### Key finding
- Tokamax is **highly configuration-sensitive** on v5p:
  - API default (`implementation="mosaic_tpu"`) produced very low forward throughput (~12-13k tok/s) at `V=32768` and `V=65536`.
  - Explicit config recovered expected performance and materially improved combined throughput.

#### v5p results (float32, value+grad, harmonic combined)
- `V=32768`:
  - xla(custom-vjp): fwd `580,870`, bwd `290,403`, combined `193,609`
  - pallas(infer): fwd `1,160,735`, bwd `284,931`, combined `228,773`
  - tokamax API default: fwd `12,956`, bwd `304,287`, combined `12,427`
  - tokamax explicit config sweep:
    - `v=512`: fwd `500,845`, bwd `303,124`, combined `188,836`
    - `v=1024`: fwd `515,722`, bwd `324,284`, combined `199,094` (best)
    - `v=1536`: combined `143,380`
    - `v=2048`: combined `147,772`
- `V=65536`:
  - xla(custom-vjp): fwd `295,078`, bwd `145,586`, combined `97,487`
  - pallas(infer): fwd `585,506`, bwd `143,561`, combined `115,292`
  - tokamax API default: fwd `12,152`, bwd `156,608`, combined `11,277`
- `V=128256` (llama3-ish vocab), tokamax explicit config sweep:
  - `v=512`: fwd `137,489`, bwd `80,900`, combined `50,931`
  - `v=1024`: fwd `141,494`, bwd `86,604`, combined `53,722` (best)
  - `v=1536`: combined `38,753`
  - `v=2048`: combined `39,217`
  - reference from earlier same-shape runs:
    - xla(custom-vjp): combined `48,606`
    - pallas(infer): combined `58,095`

#### Interpretation
- The previous "Tokamax is far slower than xla" result was true for the API-default path tested there, but not universally true once Tokamax block config is set explicitly on v5p.
- On v5p:
  - tuned Tokamax can beat xla(custom-vjp) on combined throughput for `V=32768` and `V=128256`,
  - but still trails tuned pallas(infer) on the same tested shapes.

#### Infra limitation encountered while extending this retune to v5e/v6e
- Attempts to re-run the same explicit-config Tokamax sweeps on `v5e-8` and `v6e-8` (eu-west4) were blocked by repeated `dev_tpu.py allocate` failures:
  - `ConnectionError: ray client connection timeout` after tunnel setup.
- So, only v5p got the extra config-retune follow-up in this session.

### 2026-02-21: Backport attempt from Tokamax v5p path (why it is still fast without megacore)
- Prompted question:
  - If Tokamax is still fairly fast on v5p without explicit megacore core-grid parallelism, are we missing an intra-kernel scheduling trick?
- Side-by-side inspection of Tokamax vs our backward kernel found one meaningful ordering difference:
  - Tokamax issues `x_grad` and `w_grad` async writes and waits once at the end of stage 1.
  - Our kernel was waiting on `x_grad` write completion **before** starting `w_grad` init/accumulation.
  - That ordering can serialize part of stage-1 work and reduce overlap.

#### Code change
- File:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
- Change:
  - removed early stage-1 `x_write_future.wait()`.
  - final stage-1 wait now waits on both `x_write_future` and `w_write_future`.
  - this matches Tokamax's ordering pattern more closely and restores write/compute overlap.
- Also added one-hot path switch:
  - native `jax.nn.one_hot` on non-v4 TPUs,
  - emulated compare-based one-hot on v4 (required for v4 sublane/dynamic_gather issue),
  - debug override via `LEVANTER_FUSED_CE_FORCE_EMULATED_ONE_HOT`.

#### v5p-8 A/B measurements (same node, float32, `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`)
- Node used:
  - `ray-marin-us-central1-worker-ed0b8cf7-tpu` (us-central1-a), with `~/marin` + `.venv` + `.venv_tokamax`.

1) One-hot path A/B (before overlap fix, same code except force emulated/native)
- Shape `B=8192,H=4096,V=128256`:
  - emulated: fwd `299,395`, bwd `71,728`, combined `57,865`
  - native: fwd `299,676`, bwd `71,761`, combined `57,897`
- Shape `B=8192,H=4096,V=32768`:
  - emulated: fwd `1,142,884`, bwd `284,591`, combined `228,614`
  - native: fwd `1,159,544`, bwd `284,756`, combined `228,614`
- Takeaway:
  - native one-hot is at most a small forward win (~1-1.5%) and not the main source of the Tokamax behavior.

2) Overlap-order fix impact (native one-hot, before vs after)
- Shape `B=8192,H=4096,V=128256`:
  - before: fwd `299,676`, bwd `71,761`, combined `57,897`
  - after:  fwd `299,565`, bwd `76,505`, combined `60,941`
  - combined gain: `+5.26%`
- Shape `B=8192,H=4096,V=32768`:
  - before: fwd `1,159,544`, bwd `284,756`, combined `228,614`
  - after:  fwd `1,157,776`, bwd `302,702`, combined `239,963`
  - combined gain: `+4.96%`

#### Same-node comparison after the fix (`B=8192,H=4096,V=128256`, float32)
- pallas (ours, after fix): fwd `299,565`, bwd `76,505`, combined `60,941`
- xla custom-vjp: fwd `145,267`, bwd `72,959`, combined `48,567`
- tokamax tuned (`b=1024,h=512,v=1024`): fwd `141,327`, bwd `86,531`, combined `53,670`

#### Interpretation
- The key thing Tokamax had that helped was **stage ordering / overlap discipline**, not one-hot choice.
- Our megacore-aware forward remains the major advantage on v5p.
- Backward is where Tokamax remains strong; matching its overlap pattern recovered ~5% combined throughput without changing math.
