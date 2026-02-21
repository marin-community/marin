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
