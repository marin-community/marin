# Linear CE Loss Pallas Kernel – Status (2026-01-31)

## 2026-03-11: XLA explicit batch-block override for tuning
- Plumbed explicit `block_sizes.b_block_size` through the XLA fused CE path so tuning scripts can sweep the new batch-block knob instead of always using the inferred value.
- Behavior:
  - `block_sizes is None`: keep inferred `infer_xla_b_block_size(B, v_block_size)` behavior.
  - explicit `block_sizes.b_block_size`: use it exactly for XLA if it divides `B` and satisfies the TPU/XLA int32 tile-word limit; otherwise raise.
- Added regression coverage for:
  - explicit XLA `b_block_size` reaching the custom-VJP path
  - rejection of unsafe explicit `b_block_size * v_block_size` tiles

## 2026-03-11: XLA fused CE batch tiling for issue #3530
- Added an XLA-side `infer_xla_b_block_size(b, v_block_size)` heuristic to cap the streaming working set below the TPU/XLA int32 word-count limit.
- Updated the XLA fused CE path to chunk over batch as well as vocab in both:
  - forward streaming / `return_argmax`
  - custom-VJP backward
- Added focused tests for:
  - large-batch `infer_xla_b_block_size` selection (`4194304 x 8192 -> 131072`)
  - chunked custom-VJP gradients vs streaming autodiff
  - chunked `return_argmax` forward vs reference
- This change is local to `xla.py` / `tuned_block_sizes.py`; Pallas kernels are unchanged.

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

### 2026-03-09: v5p-8 mixed-dtype large-batch-medium-h retune
- Target reproducer:
  - `B=40960`, `H=2048`, `V=128256`
  - `x.dtype=bf16`, `w.dtype=float32`, `compute dtype=float32`
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
- Root cause for the earlier “autotune didn’t fire” report:
  - the repro path called `infer_block_sizes_with_tuned_match(...)`, materialized a `BlockSizes(...)`, and passed it into `fused_cross_entropy_loss_and_logsumexp_penalty(...)`
  - the API correctly treats any caller-supplied `block_sizes` as explicit/pinned, so autotune-on-miss was bypassed
- Fixes landed:
  - tuned block-size lookup now uses the widest participating dtype bucket, so mixed `x=bf16` / `w=float32` hits the float32 tuning
  - added `large-batch-medium-h` TPU bucket for `B=32768..131072`, `H=1536..3072`, `V=120000..131072`
  - TPU autotune miss sweep for this regime now explores `h in {256,512,1024,2048}` and `v in {128,256,512,768,1024}`
  - autotune now raises if every candidate fails instead of silently caching the inferred miss config
  - the checked-in repro helper leaves `block_sizes=None` on lookup miss unless the user explicitly overrides block sizes
- Scoped-VMEM failure classification:
  - all `v_block_size=512` candidates OOM in the same forward/JVP compile site:
    - `jit(loss_fn)/jvp(jit(linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu))/pallas_call`
    - scoped allocation `50.45M`, limit `48.83M`, exceeded by `1.62M`
- Explicit mixed-dtype sweep on v5p-8 (`steps=1`, `value_and_grad=True`):

| `h_block_size` | `v=128` | `v=256` | `v=512` | `v=768` | `v=1024` |
| --- | ---: | ---: | --- | ---: | ---: |
| `256` | `82,957.69` | `132,996.62` | `OOM` | `176,259.82` | `180,141.22` |
| `512` | `82,962.24` | `132,996.63` | `OOM` | `183,573.79` | `180,073.16` |
| `1024` | `82,976.04` | `132,986.13` | `OOM` | `183,801.09` | `180,212.50` |
| `2048` | `82,966.04` | `133,023.67` | `OOM` | `183,796.71` | `180,247.28` |

- Selected tuned entry:
  - `BlockSizes(b_block_size=1024, h_block_size=1024, v_block_size=768)`
  - checked in for `TPU v5p` and `TPU v5`, for both `bfloat16` and `float32` tuned buckets in `large-batch-medium-h`
- Default-path verification after the table update:
  - launcher: `ray-run-dlwh-bench_fused_ce_default_verify-20260309-verify`
  - selected tuned block sizes: `BlockSizes(b_block_size=1024, h_block_size=1024, v_block_size=768)`
  - repro ran with `block_sizes=None`, `has_tuned_match=True`, `autotune_bypassed=False`
  - result: `compile_time_s=8.0815`, `steady_time_s=0.22973`, `tokens_per_s=178,293.57`, `loss=200.1534`
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

### 2026-03-01: Restore explicit TPU v6 tuned table entries
- Added an explicit `TPU v6` section in `tuned_block_sizes.py` (no `v5e` copy/fallback indirection).
- `TPU v6` entries now include:
  - `small-vocab`: `b=1024,h=256,v=512` (bf16/f32)
  - `llama3-ish`: `b=1024,h=512,v=1024` (bf16/f32)
  - `mid-h-large-vocab`: `b=1024,h=1024,v=1024` (bf16/f32)
  - `large-batch-small-h`: `b=1024,h=512,v=2048` (bf16/f32)

### 2026-03-01: Uniform TPU label-layout rule
- Simplified fused-CE TPU block-size validation and inference policy:
  - For all TPU device kinds, when `B >= 1024`, require `b_block_size % 1024 == 0`.
  - Removed per-generation label-layout gating in `tuned_block_sizes.py`.
  - Updated runtime validation in `pallas_tpu.py` to enforce the same uniform TPU rule.

### 2026-03-01: Diagnostics hook validation on dev TPU (v6e-8, europe-west4-a)
- Validated new bench/tune diagnostics flags end-to-end on a fresh dev TPU allocation:
  - TPU: `v6e-8`
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`
  - flags under test:
    - `--xla-dump-dir`
    - `--compiler-log-path`
- Bench run (`bench_fused_cross_entropy_loss_pallas.py`, forward-only, pallas):
  - shape: `B_tokens=1024, H=1024, V=32768`
  - output included `xla_dump_dir`, `compiler_log_path`, `xla_flags`, `libtpu_init_args` in `result_json`.
  - artifacts on TPU:
    - `/tmp/ce_diag/bench_hlo` with `148` dumped files
    - `/tmp/ce_diag/bench_compile.log` containing benchmark output and `result_json`.
- Tune run (`tune_fused_cross_entropy_loss_block_sizes.py`, forward-only, one explicit config):
  - shape: `global_batch=4, seq_len=256, data_shards=1` (kernel batch `1024`), `H=1024`, `V=32768`
  - config: `b1024_h512_v1024`
  - output included the same diagnostics metadata keys in `result_json`.
  - artifacts on TPU:
    - `/tmp/ce_diag/tune_hlo` with `148` dumped files
    - `/tmp/ce_diag/tune_compile.log` containing run output and `result_json`.
- Conclusion:
  - diagnostics hooks are functioning as intended on dev TPU runs and capture reproducible compile/HLO context.

### 2026-03-04: v4-8 retune for Grug reference shape (us-central2)
- TPU setup:
  - Cluster config: `infra/marin-us-central2-staging.yaml`
  - TPU: `v4-8` (4 chips), dev TPU via `scripts/ray/dev_tpu.py`
- Shape under test:
  - `global_batch=256`, `seq_len=4096`, `data_shards=4`
  - kernel shape `B=262144`, `H=1024`, `V=128256`
  - dtype: inputs `bfloat16`, accum `float32`

- Reproduced failure with prior/default Pallas tuning sweep:
  - `b1024_h{128,256,512}_v1024`: compile vmem OOM (`Used 20.19M of 16.00M`)
  - `b1024_h{256,512}_v2048`: compile vmem OOM (`Used 32.29M of 16.00M`)
  - `b1024_h{256,512}_v4096`: compile vmem OOM (`Allocation 33,554,432 > 16,777,216`)
  - `b2048_h{256,512}_v2048`: compile vmem OOM (`Used 48.54M of 16.00M`)

- XLA checks at same shape:
  - `bs=256` (`B=262144`): success, `steady_time_s=1.9428`, `~539,724 tok/s`
  - `bs=512` (`B=524288`): success in isolated CE tune script, `steady_time_s=3.9370`, `~532,676 tok/s`
  - Note: the `bs=512` success here is isolated CE benchmarking and does not include full-train activation/state pressure.

- Expanded Pallas sweep (explicit grid):
  - sweep: `b in {1024,2048}`, `h in {128,256,512,1024}`, `v in {128,256,512,768,1024}`
  - `b=1024`:
    - `v=128` and `v=256` succeeded for all tested `h`
    - best observed: `b1024_h128_v256` at `~663,603 tok/s` (`steady_time_s=1.5801`)
    - `v>=512` failed (vmem compile OOM)
  - `b=2048`:
    - all tested `h/v` failed (vmem compile OOM)

- Follow-up code changes (this run):
  - keep TPU default path XLA-first in `api.py` (Pallas only when explicitly requested)
  - add new TPU v4 tuned bucket for huge-batch/small-h shapes:
    - bucket: `B in [131073, 1048576], H in [256,1024], V in [120000,131072]`
    - tuned entry: `b=1024, h=128, v=256` (bf16/f32)

### 2026-03-05: v4-8 no-streaming bwd retune + A/B against bwd streaming
- Request:
  - start from `origin/main`, tune `pallas_tpu` CE on `v4-8` without XLA streaming backward,
    then compare directly against forced streaming backward.
- Code changes:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - `linear_softmax_cross_entropy_loss_pallas(...)` no longer hardcodes `use_bwd_xla_streaming=True`.
    - Added env override parsing:
      - `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH=1` => force streaming backward.
      - default (unset) => use Pallas backward.
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - added tests that validate default non-streaming selection and env-forced streaming override.
- TPU setup:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central2-staging.yaml --tpu-type v4-8`
  - device kind: `TPU v4` (4 chips on slice)
  - no special scoped VMEM flag (kept default v4 behavior)
- Tuning run (non-streaming backward):
  - command: `tune_fused_cross_entropy_loss_block_sizes.py`
  - shape: `global_batch=256, seq_len=4096, data_shards=4` -> kernel `B=262144, H=1024, V=128256`
  - grid: `b=1024`, `h in {128,256,512,1024}`, `v in {128,256,384,512}`
  - summary:
    - all `v >= 384` failed compile VMEM OOM.
    - best successful config: `b=1024, h=1024, v=256` at `~395,150 tok/s` (`steady_time_s=2.6536`).
    - next best was `b=1024, h=512, v=256` at `~354,086 tok/s`.
- Direct A/B benchmark at tuned config (`b=1024,h=1024,v=256`):
  - command: `bench_fused_cross_entropy_loss_pallas.py --batch 64 --pos 4096 --embed 1024 --vocab 128256 ...`
  - no-streaming bwd (default):
    - fwd `~900,763 tok/s` (`steady_time_s=0.2910`)
    - bwd `~99,045 tok/s` (`bwd_steady_time_s=2.6467`)
    - combined `~89,233 tok/s`
  - forced streaming bwd (`--bwd-use-xla-streaming`):
    - fwd `~900,485 tok/s` (`steady_time_s=0.2911`)
    - bwd `~166,610 tok/s` (`bwd_steady_time_s=1.5734`)
    - combined `~140,597 tok/s`
  - relative:
    - streaming backward is `~1.68x` faster on backward throughput.
    - streaming path is `~1.58x` faster on combined fwd+bwd throughput.
- Tuned table update from this run:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  - `TPU v4` + `huge-batch-small-h` updated to `b=1024, h=1024, v=256` (bf16/f32).

### 2026-03-05: focus shift to non-streaming bwd internals (matmul vs softmax) + VLIW/LLO dumps
- User direction:
  - keep non-streaming bwd and make it match/beat streaming.
  - do not pursue atomic accumulation.
  - isolate "just matmul backward" first, then add softmax complications.
  - inspect XLA choices via VLIW/LLO dumps.

- Code changes:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - removed bench-only atomic accumulation path.
    - added env flag:
      - `LEVANTER_PALLAS_TPU_BWD_SKIP_LOGITS_RECOMPUTE_BENCH`
      - when enabled, backward kernel bypasses logits recompute/softmax path and feeds a broadcast delta tile (matmul-dominant decomposition mode).
    - retained existing bench flags:
      - `LEVANTER_PALLAS_TPU_BWD_SKIP_SOFTMAX_BENCH`
      - `LEVANTER_PALLAS_TPU_BWD_SKIP_LABEL_SUBTRACT_BENCH`
      - `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH`
      - `LEVANTER_PALLAS_TPU_BWD_V_BLOCK_SIZE_BENCH`
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - updated `_make_custom_vjp` signature expectations (removed atomic arg, added `skip_logits_recompute_bench` arg).
    - full file test pass: `25 passed, 12 skipped`.

- v4-8 benchmark shape (all runs):
  - `batch=64`, `pos=4096` => global tokens `262144`
  - `embed=1024`, `vocab=128256`
  - `pallas_tpu`, block sizes explicit `b=1024,h=1024,v=256`
  - `--shard-map --data-shards 4`

- Throughput results (steady-state):
  - non-streaming bwd (full): `bwd_tokens_per_s ~384,450`, combined `~338,534`
  - streaming bwd forced: `bwd_tokens_per_s ~507,697`, combined `~430,683`
  - non-streaming bwd with `LEVANTER_PALLAS_TPU_BWD_SKIP_LOGITS_RECOMPUTE_BENCH=1`:
    - `bwd_tokens_per_s ~539,093`, combined `~453,079`
  - decomposition read:
    - matmul-dominant non-streaming path is already faster than streaming.
    - major gap in full non-streaming is from logits-recompute/softmax path, not one_hot subtraction and not backward GEMM core.

- VLIW/LLO dump setup used (from TPU post workflow):
  - `XLA_FLAGS="--xla_dump_to=<HLO_DIR> --xla_dump_hlo_as_text"`
  - `LIBTPU_INIT_ARGS` included:
    - `--xla_jf_dump_to=<LLO_DIR>`
    - `--xla_jf_dump_hlo_text=true`
    - `--xla_jf_dump_llo_text=true`
    - `--xla_jf_dump_llo_html=false`
    - `--xla_jf_dump_llo_static_gaps=true`
    - `--xla_jf_emit_annotations=true`
    - `--xla_jf_debug_level=2`
    - `--xla_mosaic_dump_to=<MOSAIC_DIR>`
    - `--xla_mosaic_enable_dump_debug_info=true`
    - `--xla_mosaic_enable_llo_source_annotations=true`

- Artifact locations on TPU:
  - streaming run:
    - HLO: `/tmp/ce_vliw/hlo_stream`
    - LLO: `/tmp/ce_vliw/llo_stream`
    - Mosaic: `/tmp/ce_vliw/mosaic_stream`
  - full non-streaming run:
    - HLO: `/tmp/ce_vliw_nonstream/hlo`
    - LLO: `/tmp/ce_vliw_nonstream/llo`
    - Mosaic: `/tmp/ce_vliw_nonstream/mosaic`
  - matmul-dominant non-streaming run:
    - HLO: `/tmp/ce_vliw_matmul/hlo`
    - LLO: `/tmp/ce_vliw_matmul/llo`
    - Mosaic: `/tmp/ce_vliw_matmul/mosaic`

- Key LLO schedule-analysis bundle counts:
  - full non-streaming custom call:
    - file: `/tmp/ce_vliw_nonstream/llo/1772746582910987391-_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined.1-78-schedule-analysis_final_bundles.txt`
    - scheduled bundles: `33035`
  - matmul-dominant non-streaming custom call (`skip_logits_recompute=1`):
    - file: `/tmp/ce_vliw_matmul/llo/1772746972106552048-_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined.1-78-schedule-analysis_final_bundles.txt`
    - scheduled bundles: `19059`
  - streaming bwd path key fusions (same source location metadata):
    - `fusion.41`: `11658` bundles
    - `fusion.45`: `8877` bundles
    - `fusion.42`: `117` bundles
    - files under `/tmp/ce_vliw/llo_stream/*fusion.4{1,2,5}-*-schedule-analysis_final_bundles.txt`

### 2026-03-05: rerun after local fix + supertile compile/VMEM findings
- User note: "ok i think i fixed it"; reran v4-8 benchmark at the same shape/config:
  - `B=262144, H=1024, V=128256`
  - block sizes `b=1024, h=1024, v=256`
  - `--shard-map --data-shards 4`
- Non-streaming rerun:
  - `bwd_tokens_per_s ~384,487`
  - `combined_tokens_per_s ~338,696`
- Forced streaming rerun (`--bwd-use-xla-streaming`):
  - `bwd_tokens_per_s ~507,856`
  - `combined_tokens_per_s ~430,919`
- Added supertile alignment hint in
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
  - change: in the supertile kernel, `w_grad` sub-tile V start now uses
    `pl.multiple_of(v_sub_start, NUM_LANES)` before `pl.ds(...)`.
  - intent: fix Mosaic inability to prove tile divisibility for HBM tiled `memref_slice`.
- Supertile run with `LEVANTER_PALLAS_TPU_BWD_SOFTMAX_V_BLOCK_SIZE_BENCH=768`:
  - previous failure mode (`MosaicError` divisibility proof) no longer reproduced.
  - new failure mode: compile-time VMEM OOM (`19.03M` required vs `16.00M` capacity).
  - top contributors include `f32[1024,1024]`, `f32[1024,768]`, `f32[1024,256]`, plus spill slots.

### 2026-03-05: fresh v4-8 dump matrix (streaming vs non-streaming decomposition)
- Shape/config (all runs):
  - `B=262144, H=1024, V=128256`
  - block sizes `b=1024, h=1024, v=256`
  - `--shard-map --data-shards 4`
- Dump dirs:
  - base: `/tmp/ce_vliw_20260305_rerun`
  - variants:
    - `nonstream_full`
    - `streaming_bwd`
    - `nonstream_skip_softmax`
    - `nonstream_skip_label_subtract`
    - `nonstream_skip_logits_recompute`
- Throughput (`bwd_tokens_per_s`, `combined_tokens_per_s`):
  - `nonstream_full`: `~384,488`, `~338,695`
  - `streaming_bwd`: `~507,831`, `~430,846`
  - `nonstream_skip_softmax`: `~398,902`, `~349,817`
  - `nonstream_skip_label_subtract`: `~391,360`, `~344,025`
  - `nonstream_skip_logits_recompute`: `~539,604`, `~453,479`
- Additional ablation:
  - `skip_softmax + skip_label_subtract` fails compile (with and without dumps):
    - VMEM OOM `18.97M` vs `16.00M`
    - largest allocation is RA spill slots `~7.94M`.

- LLO schedule-analysis focus (files with `source_line=1216`, non-empty bundles):
  - `nonstream_full`: 5 files, sum `34589`
    - custom call `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined`: `33035`
  - `streaming_bwd`: 13 files, sum `33077`
    - `fusion.41`: `11658`
    - `fusion.45`: `8877`
    - `fusion.44`: `8287`
    - remainder spread over smaller ops (`TLP`, `reshape`, `compare_select`, `slice`, `fusion.42`, etc.).
  - `nonstream_skip_softmax`: 5 files, sum `32900`
    - custom call: `31346`
  - `nonstream_skip_label_subtract`: 5 files, sum `33867`
    - custom call: `32313`
  - `nonstream_skip_logits_recompute`: 5 files, sum `20613`
    - custom call: `19059`

- Readout:
  - Removing only softmax (`skip_softmax`) or only label subtraction (`skip_label_subtract`) gives modest speedups.
  - The large gain comes from removing logits-recompute path (`skip_logits_recompute`), matching earlier hypothesis that the dominant non-streaming penalty is remat/softmax-side scaffolding, not backward GEMM core.
  - Streaming backward remains faster than full non-streaming despite similar aggregate source-line bundle totals, because work is split across multiple smaller fusions instead of one large custom-call monolith.

### 2026-03-05: splash-style split backward prototype (dq/dkv-inspired)
- Goal:
  - try a backward decomposition patterned after Splash attention backward:
    - separate `dx`-like and `dw`-like kernels (bench-only), each with simpler responsibilities.
- Code:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - added env flag: `LEVANTER_PALLAS_TPU_BWD_SPLIT_DX_DW_BENCH`
    - added split kernels:
      - `linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel_dx_only(...)`
      - `linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel_dw_only(...)`
    - added split wrappers:
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_dx_only(...)`
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_dw_only(...)`
    - wired `split_dx_dw_bench` through:
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(...)`
      - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(...)`
      - `_make_custom_vjp(...)`
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - updated fake `_make_custom_vjp` signatures and assertions for the new split flag plumbing.
- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`:
    - `25 passed, 12 skipped`.
  - `./infra/pre-commit.py --all-files`: `OK`.
- v4-8 benchmark (`B=262144,H=1024,V=128256,b/h/v=1024/1024/256`):
  - split path (`LEVANTER_PALLAS_TPU_BWD_SPLIT_DX_DW_BENCH=1`):
    - `bwd_tokens_per_s ~283,204`
    - `combined_tokens_per_s ~257,555`
  - baseline non-streaming on same code:
    - `bwd_tokens_per_s ~384,518`
  - forced streaming:
    - `bwd_tokens_per_s ~507,815`
  - split + `skip_logits_recompute`:
    - `bwd_tokens_per_s ~490,293`
    - still below non-split `skip_logits_recompute` (`~539k`).
- Readout:
  - this first splash-style split decomposition is functionally correct but slower at full backward because remat/softmax work is duplicated across separate `dx` and `dw` kernels.
  - even in matmul-dominant mode, split overhead remains measurable (`~490k` vs `~539k`).

### 2026-03-05: splash-style `block_kv_compute` experiments for CE backward
- Goal:
  - mimic Splash dkv style more closely by decoupling:
    - `softmax/remat V tile` (memory tile)
    - `inner compute V subtile` (compute tile)
  - and check whether this removes non-streaming backward spill/perf gap.
- Code:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - added bench env:
      - `LEVANTER_PALLAS_TPU_BWD_COMPUTE_V_BLOCK_SIZE_BENCH`
    - added/iterated kernel:
      - `linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel_vsubtile(...)`
    - wired `compute_v_block_size_bench` through:
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(...)`
      - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(...)`
      - `_make_custom_vjp(...)`
    - current guard:
      - compute-vsubtile path requires `num_h_blocks == 1` (current implementation assumption).
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - updated `_make_custom_vjp` fake signatures/assertions for the new arg.
- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`:
    - `25 passed, 12 skipped`.
- v4-8 benchmark (`B=262144,H=1024,V=128256,b/h/v=1024/1024/256`, `--shard-map --data-shards 4`):
  - baseline non-streaming:
    - `bwd_tokens_per_s ~384,214`
  - forced streaming:
    - `bwd_tokens_per_s ~507,680`
  - compute-subtile (`LEVANTER_PALLAS_TPU_BWD_COMPUTE_V_BLOCK_SIZE_BENCH=128`):
    - `bwd_tokens_per_s ~288,636` (slower)
  - compute-subtile + no-remat (`...COMPUTE_V_BLOCK_SIZE_BENCH=128`, `...SKIP_LOGITS_RECOMPUTE_BENCH=1`):
    - `bwd_tokens_per_s ~378,128`
    - confirms compute-subtile fragmentation slows matmul core itself vs prior non-split matmul-dominant path (`~539k`).
- Supertile+compute attempts:
  - `softmax_v=768, compute_v=128`:
    - still compile OOM at `16.74M / 16.00M` VMEM (overflow `~760 KiB`)
    - spill slots around `~3.93M`
  - `softmax_v=768, compute_v=256`:
    - compile OOM much worse (`20.33M / 16.00M`, spills `~7.89M`)
- Spill-focused edits attempted:
  - replaced conditional init with scalar-mask multiply (`tile *= keep`) to reduce temporaries.
  - attempted direct HBM load/store (no async copy) but TPU Pallas rejected it (`Loads are only allowed on VMEM and SMEM references`), so reverted.
- Readout:
  - the current compute-subtile CE backward path is not yet competitive:
    - slower when it compiles (`128` compute subtile).
    - supertile+compute variant remains just over VMEM limit due spills.
  - next work should stay spill-first:
    - simplify kernel control/dataflow to cut RA spill slots before any further tile-sweep.

### 2026-03-06: XLA vs Pallas LLO comparison and scatter-label subtraction attempt
- Goal:
  - inspect XLA LLO for backward and align Pallas structure with what XLA is doing well.

- Dump setup (v4-8, shape `B=262144,H=1024,V=128256`, block sizes `1024/1024/256`):
  - XLA path dumps at `/tmp/ce_llo_compare_20260306/xla/llo`
  - Pallas path dumps at `/tmp/ce_llo_compare_20260306/pallas/llo`

- Key XLA structure from `final_bundles`:
  - backward is split into three large fusions:
    - `fusion.50` (produces `f32[8192,8,8,128]` temporary-like block)
    - `fusion.53` (updates `bf16[65536,1024]`, aliases an input/output)
    - `fusion.54` (updates `bf16[1024,129024]`, aliases an input/output)
  - this matches a split pattern: softmax-ish block production + separate matmul-style updates.

- Instruction mix comparison (from `*-79-final_bundles.txt`):
  - XLA (sum of `fusion.50/53/54`) has much lower lane-rotation overhead in each fusion.
  - Pallas combined custom-call (`_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined`) shows:
    - `vrot.slane.down`: `10155`
    - `vpow2.f32`: `2048`
    - `vsel`: `4098`
    - `vld`: `14143`, `vst`: `10387`
    - indicates a heavy softmax/masking/reduction lane-shuffle footprint in a single monolithic kernel.

- Perf re-check (same shape):
  - baseline non-streaming pallas bwd: `~384,459 tok/s`
  - forced streaming bwd: `~507,8xx tok/s` (unchanged from prior runs)

- Ablations to isolate cost centers:
  - `LEVANTER_PALLAS_TPU_BWD_SKIP_SOFTMAX_BENCH=1`:
    - bwd `~398,832 tok/s` (small +3.7% gain)
  - `LEVANTER_PALLAS_TPU_BWD_SKIP_LOGITS_RECOMPUTE_BENCH=1`:
    - bwd `~539,770 tok/s` (large gain, surpasses streaming baseline)
  - `SKIP_SOFTMAX + SKIP_LABEL_SUBTRACT`:
    - compile OOM (`18.97M / 16.00M` VMEM), spill slots reported as largest allocation (`~7.94M`).

- Readout from ablations:
  - dominant gap is not the exp/softmax primitive by itself; it is the current logits-rematerialization path as lowered in the monolithic kernel.
  - this aligns with user hypothesis to first match XLA matmul/backward structure, then layer in complications.

- Code experiment implemented:
  - added bench-only env flag `LEVANTER_PALLAS_TPU_BWD_SCATTER_LABEL_SUBTRACT_BENCH` and threaded it through backward kernels:
    - switched label subtraction from emulated one-hot compare to row-wise scatter-add update when enabled.
  - tests updated and passing (`25 passed, 12 skipped`).

- Scatter experiment result:
  - no meaningful perf change (`~384,484 tok/s`, essentially baseline).
  - LLO instruction counts stayed effectively identical (`vrot.slane.down` remained `10155`).
  - implies current lowering canonicalizes back to the same effective kernel shape; this is not the lever.

- Next direction (based on XLA LLO):
  - prototype an explicit split backward pipeline that mirrors XLA’s decomposition:
    - split softmax/delta production from dx/dw matmul updates,
    - avoid monolithic combined custom-call control flow and associated spill pressure.

### 2026-03-06: bench-only split softmax/matmul backward pipeline prototype
- Goal:
  - implement the XLA-like staged backward decomposition directly in `pallas_tpu.py` for A/B:
    - stage A: produce `delta` as a rematerialized softmax V-super-tile,
    - stage B: consume that `delta` in smaller grad-V subtiles to update `dx` and `dw`.
- Code changes:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - added new bench env flag:
      - `LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH`
    - added delta supertile producer kernel/wrapper:
      - `linear_softmax_cross_entropy_loss_backward_pallas_delta_supertile_kernel(...)`
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_delta_supertile(...)`
    - added split pipeline wrapper:
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_split_softmax_matmul(...)`
    - wired flag through:
      - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(...)`
      - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(...)`
      - `_make_custom_vjp(...)`
      - `linear_softmax_cross_entropy_loss_pallas(...)`
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - updated `_make_custom_vjp` monkeypatch signatures/assertions for new flag plumbing.
- Local validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `25 passed, 12 skipped`
  - `./infra/pre-commit.py --all-files`
    - `OK`
- Next step:
  - run v4-8 A/B with dumps:
    - baseline non-streaming combined
    - split softmax/matmul path (`LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH=1`)
    - forced streaming
  - compare `bwd_tokens_per_s` plus LLO bundle counts/spill slots.

### 2026-03-06: first v4-8 A/B for split softmax/matmul path
- Environment:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central2-staging.yaml` (`TPU v4`, 4 chips)
  - bench shape/config:
    - `B=262144, H=1024, V=128256` (`batch=64,pos=4096,embed=1024,vocab=128256`, `--shard-map --data-shards 4`)
    - `b/h/v = 1024/1024/256`
- Baselines (same command family):
  - non-streaming combined:
    - `bwd_tokens_per_s ~384,524`
    - `combined_tokens_per_s ~338,684`
  - forced streaming (`--bwd-use-xla-streaming`):
    - `bwd_tokens_per_s ~507,567`
    - `combined_tokens_per_s ~430,691`
- Split softmax/matmul path (`LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH=1`):
  - `softmax_v` default (=256):
    - `bwd_tokens_per_s ~325,458`
    - `combined_tokens_per_s ~292,035`
  - `softmax_v=512` (`LEVANTER_PALLAS_TPU_BWD_SOFTMAX_V_BLOCK_SIZE_BENCH=512`):
    - `bwd_tokens_per_s ~298,086`
    - `combined_tokens_per_s ~269,780`
  - `softmax_v=1024`:
    - compile VMEM OOM (`18.20M / 16.00M`)
    - largest allocation: spill slots (`~8.01M`) in `_linear_softmax_cross_entropy_loss_bwd_pallas_delta_supertile`.
- Implementation fixes made during this run:
  - fixed split-path delta producer lowering constraints:
    - switched HBM output handling to VMEM staging + async copy (TPU-compatible write path).
  - this resolved earlier lowering errors (`HBM BlockSpec` restriction and direct HBM load/store rejection), but did not resolve spill-driven OOM/perf issues at larger `softmax_v`.
- Readout:
  - split-softmax/matmul as currently implemented does not close the gap to streaming; it is slower than baseline non-streaming at `softmax_v <= 512` and OOMs at `softmax_v=1024` due spills.
  - current bottleneck remains VMEM spill pressure in the delta producer kernel.

### 2026-03-06: replicate XLA structure in split path (replace Pallas delta producer)
- Goal:
  - make split non-streaming path structurally match XLA backward fusions by removing the Pallas delta custom-call.
- Code changes:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - added `_linear_softmax_cross_entropy_loss_bwd_xla_delta_supertile(...)`:
      - pure JAX/XLA delta stage (`dot -> mask -> exp/softmax -> label subtract`) with `dynamic_update`-style row scatter option,
      - no Pallas custom-call in this stage.
    - updated `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_split_softmax_matmul(...)` to call the new XLA-style delta producer.
    - split path remains bench-only (`LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH=1`).
- Local validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `25 passed, 12 skipped`
  - `./infra/pre-commit.py --all-files`
    - `OK`
- v4-8 benchmark (`B=262144,H=1024,V=128256,b/h/v=1024/1024/256`, `--shard-map --data-shards 4`):
  - baseline non-streaming:
    - `bwd_tokens_per_s ~384,491`
    - `combined_tokens_per_s ~338,685`
  - forced streaming (`--bwd-use-xla-streaming`):
    - `bwd_tokens_per_s ~507,846`
    - `combined_tokens_per_s ~430,884`
  - split path with new XLA-style delta stage (`LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH=1`):
    - `bwd_tokens_per_s ~500,664`
    - `combined_tokens_per_s ~425,655`
  - readout:
    - split path now closes almost all the gap to streaming (within ~1.4% on backward throughput).
- LLO evidence (new dump: `/tmp/ce_llo_xla_repl/split/llo`):
  - focused kernels:
    - `fusion.28`: `9321` bundles (`dot_general` in split while body)
    - `convolution_dynamic-update-slice_fusion.2`: `8812`
    - `convolution_add_fusion.2`: `3445`
  - aggregate focused instruction deltas vs old split dump (`/tmp/ce_llo_quick/split/llo`):
    - `vrot.slane.down`: `19213 -> 0`
    - `vpow2.f32`: `2048 -> 512`
    - `vsel`: `8825 -> 2879`
  - key register pressure snapshots:
    - old split delta custom-call: `vregs=1944`
    - new split delta fusion (`fusion.28`): `vregs=1799`
  - interpretation:
    - removing the Pallas delta custom-call eliminated lane-rotation-heavy lowering and recovered most of the throughput gap.

### 2026-03-06: split-path matmul refactor (super-tile GEMMs) + softmax_v scaling win
- Goal:
  - remove remaining split-path overhead after XLA-style delta stage by:
    - hoisting loop-invariant backward scalars/casts out of the softmax-block loop,
    - replacing grad-v subtile inner loops with direct super-tile GEMMs.

- Code change (`lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`):
  - `_linear_softmax_cross_entropy_loss_bwd_xla_delta_supertile(...)`
    - switched from `(dout_loss, dout_lse)` per-iteration casts to precomputed inputs:
      - `dout_loss_lse`
      - `dout_loss_plus_lse`
    - moved `labels -> int32` and scatter row-index creation to conditional paths.
  - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_split_softmax_matmul(...)`
    - precomputes `dout_loss_lse` + `dout_loss_plus_lse` once before `fori_loop`.
    - removes inner grad-subtile loop and uses direct super-tile updates:
      - `dx += dot(delta_supertile, w_supertile)`
      - `dw_supertile = dot(x, delta_supertile)` + `dynamic_update_slice`.

- Local validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `25 passed, 12 skipped`.

- v4-8 benchmark re-baseline (same shape/config):
  - shape: `B=262144,H=1024,V=128256` (`batch=64,pos=4096,embed=1024,vocab=128256`)
  - blocks: `b/h/v = 1024/1024/256`
  - launch: `--shard-map --data-shards 4`
  - non-streaming combined baseline:
    - `bwd_tokens_per_s ~384,510`
  - forced streaming (`--bwd-use-xla-streaming`):
    - `bwd_tokens_per_s ~507,243`

- Split path results after refactor (`LEVANTER_PALLAS_TPU_BWD_SPLIT_SOFTMAX_MATMUL_BENCH=1`):
  - `softmax_v=256` (default):
    - `bwd_tokens_per_s ~500,635` (about same as prior split)
  - `softmax_v=512`:
    - `bwd_tokens_per_s ~708,756`
  - `softmax_v=768`:
    - `bwd_tokens_per_s ~809,519`
  - `softmax_v=1024`:
    - `bwd_tokens_per_s ~850,804`
  - `softmax_v=2048`:
    - `bwd_tokens_per_s ~839,336`

- Readout:
  - this refactor fully removed the prior `softmax_v>256` regression and shifted the split-path optimum near `softmax_v=1024`.
  - at `softmax_v=1024`, split-path backward is now substantially faster than forced streaming on this workload:
    - `~850,804` vs `~507,243` tokens/s (`~1.68x`).

- Correctness spot-check (TPU, small shape):
  - script compares `pallas_tpu` split-path vs `xla` gradients at `tokens=16384, embed=512, vocab=4096`.
  - `softmax_v=1024` metrics:
    - `loss_abs = 7.63e-06`
    - `dx_rel_linf ~ 1.16e-2`
    - `dw_rel_linf ~ 3.29e-3`
  - result indicates no obvious regression from the super-tile GEMM rewrite.

### 2026-03-06: promote split/supertile backward to default + retune check
- Goal:
  - make the split/supertile backward path the default Pallas backward and remove old-path toggles.

- Code cleanup:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
    - `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(...)` now always dispatches to `_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_split_softmax_matmul(...)`.
    - removed old backward dispatch knobs from runtime path:
      - `split_dx_dw_bench`
      - `split_softmax_matmul_bench`
      - `compute_v_block_size_bench`
    - `_make_custom_vjp(...)` signature/call sites updated accordingly.
    - default softmax supertile policy in backward wrapper:
      - if no explicit override is set, use `softmax_v = max(1024, v_block_size)` rounded to a multiple of `v_block_size`.
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - updated monkeypatched `_make_custom_vjp` expectations for removed args.

- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `25 passed, 12 skipped`
  - `./infra/pre-commit.py --all-files`
    - `OK`

- Throughput check (v4-8, `B=262144,H=1024,V=128256`, `b/h/v=1024/1024/256`):
  - default non-streaming Pallas (new default backward):
    - `bwd_tokens_per_s ~850,739`
    - `combined_tokens_per_s ~654,819`
  - forced streaming (`--bwd-use-xla-streaming`):
    - `bwd_tokens_per_s ~507,779`
    - `combined_tokens_per_s ~430,808`

- Block-size retune sweep (combined fwd+bwd tune script):
  - command grid:
    - `b=1024`, `h in {256,512,1024}`, `v in {256,512,1024,2048}`
  - outcomes:
    - only `v=256` compiled; `v>=512` still failed VMEM in forward/JVP path.
    - successful results:
      - `b1024_h256_v256`: `~900,840 tok/s`
      - `b1024_h512_v256`: `~900,863 tok/s`
      - `b1024_h1024_v256`: `~900,954 tok/s` (best, marginal)
  - `b=2048` remains compile OOM (spot check `b2048_h1024_v256`).

- Readout:
  - backward performance is now mostly insensitive to `h_block_size` for this shape (`~850.7k tok/s` for `h=256/512/1024` at `v=256`), consistent with the new split/supertile design.
  - tuned `b/h/v` recommendation for this workload does not change materially from current table (`b1024_h1024_v256`).

### 2026-03-06: cleanup pass remove backward experiment hacks from runtime path
- Removed backward experiment env toggles from runtime/VJP plumbing:
  - `LEVANTER_PALLAS_TPU_BWD_SKIP_SOFTMAX_BENCH`
  - `LEVANTER_PALLAS_TPU_BWD_SKIP_LABEL_SUBTRACT_BENCH`
  - `LEVANTER_PALLAS_TPU_BWD_SCATTER_LABEL_SUBTRACT_BENCH`
  - `LEVANTER_PALLAS_TPU_BWD_SKIP_LOGITS_RECOMPUTE_BENCH`
  - `LEVANTER_PALLAS_TPU_BWD_V_BLOCK_SIZE_BENCH`
  - `LEVANTER_PALLAS_TPU_BWD_SOFTMAX_V_BLOCK_SIZE_BENCH`
- Simplified default backward API:
  - `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(...)` no longer accepts skip/bench knobs.
  - `_make_custom_vjp(...)` no longer takes skip/bench parameters.
- Kept only `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH` as an explicit A/B switch.
- Default non-streaming backward remains split/supertile with fixed policy:
  - `softmax_v = max(1024, v_block_size)` (rounded up to a multiple of `v_block_size`).
- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `25 passed, 12 skipped`
  - `./infra/pre-commit.py --all-files` -> `OK`
- TPU sanity (v4-8, `B=262144,H=1024,V=128256`, `b/h/v=1024/1024/256`):
  - default non-streaming: `bwd_tokens_per_s ~850,780`
  - forced streaming: `bwd_tokens_per_s ~507,813`

### 2026-03-06: full v4 sweep follow-up (OOM attribution + block-size policy)
- Sweep scope:
  - ran the full v4 bucket sweep plus rescue grids for buckets with no successful configs:
    - `llama3_ish_rescue`
    - `medium_batch_mid_h_rescue`
  - logs/artifacts under:
    - `/tmp/ce_sweep_v4_20260306/*.stdout.log`
    - `/tmp/ce_sweep_v4_20260306/*.compile.log`

- OOM attribution:
  - all observed OOM failures in this sweep were forward-path compile OOMs.
  - no backward-attributed OOMs were found in bucket or rescue runs.
  - failure traces consistently referenced forward/JVP paths (for example `...loss_fwd_pallas_mosaic_tpu...` sites).

- Decision on separate fwd/bwd block sizes:
  - based on this sweep, separate forward/backward block-size knobs are not needed as an OOM mitigation.
  - if split knobs are introduced later, it should be for throughput-only tuning, not to address these OOMs.

- v4 tuned-table impact:
  - keep `TPU v4` `huge-batch-small-h` entry at `b=1024,h=1024,v=256` (bf16/f32).
  - for this path, retune deltas among successful `v=256` variants were marginal and did not justify additional table fragmentation.

### 2026-03-06: full v5p sweep (`v5p-8`) + tuned-table updates
- Setup:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-type v5p-8`
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
  - artifacts on TPU: `/tmp/ce_sweep_v5p_20260306`
- Sweep buckets run:
  - `small_vocab`, `mid_h_large_vocab`, `large_batch_small_h`, `huge_batch_small_h`, `llama3_ish`, `medium_batch_mid_h`
  - each with `b_block_size=1024`, explicit `h/v` grids, and `--include-infer`.
- Best observed configs from this sweep:
  - `small_vocab`: `b1024_h512_v1024` (`~1.656M tok/s`)
  - `mid_h_large_vocab`: `b1024_h256_v1024` (`~362.7k tok/s`)
  - `large_batch_small_h`: `b1024_h512_v2048` (`~768.2k tok/s`)
  - `huge_batch_small_h`: `b1024_h128_v1024` (`~410.4k tok/s`; close tie with `h=256/512/1024`)
  - `llama3_ish`: `b1024_h256_v512` (`~79.5k tok/s`)
  - `medium_batch_mid_h`: `b1024_h2048_v512` (`~168.2k tok/s`; near tie with `h=256/512/1024`)
- Failure attribution:
  - aggregate failed classifications across all six buckets:
    - `forward`: `27`
    - `backward`: `0`
    - `unsupported (shape-divisibility)`: `40`
    - `other`: `0`
  - all VMEM OOMs observed in this run were forward/JVP-path OOMs.
- Tuned table updates applied (`tuned_block_sizes.py`):
  - `TPU v5p` and `TPU v5` (both bf16 and float32):
    - `small-vocab`: `v_block_size 512 -> 1024`
    - `llama3-ish`: `v_block_size 1024 -> 512`
    - `large-batch-small-h`: `v_block_size 1024 -> 2048`
    - `medium-batch-medium-h`: `v_block_size 2048 -> 512`
  - retained `h_block_size` choices from existing bucket definitions so entries remain valid across the full bucket ranges.

### 2026-03-06: full v5e sweep (`v5litepod-8`) + tuned-table updates
- Setup:
  - `scripts/ray/dev_tpu.py --config infra/marin-us-west4.yaml --tpu-type v5litepod-8`
  - TPU reported `device_kind="TPU v5 lite"`.
  - `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000`
  - artifacts on TPU: `/tmp/ce_sweep_v5e_20260306`
- Sweep buckets run:
  - `small_vocab`, `mid_h_large_vocab`, `large_batch_small_h`, `huge_batch_small_h`, `llama3_ish`, `medium_batch_mid_h`
  - each with `b_block_size=1024`, explicit `h/v` grids, and `--include-infer`.
- Best observed configs from this sweep:
  - `small_vocab`: `b1024_h128_v1024` (`~1.521M tok/s`)
  - `mid_h_large_vocab`: `b1024_h512_v1024` (`~164.1k tok/s`)
  - `large_batch_small_h`: `b1024_h128_v2048` (`~328.3k tok/s`)
  - `huge_batch_small_h`: `b1024_h128_v1024` (`~161.8k tok/s`; `h=256/512/1024` within noise)
  - `llama3_ish`: `b1024_h512_v512` (`~30.8k tok/s`)
  - `medium_batch_mid_h`: `b1024_h256_v512` (`~66.0k tok/s`; near tie with `h=512/1024/2048`)
- Failure attribution:
  - aggregate failed classifications across all six buckets:
    - `forward`: `25`
    - `backward`: `0`
    - `unsupported (shape-divisibility)`: `35`
    - `other`: `0`
  - all VMEM OOMs observed in this run were forward/JVP-path OOMs.
- Tuned table updates applied (`tuned_block_sizes.py`):
  - `TPU v5e` (both bf16 and float32):
    - `small-vocab`: `v_block_size 512 -> 1024`
    - `llama3-ish`: `v_block_size 1024 -> 512`
    - add `mid-h-large-vocab`: `b=1024,h=256,v=1024`
    - add `huge-batch-small-h`: `b=1024,h=256,v=1024`
    - add `medium-batch-medium-h`: `b=1024,h=256,v=512`
    - keep `large-batch-small-h` at `b=1024,h=512,v=2048`
  - device-key normalization update:
    - map both `TPU v5e` and `TPU v5 lite` / `TPU v5litepod` strings to `TPU v5e` so v5litepod-8 uses the v5e table directly.

### 2026-03-06: full v6e sweep (`v6e-4`, us-east1) + TPU v6 tuned-table updates
- Setup:
  - `dev_tpu.py allocate` remained blocked by Ray client timeout in this session, so the sweep ran as a Ray submission job pinned to `TPU-v6e-4-head` resources.
  - job id: `raysubmit_HkNExTDUP22WCKkJ` (`infra/marin-us-east1.yaml`)
  - artifacts on TPU: `/tmp/ce_sweep_v6e_20260306/*.stdout.log`
- Sweep buckets run:
  - `small_vocab`, `mid_h_large_vocab`, `large_batch_small_h`, `huge_batch_small_h`, `llama3_ish`, `medium_batch_mid_h`
  - each with `b_block_size=1024`, explicit `h/v` grids, and `--include-infer`.
- Best observed configs from this sweep:
  - `small_vocab`: `b1024_h512_v2048` (`~2.399M tok/s`)
  - `mid_h_large_vocab`: `b1024_h1024_v1024` (`~410.9k tok/s`)
  - `large_batch_small_h`: `b1024_h256_v2048` (`~522.2k tok/s`)
  - `huge_batch_small_h`: `b1024_h512_v1024` (`~318.3k tok/s`; `h=256/512/1024` near tie)
  - `llama3_ish`: `b1024_h1024_v512` (`~70.6k tok/s`; `h=256/512/1024` near tie at `v=512`)
  - `medium_batch_mid_h`: `b1024_h2048_v512` (`~155.2k tok/s`; `h=256/512/1024/2048` near tie at `v=512`)
- Failure attribution:
  - aggregate failed classifications across all six buckets:
    - `forward`: `22`
    - `backward`: `0`
    - `unsupported (shape-divisibility)`: `8`
    - `other`: `0`
  - all VMEM OOM failures observed in this run were forward/JVP-path failures.
- Tuned table updates applied (`tuned_block_sizes.py`, `TPU v6`, bf16 + float32):
  - `small-vocab`: `v_block_size 512 -> 2048`
  - `llama3-ish`: `v_block_size 1024 -> 512`
  - add `huge-batch-small-h`: `b=1024,h=256,v=1024`
  - add `medium-batch-medium-h`: `b=1024,h=256,v=512`
  - kept `mid-h-large-vocab` and `large-batch-small-h` entries unchanged.

### 2026-03-06: v4 robustness pass for Llama2/GPT-2/Llama3 vocab families
- Goal:
  - ensure v4 infer chooses robust tuned regimes for practical shapes when
    `V` is Llama2-like (`32k`), GPT-2-like (`50,257`), or Llama3 (`128,256`).
- Code change (`tuned_block_sizes.py`):
  - added TPU v4 mid-vocab bucket extension logic:
    - when the primary bucket lookup misses and `16,385 <= V <= 119,999`,
      map shapes into existing tuned v4 regimes (`small-vocab`, `llama3-ish`,
      `mid-h-large-vocab`, `large-batch-small-h`, `huge-batch-small-h`,
      `medium-batch-medium-h`) instead of falling back to untuned defaults.
  - this reuses already-proven v4 block-size entries and avoids introducing
    a second independently tuned table for the same execution regimes.
- Tests:
  - added `test_infer_block_sizes_tpu_v4_mid_vocab_uses_tuned_buckets` covering
    representative shapes for:
    - Llama2 vocab (`32,000`)
    - GPT-2 vocab (`50,257`)
    - Llama3 vocab (`128,256`) sanity case
  - assertions require both:
    - `tuned_match=True`
    - expected v4 block sizes for each regime.
- Validation:
  - `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -q`
    - `55 passed, 12 skipped`
- Infra notes:
  - attempted a broad live v4 Ray coverage sweep, but shared-cluster head-node
    memory pressure and intermittent submission tunnel resets prevented collecting
    a complete on-cluster matrix in this session.

### 2026-03-06: head-to-head vs XLA on currently tuned infer path (cross-platform)
- Goal:
  - run `pallas_tpu` vs `xla` head-to-head on the currently tuned infer path across platforms and record the results in one place.

- Runner:
  - `python scripts/ray/ce_h2h_platform_local.py`
  - fixed settings:
    - `H2H_STEPS=3`
    - `H2H_WARMUP=1`
    - script-level `LIBTPU_INIT_ARGS+=--xla_tpu_scoped_vmem_limit_kib=50000`
  - each case invokes:
    - `uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py ... --block-sizes infer --implementation <pallas_tpu|xla>`

- Submitted jobs and status:
  - `v4` (`infra/marin-us-central2.yaml`):
    - job: `raysubmit_x6rNA8RpxZgCgjvL`
    - status: `SUCCEEDED` (`2026-03-06 12:02:35 PST` -> `12:07:58 PST`)
  - `v5p` (`infra/marin-us-central1.yaml`):
    - job: `raysubmit_QFvDsEk4pUwbpr5B`
    - status: `SUCCEEDED` (`2026-03-06 12:02:54 PST` -> `12:13:43 PST`)
  - `v5e` (`infra/marin-eu-west4.yaml`):
    - job: `raysubmit_y6mD81ysjCJ4gh8V`
    - status: `SUCCEEDED` (`2026-03-06 11:48:14 PST` -> `12:13:36 PST`)
  - `v6e`:
    - east1 status API remained unavailable (`infra/marin-us-east1.yaml` list-jobs returns dashboard `Connection refused` as of `2026-03-06 12:42 PST`).
    - fallback attempt on existing east5 cluster:
      - job: `raysubmit_v6e_h2h_20260306_122610` (`infra/marin-us-east5.yaml`)
      - status at capture time: `PENDING` (waiting for resources/runtime env), started `2026-03-06 12:26:43 PST`.
      - stop attempts:
        - `cluster.py stop-job` timed out in this session.
        - direct `ray job stop --no-wait` was accepted, but job still reported `PENDING` as of `2026-03-06 12:45 PST`.

- Aggregated h2h results (successful platforms):
  - `v4`:
    - `ok_pairs=2/6`, `failed_pairs=4`
    - `avg_xla_over_pallas_combined=0.7412` (pallas `~+34.9%` combined on successful pairs)
    - `avg_xla_over_pallas_bwd=0.8475` (pallas `~+18.0%` bwd on successful pairs)
  - `v5p`:
    - `ok_pairs=6/6`, `failed_pairs=0`
    - `avg_xla_over_pallas_combined=0.7772` (pallas `~+28.7%` combined)
    - `avg_xla_over_pallas_bwd=0.8135` (pallas `~+22.9%` bwd)
  - `v5e`:
    - `ok_pairs=6/6`, `failed_pairs=0`
    - `avg_xla_over_pallas_combined=0.7666` (pallas `~+30.4%` combined)
    - `avg_xla_over_pallas_bwd=0.7731` (pallas `~+29.4%` bwd)

- Per-bucket highlights:
  - `v4`:
    - successful buckets:
      - `mid_h_large_vocab`: xla/pallas combined `0.5405` (pallas substantially ahead)
      - `huge_batch_small_h`: xla/pallas combined `0.9419` (pallas slight combined lead), but xla bwd higher on this bucket (`1.1315` ratio).
    - pallas failures in this run on:
      - `small_vocab`
      - `large_batch_small_h`
      - `llama3_ish` (forward compile VMEM OOM in log)
      - `medium_batch_mid_h`
  - `v5p`:
    - pallas leads combined in 5/6 buckets; near parity on `llama3_ish` (`xla/pallas combined=0.9920`).
  - `v5e`:
    - pallas leads combined in 5/6 buckets; `xla` leads `llama3_ish` (`xla/pallas combined=1.0729`).

- Instrumentation caveat (important):
  - these h2h summary blobs report top-level `backend="cpu"` / `device_kind="cpu"`.
  - logs also contain JAX warnings about falling back to CPU in the wrapper process.
  - the per-case benchmark subprocesses still run via the TPU-enabled bench invocation (`--extra tpu`) and produce TPU-scale throughput numbers; so throughput comparisons above are still useful, but the wrapper-level backend metadata is misleading.
  - follow-up hygiene item: plumb backend/device-kind from the subprocess `result_json` into the h2h summary to remove this ambiguity.
