# Grug FA4 CuTe Hopper: Research Logbook

## Scope
- Goal: Determine why Grug `gpu_fa4_cute` attention backward is slow on the May208 H100 profile, and add a Hopper path if the current path is leaving performance on the table.
- Primary metrics: May208 FA4 forward/backward kernel time per device per profiled step, especially backward main-kernel time and side-kernel overhead.
- Constraints: Keep changes local to the Grug FA4 CuTe path unless evidence justifies broader FA4 wiring; preserve CPU-safe tests for config and launcher selection; use GPU repro or microbench before making strong performance claims.

## Baseline
- Date: 2026-06-23 14:45 PDT
- Code refs:
  - `lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py`
  - `lib/levanter/src/levanter/grug/attention/_fa4_cute.py`
  - `lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py`
  - `lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py`
  - `lib/levanter/src/levanter/grug/attention/_fa4_thd.py`
- Profile refs:
  - `/Users/dlwh/.codex/worktrees/24a6/marin/scratch/rooflines/may208.json`
  - `/Users/dlwh/.codex/worktrees/079b/marin/scratch/profiles/may208`
- Baseline numbers from brief: `attention_fa4` is about 51.9 ms/device, split roughly into 26.6 ms backward main, 10.8 ms forward main, 11.2 ms rematted forward main, and 3.3 ms side kernels. Major rows have count 416, matching 26 layers * 8 devices * 2 profiled steps.

## Experiment Log

### 2026-06-23 14:45 - Initial dispatch trace
- Hypothesis: H100/SM90 `gpu_fa4_cute` is not using a Hopper-specific segmented backward path, because the launcher infers SM120 only for `(tile_m, tile_n, num_threads) == (64, 64, 128)` and otherwise falls back to the SM80 class.
- Command:
  - `sed -n '1,220p' lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py`
  - `sed -n '100,190p' lib/levanter/src/levanter/grug/attention/_fa4_cute.py`
  - `sed -n '880,1040p' lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py`
  - `sed -n '1,120p' lib/levanter/src/levanter/grug/attention/_fa4_thd.py`
- Config:
  - H100 reports compute capability 9.0, so `_gpu_compute_arch()` should return `90`.
  - `flash4_cute_kernel_config(head_dim=128, arch=90)` currently returns `forward_tile=(128, 64)`, `backward_tile=(128, 64)`, `num_threads=128`.
  - `segmented_flash_attention_backward_launcher()` currently sets `arch = 120` only when `tile_m == 64 and tile_n == 64 and num_threads == 128`; otherwise it sets `arch = 80`.
- Result: Current SM90 `gpu_fa4_cute` config selects `SegmentedFlashAttentionBackwardSm80` and passes `arch=80` into FA4 backward postprocess. This supports the config/dispatch bug hypothesis.
- Interpretation: The profile's `SegmentedFlashAttentionForwardAmpere` name is not conclusive for forward, because the local segmented forward class is named `SegmentedFlashAttentionForwardAmpere`; however the backward launcher selection is a concrete SM80 fallback for SM90.
- Next action: Extract exact May208 FA4 rows, compare `gpu_fa4_thd` Hopper wiring, and patch `gpu_fa4_cute` so SM90 can select an explicit Hopper backward path when it compiles and passes tests.

### 2026-06-23 14:52 - May208 profile row dump
- Hypothesis: Side kernels may explain a large fraction of FA4 time, or the backward main-kernel row may show the unintended SM80 path.
- Command:
  - `/Users/dlwh/.codex/worktrees/ce72/marin/.venv/bin/python - <<'PY' ... ingest_profile('/Users/dlwh/.codex/worktrees/079b/marin/scratch/profiles/may208') ... PY`
  - Extraction used `marin.tools.roofline` from `/Users/dlwh/.codex/worktrees/24a6/marin/lib/marin/src` because the active `ce72` tree does not contain that module.
- Config:
  - Per-device per-step normalization is `total_us / (8 devices * 2 profiled steps)`.
  - XProf source is `/Users/dlwh/.codex/worktrees/079b/marin/scratch/profiles/may208_xprof_tables`.
- Result:
  - FA4 total: 829.727 ms track-summed, 4160 occurrences, 51.858 ms/device-step.
  - Backward main: 424.804 ms total, 416 occurrences, 1021.164 us average, 26.550 ms/device-step.
  - Forward main: 172.628 ms total, 416 occurrences, 10.789 ms/device-step.
  - Rematted forward main: 179.952 ms total, 416 occurrences, 11.247 ms/device-step.
  - Side kernels excluding forward/remat/main backward: 52.342 ms total, 3.271 ms/device-step.
  - The backward main kernel name in this profile is `SegmentedFlashAttentionBackwardSm120..._3`.
- Interpretation: Side kernels are material in aggregate but are not the dominant problem: they are 12.3% of main backward and 6.3% of all FA4 time. The May208 profile already labels the main backward kernel as SM120-style, so the profile is not explained by an SM80-named main kernel. The active local code still has an SM90-to-SM80 fallback unless the SM90 tile is changed to the 64x64/SM120-style path, so the code should make that routing explicit and then require an H100 reprofile to determine whether the main SM120-style backward is itself slow for this shape.
- Next action: Add explicit SM90 segmented backward config and launcher tests, then run CPU-safe Grug attention tests.

### 2026-06-23 15:00 - Explicit SM90 segmented backward path
- Hypothesis: Even if May208 already captured an SM120-style main backward kernel, `gpu_fa4_cute` should not infer architecture from tile shape. SM90 should explicitly select the 64x64 segmented main path and pass `arch=90` to upstream postprocess.
- Command:
  - `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py -q`
  - `uv run --directory lib/levanter --group test pytest tests/grug/test_attention.py -q`
  - `./infra/pre-commit.py --all-files`
- Config:
  - `Flash4CuteKernelConfig` now carries `backward_arch`.
  - `flash4_cute_kernel_config(head_dim=128, arch=90)` now returns `forward_tile=(128,64)`, `backward_tile=(64,64)`, `num_threads=128`, `backward_arch=90`.
  - `segmented_flash_attention_backward_launcher()` now receives `compute_arch` and separately resolves the segmented main-kernel family from the postprocess target architecture.
- Result:
  - `test_fa4_cute_attention.py`: 2 passed, 3 skipped.
  - `test_attention.py`: 13 passed.
  - `./infra/pre-commit.py --all-files`: OK, including Ruff, Black, pyrefly, and repository hygiene checks.
- Interpretation: CPU-safe tests now lock down the Hopper config and confirm SM90 uses the SM120-style segmented main kernel while upstream postprocess receives `arch=90`. This does not prove a performance win: the May208 profile already showed an SM120-style main backward kernel at 26.550 ms/device-step, so the next required evidence is an H100 compile/reprofile of this exact patch to check whether explicit SM90 postprocess and 64x64 config reduce time or whether the main segmented backward kernel remains the ceiling.
- Next action: On an H100 GPU environment with FA4/CuTe dependencies, reprofile May208 or run a minimal forward+backward microbench comparing the previous SM90 `(128,64)`/implicit path against this explicit SM90 64x64 path. Acceptance is a material reduction in the 26.550 ms/device-step main backward row without merely shifting time to side kernels.

### 2026-06-23 15:08 - H100 microbench setup
- Hypothesis: A standalone benchmark can iterate on the per-device May208 FA4 shape faster than full Grug training.
- Command:
  - Added `lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py`.
  - Allocated `dlwh-fa4-cute` H100 dev pod through `scripts/iris/dev_gpu.py`.
  - Ran `cd /app && uv sync --all-packages --extra=gpu` on the pod.
- Config:
  - May208 model has global batch 16, sequence length 4096, hidden dim 2560, 20 Q heads, 5 KV heads, head dim 128, and mesh `replica_dcn=2,data=1,expert=8,model=1`.
  - Since Grug MoE shards batch over `("replica_dcn", "data", "expert")`, the FA4 kernel's per-device batch is `B=1`, not global `B=16`.
  - Short-window layers use window 1024; long-window layers use window 2048.
- Result:
  - Initial pod had CPU-only JAX until syncing the GPU extra.
  - A tiny sanity run without constraining JAX to one GPU exited 137 and caused the holder pod to fail. No timing data was produced.
- Interpretation: Future benchmark commands must set `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false` from the start to avoid all-GPU preallocation/compile pressure in the dev holder. Benchmark defaults now target the per-device May208 shape.
- Next action: Reallocate a clean H100 pod, sync GPU deps, and run short/long window baselines with one visible GPU.

### 2026-06-23 15:12 - Direct Iris benchmark job and SM90 postprocess failure
- Hypothesis: The dev GPU holder's 1GB memory request is too small for JAX/CUTLASS compilation; a direct Iris job with a larger memory request should run the microbench.
- Command:
  - `uv run --package marin-iris --extra controller iris --config=lib/iris/config/cw-us-east-02a.yaml job run --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 40GB --extra gpu --job-name fa4-cute-sanity --timeout 1800 -- bash -lc 'cd /app && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py --label sanity --batch 1 --seq-len 64 --q-heads 4 --kv-heads 1 --head-dim 128 --sliding-window 64 --warmup 1 --iterations 2'`
- Config:
  - Direct job requested `H100x1`, 8 CPU, 64GB RAM, 40GB disk.
  - The benchmark used the explicit SM90 segmented path from the prior patch: SM120-style main kernel with postprocess `arch=90`.
- Result:
  - The job reached FA4/CuTe compilation but failed in upstream `flash_bwd_postprocess.py` with `ZeroDivisionError: integer division or modulo by zero`.
  - Failure site: `tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])`.
- Interpretation: Native `arch=90` postprocess is not plug-compatible with the SM120-style segmented atom layouts. This matches the May208 profile naming: postprocess rows include `o120...`, while the main kernel is `SegmentedFlashAttentionBackwardSm120`. For now, SM90 should remain explicit for config validation but use SM120-compatible postprocess until a real native Hopper segmented postprocess is ported.
- Next action: Change `_segmented_backward_arches(compute_arch=90, 64x64, 128)` to return `(120, 120)`, update the CPU-safe test, and rerun H100 sanity.

### 2026-06-23 15:18 - May208-shape H100 microbench
- Hypothesis: Explicit SM90 selection of the 64x64 SM120-compatible segmented path may materially improve the May208 per-device shape, but acceptance requires about 50% H100 SoL.
- Command:
  - Direct Iris H100 jobs with `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py ...`
  - Short-window target: `B=1,S=4096,Hq=20,Hkv=5,D=128,window=1024`.
  - Long-window target: `B=1,S=4096,Hq=20,Hkv=5,D=128,window=2048`.
- Config:
  - Current explicit Hopper config is forward tile `(128,64)`, backward tile `(64,64)`, 128 threads, `backward_arch=90`, internally resolved to SM120-compatible main and postprocess paths.
  - Legacy comparison uses backward tile `(128,64)` with inferred architecture.
- Result:
  - Short current: backward `0.7395 ms`, `127.1 TFLOP/s`, `12.85%` SoL; forward `0.3298 ms`, `114.0 TFLOP/s`, `11.52%` SoL.
  - Long current: backward `1.0893 ms`, `147.9 TFLOP/s`, `14.95%` SoL; forward `0.4760 ms`, `135.4 TFLOP/s`, `13.69%` SoL.
  - Short legacy `(128,64)` backward: `1.2933 ms`, `72.7 TFLOP/s`, `7.35%` SoL.
  - Short ratio-2 GQA (`Hq=20,Hkv=10`) current: backward `0.8011 ms`, `117.3 TFLOP/s`, `11.86%` SoL.
  - Ratio-1 MHA comparison initially failed because the wrapper routed FP32 GQA accumulator buffers into the non-GQA main-kernel/postprocess contract. The wrapper now has an explicit MHA branch, and `/dlwh/fa4-cute-mha-sanity` succeeded on `B=1,S=64,Hq=4,Hkv=4,D=128,window=64`.
- Interpretation:
  - The explicit 64x64 path is a real improvement over the legacy tile for this shape, about `1.75x` on short-window backward, but still far below the requested 50% SoL.
  - Ratio-2 being slower than ratio-4 on the same Q-head FLOP model argues against GQA atomics being the only dominant limiter. They may still matter for whole-backward side kernels, but the main segmented schedule/postprocess pipeline itself is not near the H100 ceiling.
  - A small tile/config tweak is unlikely to reach 50% SoL. The next credible route is a true SM90 segmented backward port using the upstream THD Hopper schedule (`64x128`, 384 threads, Hopper-specific `PdS_stage`, `SdP_swapAB`, and `AtomLayoutNdKV`) or a larger split-dKV/packed-GQA rewrite.
- Next action:
  - Keep the explicit 64x64 Hopper routing and benchmark harness.
  - Do not claim 50% SoL on this implementation. The current evidence says reaching that target requires a native Hopper segmented schedule, not just the compatibility path.

### 2026-06-23 15:30 - Native Hopper config-level candidate failed
- Hypothesis: The segmented fork might accept the upstream THD Hopper backward tile/thread shape `(64,128)` with 384 threads if forward and backward thread counts are separated, giving a quick native-Hopper path.
- Command:
  - Temporarily exposed a backward-only thread count in the benchmark and ran `/dlwh/fa4-cute-sm90-native-sanity` with `B=1,S=64,Hq=4,Hkv=1,D=128,window=64`, backward tile `64x128`, 384 backward threads.
- Result:
  - Compile failed in the segmented backward main kernel at `_fa4_cute_segmented_bwd.py:763`, `thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded))`.
  - CuTe reported it could not partition shape `(128,128)` with the constructed tiled MMA.
- Interpretation:
  - The THD Hopper schedule is not a drop-in config for the segmented fork. It needs a real SM90 segmented backward port, including the upstream Hopper MMA/layout choices, not just changing tile/thread values.
  - The temporary experimental allowance was removed from code; the checked-in path remains the verified 64x64/128 SM120-compatible Hopper route.

### 2026-06-23 16:14 - Upstream THD wrapper ceiling check failed
- Hypothesis: Existing `gpu_fa4_thd_attention` already wraps upstream FA4 SM90 and might provide a native-Hopper performance ceiling or a routeable backend for eligible dense/packed masks.
- Command:
  - Added `--backend thd` to `lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py`.
  - Ran `/dlwh/fa4-thd-full-may208-vjp` with `B=1,S=4096,Hq=20,Hkv=5,D=128`, full causal `window=4096`.
  - Ran `/dlwh/fa4-thd-sanity-vjp` with `B=1,S=64,Hq=4,Hkv=1,D=128`.
  - Inspected installed signatures in `/dlwh/fa4-signature-inspect`.
- Result:
  - Both THD benchmark jobs aborted inside CUTLASS/JAX with `std::bad_variant_access: Unexpected index`.
  - Installed `FlashAttentionBackwardSm90.__init__` signature matches the wrapper's `subtile_factor` spelling and other expected SM90 options, so this is not an obvious Python keyword mismatch.
- Interpretation:
  - The existing THD wrapper is not currently a usable performance crib on this image, even for tiny GQA.
  - Upstream `flash_bwd_sm90.py` remains the right source to port from, but we should not route production through `gpu_fa4_thd_attention` until that wrapper crash is fixed.
  - The most likely useful porting unit is the SM90 producer/consumer WGMMA structure and its block-sparse/local masking hooks, not the old segmented SM80-style mainloop.

### 2026-06-23 16:36 - Existing upstream SM90 kernels are not directly crib-able through JAX cutlass_call
- Hypothesis: The THD abort might be a wrapper-specific issue; dense BSHD upstream SM90 forward without cu-seqlens might still compile and provide a ceiling/reference path.
- Command:
  - Searched current upstream FA4 source and issue tracker for SM90/JAX/CuTe references.
  - Inspected installed H100 package signatures in `/dlwh/fa4-installed-source-inspect`.
  - Added `--pass forward` and `--pass direct-backward` to `lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py`.
  - Changed THD wrapper calls to pass `mCuSeqlensQ=`/`mCuSeqlensK=` by keyword.
  - Added `compile_options="--enable-tvm-ffi"` to THD `cutlass_call` sites after inspecting `cutlass/jax/compile.py` in `/dlwh/cutlass-jax-compile-options-inspect`.
  - Ran:
    - `/dlwh/fa4-thd-forward-isolate`
    - `/dlwh/fa4-thd-forward-keyword-cuseq`
    - `/dlwh/fa4-thd-forward-tvmffi`
    - `/dlwh/fa4-upstream-dense-forward-sanity`
    - `/dlwh/fa4-upstream-dense-forward-tvmffi`
- Config:
  - THD tiny: `B=1,S=64,Hq=4,Hkv=1,D=128`, full causal.
  - Dense upstream tiny: BSHD tensors, same shape, upstream SM90 forward tile `(128,128)`, backward target `(64,128)`, 384 threads.
  - Upstream interface reference: `flash_attn/cute/interface.py` selects SM90 head_dim 128 causal/local backward as `tile_m=64,tile_n=128,num_threads=384`, `SdP_swapAB=True`, `AtomLayoutMSdP=1`, `AtomLayoutNdKV=2`, `AtomLayoutMdQ=1`.
- Result:
  - THD raw forward fails before backward with `std::bad_variant_access: Unexpected index`.
  - Passing cu-seqlens as keywords does not change the THD failure.
  - `compile_options="--enable-tvm-ffi"` does not change the THD failure.
  - Dense BSHD upstream SM90 forward reaches CuTe compilation but fails in `flash_fwd_sm90.py` with `DSLRuntimeError: tile_scheduler is structured different after this while`.
  - The same dense BSHD failure persists with `--enable-tvm-ffi`.
  - Internet/source check found Dao-AILab/flash-attention#2163, which documents SM90 CuTe backward varlen plumbing rough edges and shows the same optional-argument-sensitive call surface.
- Interpretation:
  - Existing upstream FA4 SM90 kernels are useful as source to port from, but are not a direct JAX `cutlass_call` route in this image. The obstacle is the upstream SM90 scheduler/control-flow shape under the JAX bridge, not just Grug segment masking.
  - A block-sparse metadata wrapper around the installed upstream SM90 kernel is not enough unless the scheduler compile issue is fixed or the call is routed through Torch/TVM FFI outside JAX.
  - Current production-safe improvement remains the explicit 64x64 SM120-compatible segmented path (`~1.75x` over legacy short-window backward), but it is not close to 50% SoL.
- Next action:
  - Do not spend more time on the current THD wrapper as a shortcut.
  - Either port the SM90 producer/consumer WGMMA scheduler into the existing segmented kernel in a way the JAX bridge accepts, or create a separate non-JAX/FFI integration. The former is a real kernel port, not a tile tweak.

### 2026-06-23 16:54 - Native SM90 port boundary and Grug backward sparse metadata
- Hypothesis: Before porting the SM90 WGMMA mainloop, we should make the target scheduler knobs and Grug block-sparse metadata contract explicit in code so the native path is not conflated with the live SM120-compatible fallback.
- Command:
  - Added `Flash4CuteSm90BackwardConfig` and `sm90_flash4_cute_backward_config(...)` in `lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py`.
  - Added `_packed_segment_backward_block_sparse_indices(...)` in `lib/levanter/src/levanter/grug/attention/_fa4_cute.py`.
  - Ran `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran `./infra/pre-commit.py --all-files`.
- Config:
  - For May208/H100 `head_dim=128`, the native SM90 target is now recorded as `tile=(64,128)`, `num_threads=384`, `num_stages_q=2`, `num_stages_do=2`, `num_stages_pds=2`, `sdp_swap_ab=True`, `atom_layout_m_sdp=1`, `atom_layout_n_dkv=2`, `atom_layout_m_dq=1`.
  - The runtime `flash4_cute_kernel_config(arch=90)` still uses the verified compatibility path: forward `(128,64)`, backward `(64,64)`, 128 threads, SM120-compatible main/postprocess.
  - `_packed_segment_backward_block_sparse_indices` derives upstream backward Q-direction sparse metadata from Grug `[B,S]` lower bounds and valid bits. It returns broadcast-head tensors `(mask_block_cnt, mask_block_idx)` with shapes `[B,1,N]` and `[B,1,N,M]`, where each N block lists contributing M blocks.
- Result:
  - Focused tests: `20 passed, 3 skipped`.
  - Pre-commit: OK, including Ruff, Black, pyrefly, and repository hygiene checks.
- Interpretation:
  - The native SM90 target is now a first-class config artifact instead of an implicit note in the logbook.
  - Grug mask metadata can be translated into the same backward block-sparse orientation upstream FA4 expects. This does not yet feed the kernel, but it is the required metadata bridge for the planned SM90 scheduler port.
  - The remaining hard part is still the WGMMA producer/consumer mainloop and TMA/global-copy adaptation for JAX `cutlass_call`.
- Next action:
  - Add a native SM90 launcher boundary that consumes `Flash4CuteSm90BackwardConfig` and the sparse metadata, then port/compile the upstream SM90 mainloop incrementally behind that boundary.

### 2026-06-23 17:08 - Native SM90 launcher/backend boundary is explicit
- Hypothesis:
  - The native Hopper port should have a separate JAX/CuTe call boundary before the WGMMA mainloop is ported, so later work can feed upstream-style sparse scheduling metadata without touching the live compatibility path.
- Command:
  - Added `segmented_flash_attention_backward_sm90_native(...)` in `lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py`.
  - Added `segmented_flash_attention_backward_sm90_launcher(...)` in `lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py`.
  - Ran `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran `./infra/pre-commit.py --all-files`; an initial formatting-only failure was fixed, then the same command passed.
- Config:
  - Native SM90 boundary takes Grug `lower_bounds`/`valid` plus backward sparse metadata `(mask_block_cnt, mask_block_idx)`.
  - The launcher validates upstream's `head_dim=128`, `tile=(64,128)`, `num_threads=384`, `num_warp_groups=2` producer/consumer schedule independently of the SM80/SM120 fallback validator.
  - The live runtime path remains unchanged: H100 backward still routes through the verified `64x64/128-thread` SM120-compatible fallback.
- Result:
  - Focused tests: `23 passed, 3 skipped`.
  - Pre-commit: OK, including Ruff, Black, pyrefly, repository hygiene, and skill metadata checks.
- Interpretation:
  - We now have a compile-call contract for the native Hopper port: JAX input/output specs include the two sparse metadata tensors, GQA scratch output shapes stay dense only when needed, and the upstream Hopper schedule reaches the intentional `NotImplementedError` instead of being rejected by fallback-specific validation.
  - This is not a performance improvement yet; it removes the next integration ambiguity before copying the SM90 producer/consumer mainloop.
- Next action:
  - Port the upstream `flash_bwd_sm90.py` mainloop behind `segmented_flash_attention_backward_sm90_launcher`, preserving Grug fine masking and using `mask_block_cnt/mask_block_idx` for Q-block scheduling.

### 2026-06-23 17:20 - Native direct-backward probe path added to benchmark
- Hypothesis:
  - The next SM90 port iteration needs a stable H100 compile entry point that preserves the native config and feeds Grug sparse metadata directly into the native backward boundary.
- Command:
  - Updated `lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py` so `_config(...)` preserves `sm90_backward`.
  - Added `--backend cute-native --pass direct-backward`, which runs the segmented forward to produce `(out, lse)`, derives `_packed_segment_backward_block_sparse_indices(...)`, and calls `segmented_flash_attention_backward_sm90_native(...)`.
  - Ran `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran `./infra/pre-commit.py --all-files`.
- Config:
  - Expected H100 smoke command once the launcher is partially implemented:
    `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py --label sm90-native-direct --backend cute-native --pass direct-backward --batch 1 --seq-len 64 --q-heads 4 --kv-heads 1 --head-dim 128 --sliding-window 64 --warmup 1 --iterations 2`.
- Result:
  - Focused tests: `23 passed, 3 skipped`.
  - Pre-commit: OK.
- Interpretation:
  - The benchmark is now ready for the native compile loop and will fail at the intentional native `NotImplementedError` until the upstream SM90 mainloop is ported behind the launcher.
  - This does not change production behavior or performance; it removes another harness issue before H100 iteration.
- Next action:
  - Implement a first compile-only native SM90 launcher slice using upstream `FlashAttentionBackwardSm90` structure, then run the `cute-native` direct-backward smoke on H100.

### 2026-06-23 18:20 - Upstream SM90 backward adapter wired behind native boundary
- Hypothesis:
  - The fastest path to the requested native Hopper implementation is to reuse upstream `FlashAttentionBackwardSm90` and its block-sparse SM90 helper hooks, feeding Grug's segment semantics through `mask_mod` and feeding Grug's Q-block schedule through `BlockSparseTensors`.
- Command:
  - Replaced the native SM90 launcher stub in `lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py` with a compile-facing adapter around upstream `flash_attn.cute.flash_bwd_sm90.FlashAttentionBackwardSm90`.
  - Added a launcher-local `mask_mod` that reads `AuxData(tensors=(lower_bounds, valid))` and applies Grug's fine predicate: query valid, key at or after the packed lower bound, and key at or before the query.
  - The adapter constructs upstream `BlockSparseTensors(mask_block_cnt, mask_block_idx)` and passes it to the SM90 backward kernel.
  - Updated `lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py` to broadcast `[B,1,N,M]` sparse metadata to `[B,Hq,N,M]` before the native call because upstream block-sparse helpers index by Q head.
  - Ran `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran `./infra/pre-commit.py --all-files`; an initial Black-only failure was fixed, then the same command passed.
- Config:
  - Native SM90 target remains `tile=(64,128)`, `num_threads=384`, `num_stages_q=2`, `num_stages_do=2`, `num_stages_pds=2`, `sdp_swap_ab=True`, `atom_layout_m_sdp=1`, `atom_layout_n_dkv=2`, `atom_layout_m_dq=1`.
  - The upstream SM90 adapter sets `is_causal=False` and `is_local=False`; Grug lower-bound/valid metadata is the single fine mask source of truth.
  - All sparse blocks are currently treated as partial blocks, so upstream applies the mask function on every scheduled block. That is correct but leaves a possible future full-block optimization.
- Result:
  - Focused tests: `23 passed, 3 skipped`.
  - Pre-commit: OK, including Ruff, Black, pyrefly, repository hygiene, and skill metadata checks.
- Interpretation:
  - The native path is no longer a placeholder: it now uses upstream SM90 scheduling/building blocks and Grug block-sparse/fine-mask inputs at the intended boundary.
  - This still needs a real H100 compile smoke. Prior direct upstream SM90 forward failed under JAX with a tile scheduler structured-control-flow error, so the critical next question is whether backward's scheduler compiles through this JAX call boundary.
  - If it compiles, the next validation steps are small-shape gradient parity and May208-shape throughput against upstream FA4 and the current fallback.
- Next action:
  - Run the `cute-native` direct-backward smoke on H100 from this branch/worktree:
    `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py --label sm90-native-direct --backend cute-native --pass direct-backward --batch 1 --seq-len 64 --q-heads 4 --kv-heads 1 --head-dim 128 --sliding-window 64 --warmup 1 --iterations 2`.

### 2026-06-23 18:52 - H100 native SM90 compile loop exposes preprocess split requirement
- Hypothesis:
  - Reusing upstream `FlashAttentionBackwardSm90` behind the JAX `cutlass_call` boundary should either compile directly with Grug metadata or identify the exact wrapper mismatch that must be ported locally.
- Command:
  - Ran repeated H100 Iris smokes with:
    `uv run --package marin-iris --extra controller iris --config=lib/iris/config/cw-us-east-02a.yaml job run --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 40GB --extra gpu --job-name fa4-cute-sm90-native-smoke-<N> --timeout 1800 -- bash -lc 'cd /app && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py --label sm90-native-direct --backend cute-native --pass direct-backward --batch 1 --seq-len 64 --q-heads 4 --kv-heads 1 --head-dim 128 --sliding-window 64 --warmup 1 --iterations 2'`.
  - Inspected the installed H100 `flash_attn.cute.flash_bwd_sm90` source/signature with `/dlwh/fa4-installed-source-inspect`.
  - Ran local checks after each code change:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`
    and `./infra/pre-commit.py --all-files`.
- Config:
  - Native SM90 target remains `tile=(64,128)`, `num_threads=384`, `num_stages_q=2`, `num_stages_do=2`, `num_stages_pds=2`, `sdp_swap_ab=True`, `atom_layout_m_sdp=1`, `atom_layout_n_dkv=2`, `atom_layout_m_dq=1`.
  - The installed H100 FlashAttention package is older than upstream main: `FlashAttentionBackwardSm90.__call__` accepts `aux_tensors: Optional[list]`, and `flash_attn.cute.utils` does not expose `AuxData`.
  - Added a fixed-window native branch for the benchmark shape: `window_size_left=args.sliding_window - 1` routes to upstream `is_local=True`, `mask_mod=None`, `has_aux_tensors=False`, avoiding dynamic Grug aux tensors for the single-segment sliding-window case.
- Result:
  - Smokes 1-2 exposed harness/API mismatches: direct backward initially used the public forward wrapper incorrectly, then expected an upstream `AuxData` helper absent from the installed H100 package.
  - Smoke 3 established the installed call surface: `aux_tensors: Optional[list]`, no `AuxData`, `subtile_factor=1`.
  - Smokes 4-7 progressed through Grug `mask_mod` type issues. The correct custom-mask contract is: unwrap `batch_idx`, `q_idx`, and `kv_idx` with `utils.ssa_to_scalar(...)`, then return `utils.scalar_to_ssa(mask_value, cutlass.Boolean)`.
  - Smoke 8 showed the installed JAX TVM FFI cannot lower `aux_tensors=[lower_bounds, valid]`: `Unsupported argument type: <class 'cutlass.jax.types.JaxArrayList'> for annotated type: None`.
  - Smoke 9 confirmed tuple is not accepted either because upstream annotates the argument as `Optional[list]`.
  - Smoke 10 still saw the aux list because the fixed-window branch lived in the same JIT body as the custom branch.
  - Smoke 11, after splitting the fixed-window JIT body, eliminated the aux-list failure and reached MLIR verification. The new failure is:
    `cute.copy op bulk copy src expects gmem based memref, but got generic` at `quack/copy_utils.py:749`, on a `bulk_copy_g2s<f32>` source shaped like a per-row stats buffer.
  - Latest local checks: `23 passed, 3 skipped`; pre-commit OK.
- Interpretation:
  - The native SM90 adapter is now reaching the upstream Hopper mainloop through JAX/CuTe far enough to expose a real memory-space contract issue.
  - Upstream SM90 backward expects preprocess outputs such as `dPsum`/`LSE log2` to be gmem-backed inputs to the backward kernel. Our current wrapper fuses preprocess and backward into one `cutlass_call`, so those intermediate buffers are generic outputs when the SM90 mainloop tries to TMA/bulk-copy them back into smem.
  - The fixed-window branch is the right fast path for the benchmarked one-segment sliding-window shape, but it still needs the preprocess/backward split before it can produce timings.
  - The general packed Grug fine-mask path is separately blocked on the installed package's raw `aux_tensors: Optional[list]` JAX FFI lowering. Solving that likely needs either a local wrapper with `lower_bounds`/`valid` as first-class kernel args or upgrading to an upstream package version with `AuxData` support if its JAX FFI lowering handles named tuples.
- Next action:
  - Split native SM90 direct backward into at least two `cutlass_call`s: preprocess first to materialize `dpsum`, `lse_log2`, and any required `dq_accum` initialization in gmem; then run the SM90 backward launcher with those arrays as inputs.
  - After the fixed-window split compiles, restore postprocess or explicit JAX postprocess for `dq_accum`/`dk_accum`/`dv_accum`, validate small-shape parity, then run May208-shape throughput.

### 2026-06-23 18:58 - Accumulator-only output boundary still needs split calls
- Hypothesis:
  - Returning only the f32 accumulator buffers from the native SM90 call might avoid the TVM-FFI `JaxArrayList` conversion failure caused by unused direct `dq/dk/dv` outputs, while keeping upstream SM90 bulk copies on gmem-backed inputs.
- Command:
  - Changed the experimental native SM90 backend to return only `(dpsum, lse_log2, dq_accum, dk_accum, dv_accum)` from the `cutlass_call`, then postprocess those accumulators in JAX.
  - Updated the fixed-window launcher signature to remove direct `dq/dk/dv` outputs.
  - Ran focused local checks:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`
    and `./infra/pre-commit.py --all-files`.
  - Ran `/dlwh/fa4-cute-sm90-native-smoke-12` and `/dlwh/fa4-cute-sm90-native-smoke-13` on H100 with `--backend cute-native --pass direct-backward --batch 1 --seq-len 64 --q-heads 4 --kv-heads 1 --head-dim 128 --sliding-window 32`.
- Result:
  - Local focused tests: `23 passed, 3 skipped`.
  - Pre-commit: OK.
  - Smoke 12 failed because the fixed-window launcher still had the old direct-output signature; this was patched.
  - Smoke 13 still fails during TVM-FFI conversion with:
    `Unsupported argument type: <class 'cutlass.jax.types.JaxArrayList'> for annotated type: None`.
- Interpretation:
  - The `JaxArrayList` failure is not only from unused direct outputs. With `--enable-tvm-ffi`, the current fused outer launcher still hits a converter path that cannot handle this nested multi-output SM90 wrapper shape.
  - Without `--enable-tvm-ffi`, the same code reaches MLIR verification and fails because upstream SM90 bulk copies see generic memrefs for preprocess outputs. That confirms the real architectural fix is still a split-call pipeline, not another output-shape tweak.
- Next action:
  - Implement the fixed-window native path as separate JAX/CuTe calls: run upstream preprocess or an equivalent JAX preprocess first, then pass gmem-backed `dpsum`, `lse_log2`, and initialized accumulators into a backward-only SM90 `cutlass_call`.
  - Keep the production H100 fallback as the explicit `64x64/128-thread` SM120-compatible path until the split native path compiles and beats it on May208.

### 2026-06-23 19:32 - Split-call SM90 compiles but is not correct or fast
- Hypothesis:
  - Splitting upstream preprocess from the SM90 backward mainloop should turn `dpsum`/`lse_log2` into gmem-backed inputs and get past the SM90 stats TMA load. If TVM-FFI is required for gmem operands, patching the installed converter for CUTLASS JAX `JaxArrayList` may unblock the JAX bridge.
- Command:
  - Added a separate native preprocess launcher and changed `segmented_flash_attention_backward_sm90_native(...)` to run preprocess first, then a backward-only upstream `FlashAttentionBackwardSm90` call.
  - Inspected the installed H100 package:
    - `FlashAttentionBackwardSm90.__call__` uses `aux_tensors: Optional[list]` and `blocksparse_tensors: Optional[BlockSparseTensors]`.
    - `cutlass.jax.compile.jit_wrapper` accepts one `JaxArrayList` wrapper argument, and the installed TVM-FFI converter has no `JaxArrayList` branch.
  - Added a guarded converter patch for the experimental native path so `JaxArrayList` lowers as a tuple of data pointers under `--enable-tvm-ffi`.
  - Ran focused local checks after edits:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran H100 jobs:
    - `/dlwh/fa4-cute-sm90-native-smoke-16`
    - `/dlwh/fa4-cute-sm90-native-diagnose-arg`
    - `/dlwh/fa4-cute-sm90-native-smoke-1925-jalptr`
    - `/dlwh/fa4-cute-sm90-native-correct-1927`
    - `/dlwh/fa4-cute-sm90-native-correct-1932-post120`
- Result:
  - The split preprocess call compiles without `--enable-tvm-ffi`.
  - The backward-only SM90 call still fails without TVM-FFI at the upstream stats load:
    `cute.copy op bulk copy src expects gmem based memref, but got generic`.
  - The `JaxArrayList` TVM-FFI patch gets the backward-only SM90 call to compile and run on the tiny fixed-window smoke.
  - Tiny fixed-window timing is unusable: `/dlwh/fa4-cute-sm90-native-smoke-1925-jalptr` reported `1873.1 ms` for `B=1,S=64,Hq=4,Hkv=1,D=128,window=32`.
  - Correctness is also not acceptable. Against the verified compatibility backend on the same forward result:
    - Simple JAX reshape postprocess: `dq mean_abs=0.298`, `dk mean_abs=0.585`, `dv mean_abs=0.702`.
    - Upstream SM120-compatible postprocess: `dq mean_abs=0.295`, `dk mean_abs=0.489`, `dv mean_abs=0.580`; `dk/dv` norms also diverged.
  - Upstream postprocess with `arch=90` still fails with `ZeroDivisionError` in `flash_bwd_postprocess.py`, so the installed package only has a usable SM120-compatible postprocess for this path.
- Interpretation:
  - The native upstream SM90 mainloop can now be reached through JAX, but the current bridge is not a viable kernel path: it is far below the production fallback and does not produce correct gradients.
  - The bad parity is consistent with accumulator layout mismatch. The mainloop outputs are not in the simple contiguous layout, and the installed upstream postprocess does not match the accumulator layout produced by this adapted call.
  - The `JaxArrayList` converter patch is useful as a diagnostic bridge but not sufficient for production. It may also explain the terrible timing if it is forcing an inefficient wrapper ABI.
- Next action:
  - Do not route Grug through the current native SM90 adapter.
  - Either port the upstream SM90 mainloop and postprocess layout together into the local segmented kernel contract, or keep the verified `64x64/128-thread` compatibility path and pursue smaller improvements there.
  - Before any May208 benchmark of the native path, require a tiny H100 parity test to pass against the compatibility backend.

### 2026-06-23 19:40 - Native SM90 failure narrowed to accumulator layout/postprocess
- Hypothesis:
  - The split-call native SM90 path might be computing bad f32 accumulators, or it might be computing correct accumulators but feeding them to the wrong postprocess/layout contract.
- Command:
  - Ran `/dlwh/fa4-cute-sm90-accum-diagnose` on H100 with `B=1,S=64,Hq=4,Hkv=1,D=128,window=32`.
  - The script compared the verified compatibility backward scratch outputs against the split-call upstream SM90 adapter before postprocess.
  - Ran `/dlwh/fa4-cute-sm90-jax-post-diagnose` to test simple JAX reshape/truncate postprocess candidates with and without `softmax_scale`.
- Result:
  - Native SM90 and compatibility f32 accumulator norms/sums match on the active data:
    - dQ accum norm `558.554`, sum `-568.897`, shape `[1,4,8192]` in both paths.
    - dK accum norm `563.436`, sum about `-0.655`; native shape is `[1,1,16384]` because tile-N pads to 128 rows, compatibility shape is `[1,1,8192]`.
    - dV accum norm `66.593`, sum `-574.839`; native shape is `[1,1,16384]` for the same padding reason.
  - Upstream SM120-compatible postprocess still gives wrong gradients from the native accumulators:
    - dQ mean abs `0.292`, max abs `2.590`.
    - dK mean abs `0.480`, max abs `3.781`.
    - dV mean abs `0.592`, max abs `8.588`.
  - Simple JAX reshape/truncate is also wrong. The best scale choices match output norms but still have large elementwise errors:
    - dQ with scale: mean abs `0.292`, max abs `2.652`.
    - dK with scale: mean abs `0.578`, max abs `4.172`.
    - dV without scale: mean abs `0.720`, max abs `9.969`.
- Interpretation:
  - The upstream SM90 mainloop is likely doing the right math for this fixed-window case, but its accumulator store order does not match either the local compatibility path's postprocess contract or a naive contiguous `[B,H,S,D]` mapping.
  - Correctness now requires porting the matching SM90 postprocess/layout contract or teaching the local wrapper the exact per-tile accumulator permutation. This is separate from the earlier JAX/TVM-FFI bridge issue.
  - Even if the layout is fixed, `/dlwh/fa4-cute-sm90-native-smoke-1925-jalptr` measured the current JAX/TVM-FFI adapter at `1873 ms` on a tiny shape, so this bridge is still not a credible route to 50% SoL.
- Next action:
  - Keep production on the verified explicit Hopper compatibility path: `64x64`, 128 threads, SM120-compatible main/postprocess.
  - Treat the native SM90 adapter as a diagnostic boundary only until both a tiny parity test and a tiny timing sanity test pass.
  - Reaching the requested ceiling needs either a local port of upstream SM90 mainloop plus matching accumulator/postprocess ABI, or a different integration that avoids the current JAX/TVM-FFI wrapper overhead.

### 2026-06-23 19:54 - Installed SM90 postprocess is not a drop-in fix
- Hypothesis:
  - The previous native SM90 parity failure might be caused by forcing the SM120-compatible postprocess. If the installed upstream arch-90 postprocess is invoked with the SM90 accumulator layout and thread count, it may decode the native accumulator store order correctly.
- Command:
  - Inspected installed H100 source snippets with `/dlwh/fa4-sm90-source-snips` and `/dlwh/fa4-sm90-target-snips`.
  - Ran `/dlwh/fa4-sm90-post384-diagnose2`, `/dlwh/fa4-sm90-post384-custommask`, `/dlwh/fa4-sm90-post384-filelauncher`, and `/dlwh/fa4-sm90-post384-filelauncher2` to test arch/thread variants through the checked-in postprocess launcher.
  - Ran `/dlwh/fa4-post-mma-sizes` to inspect valid postprocess tiled-MMA construction for `(arch, tile_m, AtomLayoutMdQ)` combinations.
- Result:
  - Source inspection confirmed a real contract difference:
    - SM90 backward stores dQ through `dQaccum_store_block_sparse_bwd_sm90(...)` and uses `sdQaccum_layout = cute.make_layout((tile_m * tile_hdim // num_wg_dQ, num_wg_dQ))`.
    - The installed postprocess has a separate arch-90 path with `s2r_tiled_copy_dQaccum` and `sdQaccum_layout = cute.make_layout((tile_m * tile_hdim // num_wg_mma, num_wg_mma))`.
  - The upstream built-in local-mask branch is not reliable in this wrapper: `/dlwh/fa4-sm90-post384-diagnose2` hit `AssertionError: mask_causal and mask_local cannot be both True` for `window_size_left=31, window_size_right=0`.
  - The custom Grug fine-mask branch avoids that local-mask assertion and reaches the native mainloop.
  - The checked-in launcher can now vary postprocess `arch` and `num_threads`, but installed arch-90 postprocess still does not provide a working drop-in:
    - `arch=90,tile_m=128,AtomLayoutMdQ=2` and similar dK/dV variants raise `ZeroDivisionError`.
    - `arch=90,tile_m=64,AtomLayoutMdQ=1` cannot be validated outside lowering and later hits postprocess construction/assertion issues through `cutlass_call`.
    - `arch=120,num_threads=128` remains the compileable fallback, but earlier parity runs show it decodes native accumulators incorrectly.
- Interpretation:
  - The native SM90 mainloop integration is blocked by a coupled ABI problem: the installed arch-90 postprocess is not directly usable for the dQ/dK/dV accumulator shapes we expose through JAX, and the SM120 postprocess compiles but consumes the wrong accumulator order.
  - This means the “quick” path of reusing upstream SM90 mainloop plus upstream postprocess is not enough. The credible path is a local SM90 postprocess/layout adapter or porting mainloop and postprocess together under one controlled accumulator contract.
  - The benchmark probe now forces `window_size_left=None` for `cute-native`, so future native probes use Grug `lower_bounds`/`valid` as the single mask source instead of upstream's failing built-in local branch.
- Next action:
  - Keep native SM90 behind the explicit experimental boundary.
  - Do not route production or performance claims through it until a file-backed tiny parity test passes.
  - If continuing the native port, implement a local SM90 accumulator decode/postprocess kernel by cribbing the arch-90 `FlashAttentionBackwardPostprocess` shared-memory layout instead of calling it as-is.

### 2026-06-23 19:39 - Fresh May208 SM90 `dQ_single_wg` probe
- Hypothesis:
  - For head dim 128, forcing upstream SM90 to use a single dQ warp group might avoid or reduce the problematic internal dQ accumulator store/postprocess path and improve the May208-shape backward throughput.
- Command:
  - Ran:
    `uv run --package marin-iris --extra controller iris --config=lib/iris/config/cw-us-east-02a.yaml job run --enable-extra-resources --gpu H100x1 --cpu 8 --memory 64GB --disk 80GB --extra gpu --job-name fa4-cute-sm90-native-may208-dqsingle-fresh --timeout 2400 -- bash -lc 'cd /app && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py --label may208-dqsingle-fresh --backend cute-native --pass direct-backward --batch 1 --seq-len 4096 --q-heads 20 --kv-heads 5 --head-dim 128 --sliding-window 2048 --sm90-dq-single-wg true --warmup 3 --iterations 10'`.
  - Ran source probes against the installed H100 `flash_attn.cute.flash_bwd_sm90.FlashAttentionBackwardSm90` to inspect `__call__`, `kernel`, `dQaccum_store`, and `num_wg_dQ` behavior.
- Config:
  - Shape: `B=1`, `S=4096`, `Hq=20`, `Hkv=5`, `D=128`, sliding window `2048`, BF16.
  - Native SM90 config: tile `(64,128)`, `num_threads=384`, Q/dO/PdS stages `2/2/2`, `SdP_swapAB=True`, `dKV_swapAB=False`, `dQ_swapAB=False`, atom layouts `1/2/1`, `dq_single_wg=True`.
- Result:
  - Job `/dlwh/fa4-cute-sm90-native-may208-dqsingle-fresh` succeeded.
  - Backward time: `0.493367 ms`.
  - Backward throughput: `326.506 TFLOP/s`.
  - Backward SoL: `0.33014`.
  - This matches the prior native SM90 May208 run within noise (`0.494968 ms`, `325.45 TFLOP/s`, `0.32907 SoL`).
- Interpretation:
  - `dq_single_wg` is not the missing performance knob for the target shape.
  - The best confirmed native-Hopper direct-backward number remains about `33% SoL`, below the requested `50% SoL`.
  - The next useful branch is source-driven: determine whether upstream's internal accumulator layout/postprocess can be reused correctly, or whether we should stop trying to wrap `FlashAttentionBackwardSm90` directly and instead tune the verified compatibility path.
- Next action:
  - Finish the focused upstream source-range probe and inspect the internal dQ store / dK-dV output layout.
  - If there is no small wrapper-side layout fix, run a one-axis tuning sweep over the verified SM90/native and compatibility configurations instead of further speculative wrapper patches.

### 2026-06-23 19:46 - Native SM90 parity still fails; local block-sparse was not the cause
- Hypothesis:
  - The fixed sliding-window branch was passing both upstream `is_local=True` and Grug block-sparse metadata. Removing the redundant sparse metadata might fix the native backward tile set and gradient mismatch.
- Command:
  - Patched the fixed-window `FlashAttentionBackwardSm90` call to omit `blocksparse_tensors`.
  - Ran focused local tests:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran small H100 parity job `/dlwh/fa4-cute-sm90-native-small-parity-nosparse-local` comparing native SM90 direct backward against the verified compatibility backward for `B=1,S=64,Hq=4,Hkv=1,D=128,window=32`.
  - Ran May208 timing job `/dlwh/fa4-cute-sm90-native-may208-nosparse-local` for `B=1,S=4096,Hq=20,Hkv=5,D=128,window=2048`.
- Result:
  - Local tests: `23 passed, 3 skipped`.
  - Small parity was unchanged and bad:
    - `dq`: mean abs `0.2952`, relative norm `1.405`.
    - `dk`: mean abs `0.4890`, relative norm `1.222`.
    - `dv`: mean abs `0.5797`, relative norm `1.213`.
  - May208 timing without local block-sparse metadata was slightly slower:
    `0.501386 ms`, `321.285 TFLOP/s`, `0.32486 SoL`.
  - Reverted the no-sparse local probe so the branch keeps the faster sparse-scheduled behavior.
- Interpretation:
  - The native SM90 path is not correct yet, so the `~33% SoL` number is not a shippable performance claim.
  - Local block-sparse scheduling is not causing the parity failure.
  - The mismatch is more likely one of: wrong mask semantics (`is_local`/causal interpretation), wrong accumulator/postprocess layout, or an input/output tensor mode mismatch at the JAX/CuTe boundary.
- Next action:
  - Test mask semantics explicitly by comparing native SM90 local-window settings against the reference on the same small shape.
  - If mask semantics are not the cause, inspect postprocess mode/layout and stop treating the native wrapper as usable until the accumulator contract is matched.

### 2026-06-23 19:59 - Existing-kernel and correctness baseline summary
- Hypothesis:
  - The remaining native SM90 mismatch might be a simple mask flag, output permutation, or an overly strict comparison against the compatibility backend rather than the true math.
- Command:
  - Tried a temporary `is_causal=True, is_local=True` native SM90 construction for the fixed local-window branch and ran `/dlwh/fa4-cute-sm90-native-small-parity-causal-local`.
  - Tried comparing segmented forward `out/lse` with upstream dense/local forward in `/dlwh/fa4-cute-forward-lse-contract-check` and `/dlwh/fa4-cute-forward-lse-contract-check-patched`.
  - Measured existing THD direct backward on May208 shape with `/dlwh/fa4-thd-may208-direct-bwd-baseline`.
  - Measured the correct compatibility custom-VJP path on May208 shape with `/dlwh/fa4-cute-may208-full-compat-baseline`.
  - Ran native-vs-compatibility `dQ` correlation/permutation diagnostic in `/dlwh/fa4-cute-sm90-native-dq-correlation`.
  - Ran both compatibility and native against a materialized JAX local-attention reference in `/dlwh/fa4-cute-native-vs-jax-reference`.
- Result:
  - `is_causal=True` plus `is_local=True` is invalid upstream: `mask_causal and mask_local cannot be both True`. The temporary change was reverted.
  - Upstream dense/local forward is not currently usable as an oracle through the installed JAX surface:
    - without the converter patch it fails on `JaxArrayList`;
    - with the converter patch it ICEs in `flash_attn/cute/pack_gqa.py`.
  - THD direct backward crashed in the installed stack with `std::bad_variant_access: Unexpected index`.
  - Compatibility May208 full-path baseline:
    - forward `0.486690 ms`, `132.394 TFLOP/s`, `13.39% SoL`;
    - backward `1.100330 ms`, `146.399 TFLOP/s`, `14.80% SoL`;
    - backward-with-forward `1.507043 ms`.
  - Native May208 direct backward remains faster but wrong:
    - best measured valid compile/timing: `0.493367 ms`, `326.506 TFLOP/s`, `33.01% SoL`;
    - target `50% SoL` would require about `0.326 ms` for this flop accounting.
  - The native `dQ` mismatch is not a simple layout permutation:
    - head-wise cosine values are near zero for all head pairs;
    - sequence reversal, head reversal, and a simple head/sequence flatten transpose do not improve relative norm (`~1.40`).
  - Materialized JAX reference confirms the compatibility backend is the correct baseline:
    - compatibility vs JAX relative norms: `dq=0.00419`, `dk=0.00490`, `dv=0.00441`;
    - native vs JAX relative norms: `dq=1.405`, `dk=1.222`, `dv=1.213`.
- Interpretation:
  - Existing kernels do not currently provide an acceptable correct path for this exact shape:
    - compatibility is correct but far below `50% SoL`;
    - THD crashes;
    - native SM90 is faster but incorrect.
  - The native wrapper should remain quarantined as an experimental path. The failure is not a simple postprocess axis permutation and not a simple causal/local flag.
  - The likely cause is a deeper contract mismatch between Grug's forward/LSE/block-sparse metadata and upstream SM90 backward's expected schedule/mask state, or a misuse of the upstream GQA/native backward ABI under JAX.
- Next action:
  - Do not ship the native SM90 path into production selection until a tiny JAX-reference parity test passes.
  - If continuing, port the upstream SM90 backward as a local kernel with first-class Grug mask/lower-bound arguments instead of wrapping the installed `FlashAttentionBackwardSm90` object, or add a dedicated tiny correctness harness that can bisect native mainloop inputs (`dPsum`, `LSE log2`, and block scheduling) against the compatibility implementation.

### 2026-06-23 19:56 - Continuation after app crash: postprocess route remains blocked
- Hypothesis:
  - The native SM90 May208 direct-backward timing around `33% SoL` might become usable if the wrong-gradient issue is only caused by invoking the SM120-compatible postprocess instead of the installed arch-90 postprocess.
- Command:
  - Added `arch` and `num_threads` knobs to `flash_attention_backward_postprocess_launcher(...)` while preserving the existing defaults `arch=120,num_threads=128`.
  - Ran `/dlwh/fa4-sm90-post384-filelauncher`, `/dlwh/fa4-sm90-post384-filelauncher2`, and `/dlwh/fa4-post-mma-sizes`.
  - Updated the native benchmark probe to pass `window_size_left=None`, forcing Grug `lower_bounds`/`valid` to be the single fine-mask source instead of upstream's failing built-in local branch.
  - Ran:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`
    and `./infra/pre-commit.py --all-files`.
- Result:
  - The file-backed postprocess launcher compiles with the existing arch-120 default, but arch-120 is already known to decode the native SM90 accumulators incorrectly.
  - The installed arch-90 postprocess does not provide a wrapper-side fix:
    - dK/dV-style `tile_m=128,AtomLayoutMdQ=2` construction raises `ZeroDivisionError`.
    - dQ-style `tile_m=64,AtomLayoutMdQ=1` hits postprocess construction/assertion issues through `cutlass_call`.
  - Focused tests: `23 passed, 3 skipped`.
  - Pre-commit: OK.
- Interpretation:
  - The native SM90 direct-backward timing is interesting but remains non-shippable because parity is bad.
  - The next implementation step is not another flag flip. It needs a local SM90 accumulator decode/postprocess kernel or a full local mainloop+postprocess port with one controlled accumulator ABI.
  - The production-safe path remains the explicit SM90 compatibility path; it is correct and faster than legacy, but only about `13-15% SoL` on May208-shape microbench.
- Next action:
  - If continuing toward `50% SoL`, do a bounded tuning sweep of the correct compatibility path while separately prototyping a local SM90 postprocess adapter. Do not use the native path for training until the tiny H100 parity gate passes.

### 2026-06-23 20:22 - Native SM90 postprocess/gmem fix passes parity and is routed for May208
- Hypothesis:
  - The native SM90 mainloop was producing usable accumulators, but the wrapper was feeding them through the wrong split-call postprocess/memory-space contract. Reusing the arch-90 postprocess with the SM90-compatible tile/atom layout and forcing split-call accumulators back to gmem before postprocess might make the native path correct.
- Command:
  - Added optional `cluster_size`, `use_2cta_instrs`, and `accum_is_gmem` knobs to `flash_attention_backward_postprocess_launcher(...)`.
  - Changed `segmented_flash_attention_backward_sm90_native(...)` to run arch-90 postprocess with `tile_m=64`, `AtomLayoutMdQ=1`, `cluster_size=1`, and `accum_is_gmem=True`.
  - Moved `_packed_segment_backward_block_sparse_indices(...)` into `_fa4_cute_backend.py` and routed production `segmented_flash_attention_backward(...)` to the native SM90 path for GQA, `head_dim=128` only. MHA and other head dims continue to use the compatibility fallback.
  - Ran focused CPU-safe tests:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
  - Ran H100 jobs:
    - `/dlwh/fa4-cute-sm90-post90-gmem-smoke`
    - `/dlwh/fa4-cute-sm90-post90-parity`
    - `/dlwh/fa4-cute-sm90-post90-may208`
    - `/dlwh/fa4-cute-sm90-native-full-may208`
    - `/dlwh/fa4-cute-sm90-production-parity`
    - `/dlwh/fa4-cute-sm90-may208-native-vs-compat`
- Result:
  - The initial arch-90 postprocess attempt failed MLIR verification because `flash_bwd_postprocess.py` expected a gmem accumulator source and saw a generic memref. Casting the split-call accumulator input to a gmem tensor fixed the verifier failure.
  - Tiny direct native backward parity against JAX now matches the compatibility backend:
    - compatibility vs JAX: `dq rel=0.00478`, `dk rel=0.00528`, `dv rel=0.00456`.
    - native vs JAX: `dq rel=0.00478`, `dk rel=0.00528`, `dv rel=0.00456`.
  - Production `gpu_fa4_cute_attention` custom-VJP parity for D128 GQA also passes:
    - `dq rel=0.00459`, `dk rel=0.00516`, `dv rel=0.00449`.
  - Full May208-shape native-vs-compatibility backward comparison also matches tightly:
    - `dq rel=2.13e-05`, `dk rel=1.00e-05`, `dv rel=8.24e-06`.
  - May208-shape direct native backward, `B=1,S=4096,Hq=20,Hkv=5,D=128,window=2048`:
    - `0.640741 ms`, `251.408 TFLOP/s`, `25.42% SoL`.
  - May208-shape production full benchmark through `--backend cute --pass full`:
    - forward `0.481672 ms`, `133.774 TFLOP/s`, `13.53% SoL`.
    - backward `0.636287 ms`, `253.168 TFLOP/s`, `25.60% SoL`.
    - backward-with-forward `1.043764 ms`.
  - Focused tests: `24 passed, 3 skipped`.
- Interpretation:
  - The fast native SM90 path is now correct for the May208-relevant D128 GQA route and is wired into production selection for that route.
  - The corrected native path is slower than the earlier wrong-gradient direct number (`~0.493 ms`) but still materially faster than the correct compatibility baseline (`~1.10 ms` backward, `~1.51 ms` backward-with-forward).
  - This does not reach the original `50% SoL` target; it raises the correct backward path from about `14.8% SoL` to about `25.6% SoL`. Further improvement likely needs either reducing split-call/postprocess overhead or a local fused SM90 mainloop+postprocess ABI.
- Next action:
  - Run full pre-commit.
  - For another performance pass, profile the corrected native full path to quantify preprocess/postprocess and split-call overhead before attempting more kernel work.

### 2026-06-23 20:50 - Full-block metadata improves correct native path, but tuning does not reach 50% SoL
- Hypothesis:
  - The corrected native SM90 path still applies the Grug fine mask to every scheduled Q/K tile. Splitting backward block-sparse metadata into partial and full blocks should let upstream skip mask work on fully covered blocks and recover some May208 throughput.
- Command:
  - Added `_packed_segment_backward_block_sparse_indices_with_full(...)` and routed the production D128 GQA SM90 path through partial and full block metadata.
  - Updated the native benchmark path to pass both sparse lists into `segmented_flash_attention_backward_sm90_native(...)`.
  - Ran component timing, stage sweeps, full-block parity/perf, built-in-local upper-bound, and tile sweeps on H100:
    - `/dlwh/fa4-cute-sm90-component-timing`
    - `/dlwh/fa4-cute-sm90-stage-sweep`
    - `/dlwh/fa4-cute-sm90-fullblocks-may208`
    - `/dlwh/fa4-cute-sm90-fullblocks-parity`
    - `/dlwh/fa4-cute-sm90-builtin-local-upper`
    - `/dlwh/fa4-cute-sm90-tile-sweep`
  - Ran focused CPU-safe tests:
    `uv run --directory lib/levanter --group test pytest tests/grug/test_fa4_cute_attention.py tests/grug/test_attention.py -q`.
- Result:
  - Full-block metadata keeps May208-shape native-vs-compatibility parity tight:
    - `dq rel=2.10e-05`, `dk rel=2.97e-05`, `dv rel=2.01e-05`.
  - May208 production full benchmark with full-block metadata:
    - forward `0.483080 ms`, `133.384 TFLOP/s`, `13.47% SoL`.
    - backward `0.524739 ms`, `306.986 TFLOP/s`, `31.04% SoL`.
    - backward-with-forward `0.930608 ms`.
  - Component timing says the native mainloop is still the largest piece:
    - preprocess `0.117054 ms`, mainloop `0.567776 ms`, dQ post `0.098026 ms`, dK post `0.060067 ms`, dV post `0.058411 ms`.
    - The standalone component sum overestimates the end-to-end path, but the mainloop alone is above the approximate `0.326 ms` backward time needed for `50% SoL`.
  - Stage sweep did not find a better setting:
    - default `0.640086 ms`, `25.45% SoL`.
    - `PdS_stage=1`: `0.660997 ms`.
    - `dO_stage=1`: `0.646084 ms`.
    - all stages 1: `0.774596 ms`.
    - `dQ_single_wg`: `0.653109 ms`.
    - `Q_stage=1` and `PdS_stage=3` were invalid under upstream assertions.
  - Built-in local-mask single-segment upper bound is correct but still below target:
    - built-in local `0.489715 ms`, `33.26% SoL`.
    - custom full-block path in the same run `0.523483 ms`, `31.11% SoL`.
  - Tile sweep did not improve the routed full-block path:
    - `64x128`: `0.547632 ms`, `29.74% SoL`.
    - `128x128`: invalid, `330752` bytes shared memory exceeds `232448` byte SM90a limit.
    - `128x64`: invalid, `265216` bytes shared memory exceeds `232448` byte SM90a limit.
    - `64x64`: `0.731031 ms`, `22.28% SoL`.
  - Focused tests after the full-block metadata changes: `25 passed, 3 skipped`.
- Interpretation:
  - The fast path is now correct and materially faster than the compatibility baseline: May208 backward moved from about `1.10 ms` / `14.8% SoL` to about `0.525 ms` / `31.0% SoL`.
  - It still misses the original `50% SoL` target by a large margin. The best existing-kernel upper bound found in this wrapper is about `33% SoL`, so another small config knob is unlikely to close the gap.
  - The next credible route to `50% SoL` is deeper kernel work: reduce split-call/postprocess overhead, fuse or locally control the SM90 mainloop+postprocess ABI, or write a more specialized Grug local-window/GQA backward schedule instead of wrapping the installed upstream kernel as-is.
- Next action:
  - Run full pre-commit after these final edits.
  - Keep the current production route limited to the verified D128 GQA SM90 case; leave MHA and other head dims on the compatibility fallback.

### 2026-06-23 20:58 - Goal audit against upstream-local path
- Hypothesis:
  - The remaining completion route is the "90% of upstream perf in our context" clause. For the May208 per-device context, the closest upstream comparator that compiles in this JAX wrapper is the same native SM90 backward kernel using upstream's built-in local-window mask instead of Grug's lower-bound mask.
- Command:
  - Added `--native-mask-mode {grug,builtin-local}` to `lib/levanter/scripts/bench/bench_grug_fa4_cute_attention.py` for direct native-backward comparison.
  - Ran `/dlwh/fa4-cute-sm90-goal-audit`:
    - production full benchmark, May208 D128 GQA, window 2048.
    - direct native backward with Grug mask.
    - direct native backward with built-in local mask.
    - a tiny D128 JAX-reference absolute-error audit.
  - Ran `/dlwh/fa4-cute-sm90-upstream-atol`:
    - direct Grug-mask native backward vs built-in-local native backward on tiny D128 and May208 D128.
- Result:
  - May208 production full benchmark:
    - forward `0.478742 ms`, `134.592 TFLOP/s`, `13.61% SoL`.
    - backward `0.522865 ms`, `308.086 TFLOP/s`, `31.15% SoL`.
    - backward-with-forward `0.930058 ms`.
  - May208 direct native backward:
    - Grug mask: `0.545956 ms`, `295.056 TFLOP/s`, `29.83% SoL`.
    - built-in local mask: `0.504048 ms`, `319.587 TFLOP/s`, `32.31% SoL`.
    - Direct Grug-mask perf is `92.3%` of the built-in-local upstream path. Production VJP backward timing is `96.4%` of the same direct built-in-local timing.
  - Direct Grug-mask backward vs built-in-local backward absolute differences:
    - tiny D128: `dq/dk/dv max_abs=0.0`.
    - May208 D128: `dq max_abs=0.000488`, `dk max_abs=0.000061`, `dv max_abs=0.000977`; all are below `0.01`.
  - Tiny D128 comparison against materialized JAX float32 reference has good relative norms but does not satisfy strict max-abs `0.01` for every tensor:
    - output `max_abs=0.015625`, rel `0.00343`.
    - `dq max_abs=0.0078125`, rel `0.00478`.
    - `dk max_abs=0.03125`, rel `0.00528`.
    - `dv max_abs=0.0625`, rel `0.00456`.
- Interpretation:
  - Against the available upstream-local FA4/CuTe path in this exact single-segment May208 context, the routed Grug D128 GQA native path satisfies both goal clauses: max absolute error below `0.01` and at least `90%` of upstream-local performance.
  - Against a materialized JAX float32 reference, strict max-abs `0.01` is too tight for BF16 output/gradient tensors with standard-normal inputs; the evidence there supports BF16-relative correctness rather than a strict absolute bound.
  - The routed production path still does not satisfy the alternate `50% SoL` clause; it is about `31% SoL`.
- Next action:
  - Run focused tests and full pre-commit after the benchmark-harness/logbook edits.

### 2026-06-23 21:01 - JAX strict-atol miss is shared with upstream FA4
- Hypothesis:
  - The strict `0.01` max-absolute misses against materialized JAX may be a BF16 reference-casting artifact rather than a Grug kernel error.
- Command:
  - Ran `/dlwh/fa4-cute-sm90-jax-atol-diagnose` on tiny D128 GQA, comparing the production FA4 path against:
    - materialized JAX float32-logit reference.
    - the same JAX output rounded to BF16.
    - JAX gradients through both the unrounded and BF16-rounded reference output.
- Result:
  - Output vs JAX float32 reference and output vs JAX BF16-rounded reference are identical:
    - `max_abs=0.015625`, `relative_norm=0.00343`.
  - Gradient errors also do not come from rounding the reference output:
    - `dq max_abs=0.0078125`, rel `0.00478`.
    - `dk max_abs=0.03125`, rel `0.00528`.
    - `dv max_abs=0.0625`, rel `0.00456`.
    - JAX unrounded-reference gradients vs JAX BF16-rounded-output gradients are exactly equal for this case.
- Interpretation:
  - The strict `0.01` max-abs mismatch against materialized JAX is not a wrapper-specific regression and not a reference-cast artifact. It is the same FA4 numerical contract shared with the upstream-local SM90 path.
  - The actionable correctness target for the performance goal is therefore upstream-local FA4/CuTe parity in the same May208 context, which the Grug path satisfies with max abs below `0.001`.
  - The wrapper now satisfies "within `0.01` atol and at least `90%` of upstream perf" when "correctness" is interpreted against the available upstream-local FA4/CuTe implementation. It does not satisfy strict `0.01` max-abs against materialized JAX, and it does not satisfy the alternate `50% SoL` clause.
- Next action:
  - Run focused tests and full pre-commit.

### 2026-06-23 21:08 - B=8 per-device batch scaling probe
- Hypothesis:
  - The low B=1 May208-shard SoL may be dominated by occupancy/split-call overhead; increasing per-device batch to B=8 should raise the upstream-local ceiling if the kernel is throughput-limited by small grid size.
- Command:
  - Ran `/dlwh/fa4-cute-sm90-b8-audit` on H100:
    `B=8,S=4096,Hq=20,Hkv=5,D=128,window=2048`, BF16, warmup 3, iterations 10.
  - Compared production full, direct Grug-mask native backward, and direct built-in-local native backward.
- Result:
  - Production full:
    - forward `2.907771 ms`, `177.277 TFLOP/s`, `17.92% SoL`.
    - backward `3.142215 ms`, `410.125 TFLOP/s`, `41.47% SoL`.
    - backward-with-forward `5.984770 ms`.
  - Direct Grug-mask native backward:
    - `3.150109 ms`, `409.097 TFLOP/s`, `41.36% SoL`.
  - Direct built-in-local native backward:
    - `3.015097 ms`, `427.416 TFLOP/s`, `43.22% SoL`.
  - Production backward is `95.95%` of the B=8 built-in-local upstream path.
- Interpretation:
  - Batching helps materially: direct built-in-local improves from about `32.3% SoL` at B=1 to `43.2% SoL` at B=8.
  - The current Grug path remains close to the upstream-local path at B=8, so the wrapper overhead is not the main remaining gap.
  - Even at B=8, this available upstream-local JAX/CuTe path is still below the alternate `50% SoL` target.
