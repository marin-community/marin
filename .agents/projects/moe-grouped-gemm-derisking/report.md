# De-risking the MoE grouped-GEMM path for the d6144 hero

Date: 2026-09-07. Branch: `claude/moe-grouped-gemm-derisking-m9ood6`. Code state: `main` at `61e59be`.

## How to read this

This session ran in a container with no GPU and no cluster credentials (Iris and Echo both
refuse without an IAP identity). Nothing below was measured fresh. Every number is one of:

- a measurement recorded in this repository's own history (PR bodies, issue threads, the
  `#8317` tuning ledger on `research/mcwitt/8317-ragged-tune`), cited where it appears;
- an arithmetic consequence of those measurements and the model shape;
- a property of the code, with file and line references into `main`, QuACK 0.6.1
  (`02c7f69`), and CUTLASS DSL 4.6.0 (`cutlass/jax`, read from the v4.6.0 tag).

Where the task asked for a number that does not exist anywhere in that record, the section
says so and points at the harness under `scripts/` that produces it on a GB200.

## The three things that reframe the question

**1. The stated target does not match the code.** The task says Hopper and FP8. The hero recipe
in the repository is GB200 NVL72 in bf16 (`experiments/grug/moe_hero_ep/README.md`;
`HERO_MIXED_PRECISION = "params=bfloat16,compute=bfloat16,output=bfloat16"` in
`hero_recipe.py:40`). The QuACK wrapper is gated to compute capability 10.0
(`lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py:47,127`) and accepts only
bf16, fp16 and fp32 (`quack_moe_cute.py:33-37`); the weight-gradient wrapper refuses anything
but 16-bit floats (`quack_moe_cute.py:267-268`). There is no FP8 code anywhere under
`levanter/grug` (grep for `float8|fp8|e4m3` returns nothing). On an H100 the ragged EP backend
silently selects `_ragged_dot_expert_mlp` (`ep_ragged_all_to_all.py:144-153`), which runs
haliax's Pallas-Triton kernel at tokamax's untuned default block sizes. If the run really is
Hopper/FP8, the QuACK wrapper is not the risk, because it never executes; the fallback is, and
it has had no tuning at all. The rest of this report assumes the run the repository describes:
GB200, bf16, 535 B total / 22.8 B active (the shape checks out analytically, see Task 2).

**2. The production hero is not running the QuACK grouped GEMM today.** `HERO_MODEL` selects
`fixed_pooled_wave_all_to_all` (`heuristic.py:111`, restored by PR #8884 after the ragged
transport hung at 11 racks, issue #8870). That backend computes the expert MLP as two dense
`jnp.einsum` calls over fixed-capacity cells (`ep_fixed_pooled_wave_all_to_all.py:458-460`),
i.e. XLA-autotuned cuBLAS batched GEMMs. QuACK grouped GEMMs run only on the opt-in
`ragged_all_to_all` backend (`ep_ragged_all_to_all.py:97-120`), which the scaling ladder
selects for the sub-hero rungs (`launch_scaling_ladder.py:132`) and which is meant to take the
hero back once #8870 is closed. QuACK is also in the dropless evaluator
(`train.py:87`, `sonic_cute`) and in Newton-Schulz (`grugmuon_hero.py:189-203`, symmetric GEMM).
So the decision in front of you is whether to move the hero *onto* the QuACK path, not whether
to move it off.

**3. The benchmark that "favors this path" compared the wrong things.** PR #8814's cuDNN
comparison covered only the two weight-gradient GEMMs, against a cuDNN grouped-Wgrad wrapper
that needed 256-row-aligned groups and a 365 ms/step alignment copy. The four activation-path
GEMMs were never benchmarked against anything except haliax's `ragged_dot`, and
`bench_grouped_wgrad.py` labels that row "xla ragged_dot" while it actually dispatches to the
Pallas-Triton kernel on GPU (`haliax/nn/ragged_dot.py:392-393`). The end-to-end evidence that
does exist is a tie: ragged transport with QuACK versus pooled-wave with dense cuBLAS measured
22.87 vs 22.71 MFU (PR #8549) and 22.46 vs 22.52 MFU restored from the step-6000 checkpoint
(#8317, 2026-08-23), with the transport and the kernel changing together.

## Task 1: baselines

### What is actually reachable from this stack

| Path | Reachable? | What it really is on NVIDIA | Tuning it has received |
| --- | --- | --- | --- |
| `jax.lax.ragged_dot_general` (XLA) | yes | `RaggedDotRewriter` expands the ragged dot into a dense dot over a group-broadcast, group-masked LHS: `[rows, K]` becomes `[rows, E, K]` and the dot contracts over `(E, K)` (`xla/hlo/transforms/expanders/ragged_dot_rewriter.cc`, `RaggedToDense` and `CreateRaggedNonContractingDotDims`). E times the FLOPs and an E-times temporary. The weight gradient (ragged contracting mode) expands both operands. Autotuning picks the cuBLAS algorithm for the resulting dense dot; it cannot recover the E-times work. | none needed; structurally 3x FLOPs per chunk (E_chunk = 3), 384x on a non-EP layout |
| XLA cuDNN ragged-dot fusion | not on the pinned stack | `--xla_gpu_experimental_use_ragged_dot_fusion=true` lowers bf16/f16 non-contracting ragged dots to a cuDNN grouped GEMM (`ragged_dot_fusion_rewriter.cc`; gate at `ragged_dot_rewriter.cc:403-420`). Requires cuDNN >= 9.22 (`kMinCudnnVersionForRaggedDotFusion`); the lock pins `nvidia-cudnn-cu13 9.19.0.56`. Weight gradients still take the expansion. The flag exists in jaxlib 0.11.1 (`strings libjax_common.so`). | zero; needs a cuDNN bump to even run. This is the "cuDNN does provide grouped GEMM" path the task premise says does not exist. |
| cuBLASLt grouped GEMM | no | XLA's `--xla_gpu_experimental_use_ragged_dot_grouped_gemm` is hipBLASLt-only (`CanBeHandledByGpublasltGroupGemm`, ROCm branch only). No JAX binding for `cublasLtMatmul` grouped mode in this stack. Reaching it means a custom FFI extension. | n/a |
| CUTLASS grouped GEMM | yes, and it is QuACK | From JAX, the CUTLASS path *is* the CuTe-DSL one: `cutlass.jax.cutlass_call` plus QuACK's `GemmSm100` kernels. Megatron's GroupedMLP uses the C++ CUTLASS/TE grouped GEMM through PyTorch; that has no JAX binding here. | the shipped configuration, tuned at the hero shape in #8317 (see below) |
| Pallas-Triton (haliax) | yes | tokamax's `_ragged_dot` kernel vendored into `haliax/nn/ragged_dot.py:107-290`, at tokamax's own default config: `block_m=128, block_n=128 (256 on Blackwell), block_k=32, num_warps=4, num_stages=4`. Tokamax marks that default `TODO: Create heuristics` and autotunes over a grid when used through its op API; haliax hardcodes the default. `ragged_dot` also pads M to a multiple of 512 (`ragged_dot.py:430-432`). | none |
| Pallas Mosaic-GPU (Blackwell) grouped GEMM | on a branch | PR #8320 (`marin-ep`, draft) runs its down GEMM on "the Pallas Blackwell grouped GEMM"; not on `main`. | unknown |
| Padded dense cuBLAS (`nvjet`) | yes | What the pooled-wave hero runs (`ep_fixed_pooled_wave_all_to_all.py:458-460`), and what the `ep-ragged-dense` arm ran on the ragged transport. | XLA autotune; SwiGLU unfused |

### Every kernel-level measurement that exists at the hero chunk shape

The chunk shape is one EP64 receiver chunk: 301,466 rows (capacity 1.15 x 65,536 tokens x
top-8 / 2 chunks), 3 experts, latent 3072 in, intermediate 3072 (`bench_grouped_wgrad.py:54-57`).
Group sizes were uneven and off 256-row alignment. Source: PR #8814 body, measured on one GB200.

| GEMM | QuACK varlen-k, tile (256,256) cluster (2,2,1) | cuDNN grouped Wgrad (256-row-aligned wrapper) | QuACK at its default tile |
| --- | --- | --- | --- |
| dw13 `[rows,3072]^T @ [rows,6144]` | 7.22 ms, ~1.55 PFLOP/s | 10.15 ms, ~1.10 PFLOP/s | slower than cuDNN (not quantified in the record) |
| dw2 `[rows,3072]^T @ [rows,3072]` | 3.41 ms, ~1.64 PFLOP/s | 4.68 ms, ~1.19 PFLOP/s | slower than cuDNN |

TFLOP/s here is my arithmetic from the recorded milliseconds at ~98% active rows; treat the
peak as ~2.25-2.5 PFLOP/s dense bf16 per B200/GB200 GPU, so the tuned QuACK weight gradients
run at roughly 60-70% of peak. No activation-path (gate/up, down, dh, dx) kernel-level
number exists in the record for any implementation. The only activation-path evidence is
end-to-end:

| Arm (hero shape, EP64, one rack, #8317 ledger) | MFU | Delta | What changed |
| --- | --- | --- | --- |
| h17b: QuACK at default tile (256,128), no CLC | 20.20% | | frontier before GEMM tuning |
| h20: tile (256,256) | 21.03% | +0.83 | scheduling only |
| h21: + CLC persistence | 21.63% | +0.60 | scheduling only |
| h22: + cluster (2,2,1) on the plain grouped GEMMs | 21.82% | +0.19 | scheduling only |
| h14: padded-dense cuBLAS experts on the same ragged transport | 19.84% | -0.16 vs h02 (20.00%) | kernel family; SwiGLU unfused, ~15% padding FLOPs |
| h13: padded-dense at EP16 / i6272 | 22.56% | +0.5 vs grouped | wider experts favor dense |

Before tuning, the Aug-15 anatomy put the grouped expert GEMMs at "~4.5 s/step at ~40%
efficiency vs ~65% for the dense path's nvjet batched GEMMs". After tuning, the +1.62 MFU
corresponds to about 1.3 s/step less, i.e. roughly 3.1 s/step at about 57% efficiency. So the
honest kernel-vs-kernel picture at the hero shape is: tuned QuACK is somewhat faster than
padded dense cuBLAS on the same transport, by an amount that was never isolated after tuning
and is on the order of 1-2 MFU points; untuned QuACK was a wash with it.

### Tuning effort per baseline, and where more tuning would close a gap

- **QuACK**: tile, cluster, swizzle and CLC were swept at the hero shape (#8317) and
  `bench_grouped_wgrad.py --sweep` covers the weight gradients. The knob space is exhausted.
- **Pallas-Triton**: zero tuning. `block_k=32` is small for a 3072-deep contraction on
  Blackwell; a sweep over `block_k in {64,128}` and `num_stages` is the obvious first move and
  is what the `--sweep` flag in `scripts/bench_grouped_gemm_baselines.py` does. Expect a real
  gain, but Triton on SM100 does not use tcgen05/TMEM, so its ceiling is well below QuACK's.
- **XLA expansion**: nothing to tune; it does 3x the work per chunk. It is the correctness
  reference, not a throughput candidate.
- **cuDNN fusion**: untested; needs cuDNN 9.22+. Worth one afternoon once the bump is in,
  because it is the only zero-custom-code grouped kernel on NVIDIA.
- **Padded dense**: the SwiGLU materialization and window gather/scatter were never fused;
  h13/h14 suggest that at i3072 the padding tax roughly cancels nvjet's efficiency edge.

## Task 2: how much of the step is the expert GEMM

Per-token FLOPs at the hero shape (my arithmetic from `heuristic.py:86-118`, forward only):

| Component per layer | MFLOP/token | Share of per-layer FLOPs |
| --- | --- | --- |
| routed experts (2 x 3 x 3072 x 3072 x 8) | 453.0 | 45% |
| shared experts (2 x 3 x 6144 x 3072 x 2) | 226.5 | 23% |
| attention projections | 188.7 | 19% |
| latent down/up projections | 75.5 | 8% |
| attention scores (3 of 4 layers windowed at 2048) | 50.3 | 5% |
| router | 4.7 | <1% |

Routed expert GEMMs are 44% of model FLOPs once the LM head is included. Active parameters come
out at 22.8 B and total at 535 B, matching the task's numbers.

Measured step anatomy on the ragged path at the hero shape, one rack (#8317, merged
per-stream intervals from the h05/h22 XProf traces, and the leg-2 keep):

| Bucket | s/step (h22 era) | s/step (leg-2 keep) | Share |
| --- | --- | --- | --- |
| compute stream busy | 13.52 | 14.52 | 77% |
| collectives, exposed | 3.22 | ~2.4 | 18% |
| device idle | 0.54 | | 3% |
| step span | 17.54 | 17.4 | |

Within the compute stream, "GEMMs" (all of them, dense included) were 57.7% = 8.55 s/step in
the h22-era trace. The routed grouped GEMMs alone were "~4.5 s/step at ~40% efficiency" before
tuning, which agrees with the analytic share (44% of the 4.0 s of peak-equivalent FLOP time a
22.8% MFU step contains is 1.76 s; at 40% efficiency that is 4.4 s). After the +1.62 MFU
tuning, about 3.1 s/step. One comment in the same thread attributes "~9 s/step" to the QuACK
activation GEMMs; that exceeds the whole GEMM bucket and is inconsistent with the efficiency
figure, so I treat it as an over-attribution.

The ceiling:

- Expert grouped GEMMs are about 3.1-4.5 s of a 17.5 s step, i.e. **18-26% of step time**.
  An infinitely fast expert GEMM would raise MFU from ~22.8% to roughly 28-31%.
- That is not under 5%, so the kernel choice is a first-order lever. But the realistic spread
  between implementations that exist (tuned QuACK at ~57% efficiency, padded dense cuBLAS at
  ~65% on more FLOPs, an autotuned Triton kernel at an unknown but lower ceiling) is about
  +/-1 s/step, i.e. **+/-5% of step time, +/-1 MFU point**. Swapping kernels cannot move the
  needle more than the two larger, unrelated buckets the same traces indict: 2.4-3.2 s/step of
  fully exposed collectives and the ~5 s/step non-GEMM fusion/permute tail.
- The 10-20% throughput loss the task is willing to accept for safety corresponds to the whole
  expert-GEMM bucket running 2x slower; no candidate baseline is that bad except the XLA
  expansion.

## Task 3: wrapper audit

Files: `lib/levanter/src/levanter/grug/_moe/quack_moe_cute.py` (wrapper),
`sonic_cute.py` (custom VJPs and tile configs), `ep_ragged_all_to_all.py` (selection and
buffer bookkeeping), `lib/levanter/src/levanter/cutlass_kernel_cache.py` (compile cache),
QuACK 0.6.1 `quack/tile_scheduler.py`, `quack/varlen_utils.py`, `quack/gemm_sm100.py`,
CUTLASS DSL 4.6.0 `cutlass/jax/{primitive,compile,types}.py`.

### Host-device syncs and problem descriptors: none, and device-side

`cu_seqlens` is passed as a device int32 array through a dynamic `TensorSpec`
(`quack_moe_cute.py:128,141,194,198`). Nothing calls `.tolist()`, `np.asarray`, or
`block_until_ready` on it. The kernel reads group boundaries on device: the varlen-M scheduler
computes per-group tile counts from `cu_seqlens_m` in-kernel with warp prefix sums
(`tile_scheduler.py:1082-1160`), and the varlen-K path reads `cu_seqlens_k[batch_idx]` for the
K-loop length (`gemm_sm100.py:1509-1510`). The launch grid is sized on the host from static
quantities only: `total_m` (the static buffer row count) and the group count, using the worst-
case bound `(total_m + L*(block-1)) // block` (`tile_scheduler.py:1011-1035`). Group sizes are
therefore never needed at trace time and there is no sync on the critical path.

### Zero-token experts and severe imbalance

- varlen-M (gate/up, down, dh, dx): an empty group contributes zero tiles
  (`_get_num_m_blocks`, `tile_scheduler.py:1082-1095`); nothing is read or written for it.
- varlen-K (dw13, dw2): an empty group clears the accumulator and writes zeros
  (`clear_acc=(varlen_k and k_len == 0)`, `gemm_sm100.py:1694`). Covered by
  `tests/kernels/test_quack_grouped_wgrad.py` (`empty-group`, `leading-empty-groups`).
- Imbalance: all rows in one expert is just one long batch; the over-provisioned grid handles
  any split. Rows past `cu[-1]` are never written; the backend masks them
  (`_zero_inactive_grouped_rows`, `sonic_cute.py:119,135,140`).

### The concrete defect: the CLC padding drain in QuACK 0.6.1

The shipped activation-path configuration uses Cluster Launch Control persistence
(`_QUACK_USE_CLC = True`, `sonic_cute.py:58-60`, worth +0.60 MFU). The varlen-M scheduler
over-provisions the grid and relies on `cancel_pending_tail` (`tile_scheduler.py:504-559`) to
drain phantom clusters. In 0.6.1 that drain fires unconditionally at retirement, including a
retirement caused by a *spurious* invalid `try_cancel` response, and issues further cancels
after observing failure, which the PTX ISA defines as undefined behavior. Its own docstring
records the unproven "grant monotonicity" assumption (`tile_scheduler.py:509-517`). Upstream
fixed it in `Dao-AILab/quack@58cb592` (shipped in 0.6.4) after finding that real pending tiles
got cancelled and their output rows kept stale allocator memory, under three conditions: a
co-tenant using the GPU, >= ~10 CLC grids queued back-to-back on a stream, and a grid much
larger than the resident capacity (`AI/clc_spurious_invalid_investigation.md` at that commit).

This is not hypothetical for us. The #8870 investigation reproduced it on GB200 with a
competing process: the 0.6.1 scheduler passed 1 of 4 contention cases, the drain fix and the
no-CLC control passed all of theirs, and the corrupted elements retained the output sentinel
(unwritten tiles). The training loop queues six grouped GEMMs per chunk, two chunks per layer,
48 layers, while NCCL device kernels share the SMs, so two of the three ingredients are always
present. The same-process competing-stream test was still pending when the thread went quiet.
A silent wrong tile in one expert's output is exactly the failure class the task is most
worried about. Branch `upgrade-quack-064` carries 0.6.4 plus the wrapper changes for its new
batch-first tensor convention (`quack_moe_cute.py` there uses `mode=(0, 2, 1)` where main uses
`(2, 1, 0)`); it is not merged.

### Kernel selection determinism

There is no runtime autotuner. Tile, cluster, swizzle and persistence mode are Python constants
(`sonic_cute.py:57-68`), so the compiled kernel is the same across process restarts. Within a
kernel, each output tile is owned by exactly one CTA with a fixed K order; there is no split-K
and no atomics in either grouping mode, so outputs are bit-deterministic regardless of which SM
the CLC scheduler hands a tile to. The record agrees: loss was bit-identical (11.806) across
every scheduling arm in #8317. The persistent object cache keys on launcher configuration,
`levanter/grug` source hash, the workspace lock hash, the argument spec and the device
architecture (`cutlass_kernel_cache.py:169-201`); it does not key on the CUDA driver or ptxas
version, and a non-locked QuACK checkout would not be covered by the lock hash.

### Aliasing and donation

No `input_output_aliases` are passed anywhere (`quack_moe_cute.py:131-140,191-197`), so
every call writes fresh XLA-owned outputs and cannot clobber a live input. The bridge itself is
`jax.ffi.ffi_call` with row-major layout constraints (`cutlass/jax/primitive.py:319-333`). Two
consequences worth knowing: the pre-activation `gu` `[C, 2I]` bf16 (3.7 GB per chunk at the hero
shape) is a second kernel output kept as a residual for the backward, and the rows past `cu[-1]`
of every output are uninitialized memory. The backend masks `y` and `dx`; `gu`, `h` and `dh`
padding rows are garbage that flows into the elementwise SwiGLU backward and then into kernels
that never read those rows, so it does not reach results, but it would trip a NaN checker and
it depends on the "never reads past `cu`" property holding for every kernel.

### Sharding

`cutlass_call` has no sharding, batching, JVP or transpose rule (`primitive.py:336-364`). The
wrapper is only ever invoked inside `shard_map` bodies (`grug_moe.py:270-288` for EP,
`grug_moe.py:340-353` for the no-EP path) on per-device local arrays, inside `custom_vjp`
functions, so none of those rules is ever needed. All collectives around it (`ragged_all_to_all`,
`all_gather`, `psum`) are JAX ops owned by the backend, not the kernel. `check_vma=False` on
both `shard_map` calls means JAX does not verify manual-axis variance.

### Workspace, alignment and layout assumptions

- No workspace: static or CLC persistence needs no tile-count semaphore
  (`make_scheduler_args(mac, swizzle, None)`); TMA descriptors are built in-kernel from runtime
  shapes and compact strides (`cutlass/jax/types.py:446-505`).
- Pointer alignment assumed 256 bytes (`TensorSpec.ptr_assumed_align`), which XLA guarantees for
  buffer allocations. Static slices of the expert axis (`ep_ragged_all_to_all.py:399-400`) land
  on multiples of 3072 x 6144 x 2 bytes, so even an offset view would be aligned.
- The specs promise 16-byte vectorization on the feature dims (`divisibility=(1, 8)`,
  `quack_moe_cute.py:125-130,224-228,287-289`). Only `quack_grouped_wgrad` verifies the promise
  (`:271-272`); `quack_gated_grouped_gemm` and `quack_grouped_gemm` do not, so a hidden or
  intermediate width not divisible by 8 would compile a kernel with a false alignment assumption.
  Not reachable at the hero shape, but it is a silent-failure hole.
- `mode` permutations (`(2,1,0)`, `(1,2,0)`, `(1,0)`) are the load-bearing wiring. A wrong
  permutation returns plausibly-shaped wrong numbers. The weight-gradient wiring has a unit test;
  **the gated and plain activation-path GEMMs have no unit test at all**. They are covered only by
  the 4-GPU `ragged_ep_check` (`lib/marin/src/marin/testing/moe/ragged_ep.py`), which runs on
  demand and was found never to have been run before August (#8578).

### Compile behavior

`FunctionSpec` includes every argument's concrete shape (`cutlass/jax/compile.py:44-71`), so a
kernel compiles once per distinct shape even though `static=False` makes the binary
shape-agnostic. On the ragged path that is six CuTe compiles per process for the hero
(gate/up, down, dh, dx, dw13, dw2; both chunks share a shape), plus the dropless evaluator and
Newton-Schulz kernels; the hung run logged nine CuTe cache keys per task (#8870). Each is a
`cute.compile` under a global lock (`compile.py:213-243`). JAX's persistent compilation cache does
not cover them, which is why `cutlass_kernel_cache.py` exists; with that cache warm the delta is
one object-store fetch per kernel. The cold delta was not measured in the record; the cold hero
compile was 22.9 min for the watch executable plus 6.6 min for the plain one (#8870), which is
XLA end to end and does not separate the CuTe share.

### Error handling on unsupported input

| Input | Behavior |
| --- | --- |
| dtype outside bf16/fp16/fp32 | `KeyError` at trace time (`quack_moe_cute.py:40-41`): clean |
| fp32 operands to the activation-path GEMMs | accepted, computed in TF32 with no diagnostic; only the wgrad wrapper refuses (`:264-268`) |
| feature dim not a multiple of 8 | activation path: unchecked, kernel compiled with a false divisibility; wgrad: `ValueError` |
| `cu[-1] > rows`, non-monotone `cu`, `len(cu)-1 != E` | unchecked; out-of-bounds reads or writes in the kernel, so a fault at best and silent garbage at worst. The backend's capacity clipping is what keeps this from happening. |
| activation other than SiLU | falls back to `ragged_dot` (`ep_ragged_all_to_all.py:151`): clean |
| non-SM100 GPU | falls back to `ragged_dot` with a warning: clean, and the reason the Hopper question matters |

## Task 4: numerics

### What the record establishes

- Both kernel families take bf16, accumulate in fp32, write bf16. The QuACK gated epilogue
  evaluates SwiGLU on the fp32 accumulator before rounding (`quack/gemm_act.py`), so its `h` is
  slightly *more* accurate than the `ragged_dot` path, which rounds gate/up to bf16 first. The
  backward recomputes the SwiGLU derivative from the bf16 `gu` in both paths
  (`common.py:70-82`), so the two agree to a rounding.
- Weight gradients: QuACK reproduced the corrected reference to three significant figures on
  four seeds, and the unaligned/empty-group unit tests pass 9/9 on GB200 (PR #8814).
- Whole ragged EP path against an exact fp32 dense MoE on 4 GPUs: 0.4-0.5% relative deviation
  in values, bf16 reduction-order noise in gradients (median-per-slice gate at 5e-2,
  `ragged_ep.py:83-94`), both dropless and in the drop regime (#8317 leg 2, PR #8549).
- Multi-step: no controlled kernel-only divergence run exists. The d768 10.8k-step pair (ragged
  + QuACK vs pooled + dense cuBLAS) had ragged ahead on loss and evals, but the drop rate differed
  by 87x, so it says nothing about the kernel.
- FP8: not applicable; there is no FP8 quantization, scaling factor or accumulation-precision
  question on this path. If FP8 is introduced, QuACK's SM100 GEMM does support fp8 operands, but
  its blockscaled scale-factor layout is padded per group (`quack/gemm_tvm_ffi_utils.py:25-108`)
  and the wrapper would need to produce it; that is new code with its own audit.

### What I could not measure, and the harness that does it

`scripts/numerics_expert_mlp.py` drives the two `_ExpertMlp` callables the ragged backend
selects between, through the same buffer bookkeeping, and reports per-tensor max/mean absolute
and max relative error for the forward, `dx`, `dw13` and `dw2` against an fp32 per-group
reference, over nine cases: balanced, uneven, tile-unaligned, empty middle group, leading empty
groups, one expert holding every row, one-row groups, extreme imbalance, all empty. It also
asserts QuACK's padding rows are exactly zero. `--hero` runs the real chunk shape. The
ragged_dot and reference rows were validated on CPU (all cases finite, forward error
5e-3 to 7e-3 of scale, gradients 3e-3 to 8e-3, as expected for bf16); the QuACK rows need a
GB200. `scripts/multistep_divergence.py` trains a small MoE stack twice from one initialization
with each kernel and logs loss gap plus max relative divergence of parameters and Adam moments
every ten steps; its self-check (ragged_dot vs ragged_dot) is exactly zero at every step on CPU.
`scripts/bench_grouped_gemm_baselines.py` and `run_all.sh` produce the Task 1 table for every
reachable implementation, one process each, with the cuDNN-fusion flag isolated.

## Task 5: options and recommendation

Throughput is relative to the current wrapper on the ragged transport (h22 frontier 21.8%,
leg-2 keep 22.8% MFU).

**Option A. Keep the QuACK wrapper, with these mitigations as preconditions.**
Expected throughput 1.00x (0.97x if CLC has to stay off). Effort: days.

1. Land QuACK 0.6.4 (branch `upgrade-quack-064`) or, until it is validated end to end, set
   `_QUACK_USE_CLC = False` in `sonic_cute.py:58`. Cost of the latter: -0.60 MFU (h21 vs h20).
   Do not run the hero on 0.6.1 with CLC on.
2. Add unit tests for the gated and plain varlen-M GEMMs mirroring
   `test_quack_grouped_wgrad.py` (uneven, unaligned, empty, `b_major` both ways, NaN sentinel
   past `cu[-1]`), and add the missing guards in the activation-path wrappers: feature dims
   divisible by 8, 16-bit dtype, `len(cu) - 1 == E`. Assert `cu[-1] <= rows` in
   `_cute_expert_mlp` with a checkify or at least a debug mode.
3. Run `scripts/run_all.sh` on one GB200 (an hour) and `multistep_divergence.py --steps 300`
   before the hero moves back to ragged, and run `ragged_ep_check` on every QuACK or CUTLASS
   DSL bump.
4. Retained risks: a proprietary, experimental `cutlass.jax` bridge (its own docstring says so),
   QuACK API churn (0.6.4 changed the calling convention), and a compile path outside JAX's
   cache. Eliminated: the known silent-corruption defect, the untested wiring, the silent
   TF32/alignment holes.

**Option B. Padded-dense cuBLAS experts on the ragged transport (`ep-ragged-dense`).**
Expected throughput 0.92-0.99x (h14 was -0.16 MFU against *untuned* QuACK; against the tuned
frontier the gap is probably 1-2 MFU points, unmeasured). Effort: the arm exists on
`research/mcwitt/8317-ragged-tune`; porting is days. No custom kernels, XLA-autotuned cuBLAS,
deterministic, SwiGLU unfused. Retained risk: memory (a `[E_chunk, cap, 2I]` dense
intermediate) at a step that already sits 0.3 GiB under the allocator threshold, and ~15%
padding FLOPs that grow with router skew. This is the "simpler and more standard" path and it
lands at rough parity; it is the right fallback if Option A's harness turns up a numerics
problem.

**Option C. Pallas Mosaic-GPU Blackwell grouped GEMM (tokamax, or the kernel in PR #8320).**
Expected throughput unknown; effort weeks, plus tuning. Open source, JAX-native, autotunable.
Not a de-risking move on this timeline: it trades a kernel with 25 hero-scale runs behind it
for one with none.

**If the target really is H100/FP8**, none of the above applies. The reachable grouped GEMMs
there are the untuned Pallas-Triton kernel (sweep `block_k`, `num_stages`; expect a large gain
from zero effort so far), tokamax's Mosaic-GPU Hopper ragged dot, or a new wrapper over
QuACK's `GemmSm90` (supports varlen and fp8 k-major). That is a different project.

### Recommendation

Option A. The measured end-to-end distance between the QuACK path and the simplest standard
alternative is one or two MFU points in QuACK's favor, and the largest risk found is a
specific, reproduced, already-fixed-upstream scheduler defect rather than anything structural
in the wrapper: no host syncs, device-side descriptors, deterministic configs, no aliasing,
correct empty-group handling in both grouping modes. The wrapper's real weaknesses are test
coverage and input validation, both cheap to fix.

I would change this recommendation if any of the following comes back from the harness:

- the QuACK rows of `numerics_expert_mlp.py --hero` show forward or gradient error more than
  about 3x the ragged_dot row on any case, or any non-zero padding row;
- `multistep_divergence.py --steps 300` shows parameter or Adam-state divergence between QuACK
  and ragged_dot that keeps growing after step ~100, where a ragged_dot-vs-Triton control
  plateaus at bf16 noise (~1e-3 relative);
- the same-process competing-stream CLC test still corrupts output on 0.6.4 or with CLC off,
  which would indicate a second scheduler defect;
- the hero is actually going to Hopper, in which case the question is the Triton fallback's
  tuning, not this wrapper.
