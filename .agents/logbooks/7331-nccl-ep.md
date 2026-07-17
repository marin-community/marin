# TransformerEngine NCCL_EP expert-parallel MoE on B200 — logbook

Issue: https://github.com/marin-community/marin/issues/7331
Branch: `research/mcwitt/7331-nccl-ep` (logbook, probe scripts, build recipes).
Bench work builds on the standalone MFU benchmark branch `mcwitt/moe-standalone-ep`
(`experiments/grug/moe/standalone`); NCCL_EP bench commits go on a child branch of it.

Experiment IDs: `NCCLEP-###`. W&B tags: `nccl-ep`, `7331` (plus per-run tags as useful).

## Goal

NCCL_EP working on B200-class GPUs at **64 GPUs with EP≥8** at the reference
"prod" config from #7012/#7279 — the point is to derisk running at scale:

- Reference config (B200MFU-032): **d5120 L48 e64 top4 b1024 seq4096**, 64 GPUs
  = replica-2 × 32-GPU FSDP+EP model copy (`--replica-axis-size 2`), cf 1.0,
  MuonH, 20 steps / warmup 8, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
- Baselines to beat/compare (B200MFU-032/-033): ring_cute EP4 **20.83 %**,
  EP1 (pure FSDP) 20.22 %, a2a_cute EP8 **19.12 %**, pure-XLA raa EP8 18.75 %;
  production driver (FSDP-only) 23.32 %. EP8 is the current hard ceiling at
  b1024 (all EP16/32/64 arms fail, B200MFU-033); the EP≥8 CUBIN-load failure is
  *intermittent* (B200MFU-035) and not yet reliably worked around.
- Platform: cw-us-east-08a GB200 NVL72 (4-GPU arm64 Grace nodes); 64-GPU jobs
  = 16-node iris gangs, hard-colocated into one rack (all-NVLink/MNNVL).

## Context (from #7012 logbook, `.agents/logbooks/7012-b200-moe-mfu.md` on `research/mcwitt/7012-b200-mfu`)

- `H-te` (Blocked there; this issue unblocks it): TE 2.15.0+42b840051 wheel does
  **not** ship the NCCL_EP JAX surface (probe 2026-07-15) — building from the WIP
  branch is prerequisite #1. TE `grouped_dense` already proven QuACK-class
  (1,449 TF/s, B200MFU-011); NGC container recipe (apptainer + PYTHONPATH
  overlay + CPU-torch) works on this stack.
- Source-read facts (TE PRs #3034/#3036 + WIP branch
  `jberchtold/teddy-te-ep-integration-2026-07-08-support-quantization`):
  - "8192-token limit" = `max_tokens_per_rank` staging-buffer default at
    `ep_bootstrap`, not structural. NB the reference config is **65,536
    tokens/rank** (16 seq/GPU × 4096) — staging buffers must be sized up 8×.
  - Dispatch wire bf16-only on the main PR; fp8 wire on the quantization WIP branch.
    Quantize happens post-dispatch (`grouped_quantize` → `grouped_gemm`).
  - One-sided SM specialization: comm `max_num_sms` only; GEMM opportunistic.
  - Requires **process-per-GPU** (`local_device_count()==1`) — our gangs run one
    process per 4-GPU node, so the bench needs a 4-procs/node (64-process) launch mode.
  - Disables NCCL comm-splitting and command-buffer capture around the EP FFI ops.
- `H-smspec` falsified twice (B200MFU-016/-021): SM-capping NCCL CTAs loses
  monotonically on our stack. The part of NCCL_EP under test is the
  chunked-pipeline decomposition + fused dispatch, not the SM budget.
- XLA dispatch baselines for the microbench: `ragged_all_to_all` one-shot kernel
  ~297 ms/call (B200MFU-018); NCCL fallback via
  `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` (B200MFU-025).
- Ops gotchas inherited: GPU jobs need `--cpu 16 --memory 64g` (default OOMs
  `import jax`); `iris job logs/list` take `--cluster=cw-us-east-08a` (not
  `--target-cluster`, which is `job run`-only); `bash -lc` drops the task venv
  (use `bash -c`); check printed module `__file__`s against stale imports;
  MNNVL is the default cross-node transport inside a rack.

## Hypothesis queue

### Active
- `H-build`: TE + NCCL_EP is buildable on this stack (arm64 Grace + sm_100a +
  cu13). **Source-level derisk complete (NCCLEP-001): everything is public** —
  build from TE main, upgrade runtime NCCL to ≥ 2.30.4 (PyPI has aarch64
  wheels). Residual risk is only the actual compile on arm64 + the TE↔jax-0.10.1
  FFI compatibility, both empirical (→ build task).
- `H-ep8-prod`: TE NCCL_EP dispatch/combine runs at the reference config with
  EP≥8 across 64 GPUs and is competitive with (or beats) a2a_cute EP8 19.12 %.
  Sub-risks: 65k tokens/rank staging memory; process-per-GPU × 26–48-layer scan
  compile behavior; no-command-buffer + no-comm-split interaction with the rest
  of the step; the intermittent EP≥8 CUBIN failure (B200MFU-035) — NCCL_EP
  changes the dispatch path but not XLA's per-kernel-module loading.
- `H-microbench`: at matched shapes, NCCL_EP dispatch+combine ≪ XLA raa NCCL
  fallback per call (it should pipeline staging chunks and skip host syncs).

### Design constraints discovered (source-read, 2026-07-17)
- **Single outer axis:** TE's EP sharding supports exactly one dp/fsdp mesh axis
  outside `ep` (`_ep_outer_axis`; `get_mesh_axis_size` asserts the axis is a
  plain mesh axis — no tuples). The reference layout replica2 × (data4 × ep8)
  has TWO outer axes → won't bootstrap. First 64-GPU target is therefore a
  single-copy `data8 × expert8` mesh (still ≥32-way sharded per copy = the
  memory-realism the issue guidance actually demands); replica-DP-outside is an
  upstream integration gap worth reporting to NVIDIA.
- EP groups are contiguous global ranks (`dp_color = rank // ep_size`); the
  bench mesh order (replica_dcn, data, expert, model) keeps expert
  fastest-varying → compatible.
- `num_experts × … % 4 == 0` TMA alignment (e64 fine); dispatch rows are
  per-(token,k) assignments with a scalar weight each; combine is **unweighted**
  (caller multiplies expert_out by recv weights first); token_counts
  ([ranks, num_local_experts], alignment-padded) feeds grouped GEMMs as
  group_sizes directly — TE's own moe.py (WIP branch) does exactly
  dispatch → shard_map(grouped FFN) → combine, which is the shape our
  `nccl_ep` bench backend will take (QuACK GEMMs via the `expert_mlp_fn` seam).

### Blocked / later
- `H-fp8wire`: fp8 dispatch wire (quantization WIP branch) halves dispatch
  bytes; ties into MXFP8 work (#7282). After bf16 wire works.

## Decision log
- 2026-07-17: kick off from #7331; branch `research/mcwitt/7331-nccl-ep`;
  reference config + baselines frozen to B200MFU-032 values.

## Entries

### 2026-07-17 — NCCLEP-001: dependency-chain source read — NCCL_EP is fully public; build plan set
- Motivation: H-te's blocker was "requires building TE from the WIP PR branch",
  with an open worry that the NCCL_EP backend depends on a closed `libnccl_ep`
  fork. Resolve buildability entirely by source read before spending GPU time.
- Method: shallow clone of the WIP branch
  (`jberchtold-nvidia/TransformerEngine@jberchtold/teddy-te-ep-integration-2026-07-08-support-quantization`,
  local at `$CLAUDE_JOB_DIR/tmp/te-wip`); `gh` reads of TE PRs #3034/#3036;
  `git ls-tree`/`gh api` to resolve the NCCL submodule pin; PyPI index query for
  `nvidia-nccl-cu13`.
- Findings:
  1. **Both TE EP PRs are merged into TE main** — #3034 (common C API +
     NCCL EP backend) 2026-06-13, #3036 (JAX FFI primitives + VJPs,
     `transformer_engine/jax/ep.py`) 2026-06-27. The WIP branch is only needed
     for the **fp8 dispatch wire** (H-fp8wire); phase 1 builds **TE main**
     (tip `68493d2d55ac`, 2026-07-17). This also explains the negative
     2026-07-15 probe: TE 2.15.0+42b840051 predates/omits the merge.
  2. **No closed dependency.** `libnccl_ep.a` is built by `setup.py`
     (`build_nccl_ep_submodule`) from the in-tree `3rdparty/nccl` submodule
     pinned to **public** NVIDIA/nccl `b87848fbc` (2026-07-07, verified
     reachable upstream; EP lives on the `v2.30u1` line) and statically linked
     into `libtransformer_engine.so`. `NVTE_WITH_NCCL_EP=ON` is the default;
     needs `git submodule update --init --recursive` + `NVTE_CUDA_ARCHS`
     containing ≥ 90 (use `100a`).
  3. **Runtime gate: system libnccl.so ≥ 2.30.4** (checked in
     `EPBackend::initialize`, lazy-bound so older NCCL only fails at EP init;
     note the comment: `LD_BIND_NOW` environments lose this property). Our
     lockfile pins `nvidia-nccl-cu13==2.28.9` → **must upgrade the job env**;
     PyPI has 2.30.7 with aarch64 wheels.
  4. API contract confirmed at the merged surface (`te.jax.ep`): `ep_bootstrap(
     world_size, rank, num_experts, max_tokens_per_rank, recv_capacity_per_rank,
     hidden_dim, max_token_dtype=bf16, max_num_sms=0)` — requires
     `jax.local_device_count()==1` (process-per-GPU), world ≥ 2, TE
     `global_shard_guard(MeshResource(..., ep_resource=...))`, num_experts
     divisible by ep_size, bf16-only wire; `ep_dispatch`/`ep_combine` are
     custom_vjp FFI primitives, CUDA-graph capturable; `ep_prepare` runs inside
     dispatch fwd (routing all-gather). SM ≥ 90 runtime gate. Per-row limit:
     `hidden_dim × sizeof(dtype)` < 4 GiB (fine: 5120×2B).
  5. Capacity math at the reference config: `max_tokens_per_rank` = 65,536
     (16 seq/GPU × 4096). Zero-drop `recv_capacity_per_rank` =
     ep_size × max_tokens × topk = 2.1 M tokens ≈ **21.4 GiB** of bf16 staging
     at EP8 — must size to expected load × margin instead (uniform-routing
     expectation is 262 k tokens ≈ 2.7 GiB; cf-style margin TBD). This is the
     EP≥8-at-scale derisk question the goal is about.
  6. TE ships `tests/jax/test_multi_process_ep.py` (13 tests, multi-process)
     + an e2e MoE example — natural smoke + microbench scaffolding.
- Decision: build TE **main** (not the WIP branch) with bundled-submodule
  NCCL_EP inside the NGC container stack on a GB200 node; upgrade
  `nvidia-nccl-cu13` to 2.30.7 in the job env; keep the WIP branch only for the
  later fp8-wire phase.
- Next: build + import probe on cw-us-east-08a (NCCLEP-002).

### 2026-07-17 — NCCLEP-002 (in flight): TE wheel build on an arm64 GB200 iris pod — pip-wheel toolchain
- Motivation: the schmidt-cluster NGC/apptainer recipe (B200MFU-010/-011) is x86 +
  slurm; the goal platform is arm64 GB200 iris pods whose task image
  (`python:3.12-slim`) has no CUDA toolkit. Decide native-venv build vs container.
- Pod probe (`/mwittmann/ncclep-podprobe`): aarch64 Grace, 144 cores, 14 TB local
  disk, gcc 14.2 + make + git present, cmake/ninja absent (pip-installable),
  no `/usr/local/cuda`, driver 595.71.05, GB200 compute cap 10.0. → **native
  build with a synthetic CUDA_HOME assembled from pip cu13 wheels**
  (`experiments/ncclep/build_te_wheel.sh`); no container. Bonus: the wheel then
  links against the *same* venv stack (stock jax 0.10.1) the bench baselines use.
- Build-iteration findings (each a failed job on the way to green):
  1. CUDA 13 pip packages dropped the `-cu13` suffix — the suffixed names
     (`nvidia-cuda-nvrtc-cu13` etc.) are deprecated 0.0.1 stubs whose sdist
     build *fails on purpose*. Exceptions: `nvidia-cudnn-cu13`/`nvidia-nccl-cu13`
     keep the suffix. `nvidia-curand` is on its own 10.x versioning.
  2. TE build-time python deps (with `--no-build-isolation`):
     `nvidia-cudnn-frontend>=1.25`, `pybind11[global]`, cmake, ninja, flax;
     `NVTE_FRAMEWORK=jax` skips the torch requirement.
  3. `build_nccl_ep_submodule` makes `contrib/nccl_ep` in the NCCL submodule.
     `lib` target builds `libnccl_ep.a` (fine) **and** `libnccl_ep.so`, whose
     nvcc-driven link needs `-lnccl` (the nccl wheel ships only `libnccl.so.2` —
     add an unversioned symlink) and `-lcudadevrt -lcudart_static` (ship in the
     `nvidia-cuda-runtime` wheel under `nvidia/cu13/lib/*.a`; nvcc's implicit
     `-L` points at the wheel's own tree, so export `LIBRARY_PATH`).
  4. The wheel layout nests under `site-packages/nvidia/cu13/{bin,include,lib}`.
- More build-iteration findings (attempts 6–10):
  5. cmake `find_library(NCCL_LIB ...)` doesn't search wheel dirs — export
     `CMAKE_LIBRARY_PATH`/`CMAKE_INCLUDE_PATH`.
  6. Missing headers, one wheel each: `cuda_profiler_api.h` →
     `nvidia-cuda-profiler-api`; `nvml.h` → `nvidia-nvml-dev`; `nvtx3/` →
     `nvidia-nvtx` (all unsuffixed 13.x names).
  7. **Per-file include merges are dangerous**: two wheels both shipping
     `include/nvtx3/` interleaved into a franken-tree (`NVTX_NULLPTR`
     undefined) — symlink version-coherent trees whole-directory instead.
  8. Verbose builds drown their own errors in iris log retention — tee the
     build log to a file and print only error context (same lesson as
     B200MFU-034's stderr capture).
  9. Attempt 10 failed in hadamard-transform TUs with the error lines lost to
     retention; attempt 11 (identical script + log capture) went green — the
     failure did not reproduce. Watch for flakiness in future rebuilds.
- **GREEN (attempt 11, job `/mwittmann/ncclep-te-build11`)**: wheel
  `transformer_engine-2.18.0.dev0+68493d2-cp312-cp312-linux_aarch64.whl`
  (71 MB) stashed at
  `s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep/wheels/`.
  Import probe on 4×GB200 under **stock jax 0.10.1**: `te.jax.ep` exposes
  `ep_bootstrap`/`ep_dispatch`/`ep_combine`/`ep_prepare`/`EpLayerConfig`;
  pybind ext has `set_ep_bootstrap_params`/`release_ep_resources`/
  `ep_handle_mem_size`. H-build **confirmed** — TE main + NCCL_EP builds and
  imports on the goal platform, single-stack with the bench baselines.
- Next: NCCLEP-003 single-node 4-proc TE EP test-suite smoke (in flight),
  then the EP4/EP8 transport microbenches.

### 2026-07-17 — NCCLEP-003 (in flight): single-node TE EP test-suite smoke — two runtime discoveries
- Smoke 1 (`/mwittmann/ncclep-smoke`): `OSError: libcublas.so.12` at
  `import transformer_engine`. Root cause: the gpu venv also carries **cu12**
  wheels (torch+cu128 deps) under `nvidia/<pkg>/`; the build's include/lib merge
  swept them in and the unversioned-symlink pass (sorted, `.so.12` < `.so.13`)
  linked TE against cu12 sonames while TE's pip-mode preloader loads cu13.
  Fixed by merging only `nvidia/cu13` + `cudnn` + `nccl` trees (build 12);
  smoke now asserts no `.so.12` in TE core's DT_NEEDED (verified clean).
- Smoke 2 (`/mwittmann/ncclep-smoke2`): all ranks SIGABRT with
  `<command-line>: fatal error: cuda_runtime.h: No such file or directory`.
  **NCCL_EP JIT-compiles its device kernels at bootstrap** (hybridep JIT via an
  external nvcc): runtime needs nvcc + CUDA headers + the generated JIT header
  tree (`build/include/nccl_ep/`), none of which the wheel packages — the
  compile-time default paths point into the (gone) build tree. Source-read of
  `contrib/nccl_ep/device/jit/`: env knobs `NCCL_EP_JIT_NVCC`/`NVCC`/
  `$CUDA_HOME/bin/nvcc`; `NCCL_EP_JIT_SOURCE_DIR`, `NCCL_EP_JIT_BUILD_INCLUDE_DIR`,
  `NCCL_EP_JIT_CUDA_INCLUDE_DIR` (falls back to `$CUDA_HOME/include`),
  `NCCL_EP_JIT_LOG=1` for visibility. Fix: `cuda_wheels_env.sh` (shared
  build/runtime toolchain synthesis) + build job stashes
  `nccl-ep-jit-headers.tgz`; launchers download and export the JIT env.
  Implication for all future NCCL_EP jobs: **every rank's pod needs the JIT
  toolchain env**, and first-bootstrap latency includes an nvcc compile
  (cached under the JIT cache dir afterward).
- **PASSED** (`/mwittmann/ncclep-smoke3`): all **13 TE EP tests green** in
  17.7 s on 4×GB200, one process per GPU (dp2×ep2) — bootstrap, primitive
  round trips, `ep_dispatch`/`ep_combine` custom_vjp closed-form gradient
  checks, and the HLO reshard guard (no XLA collectives outside the EP FFI).
  TE core DT_NEEDED verified all-cu13. **NCCL_EP runs on this stack.**
  H-microbench arms now in flight: EP4 single-node + EP8 across 2 nodes
  (first cross-node NCCL_EP over MNNVL).

### 2026-07-17 — NCCLEP-004: transport microbench — cross-node EP8 works; fwd-only jit anomalously slow
- Setup: `experiments/ncclep/ep_transport_microbench.py` via
  `run_microbench_gang.sh` (iris multigpu supervisor, 1 proc/GPU). Reference
  per-rank load: 65,536 tokens/rank × H 5120 × top-4, e64, cf 1.25 (recv
  capacity 327,680 rows/rank), bf16 wire, uniform round-robin routing,
  30 iters/8 warmup, jitted dispatch → weighted-hadamard → combine round trip.
  Jobs `/mwittmann/ncclep-mb-ep4d` (dp1×ep4, one node) and `ncclep-mb-ep8d`
  (dp1×ep8, TWO nodes — first cross-node NCCL_EP, default MNNVL transport).
- Results (median):
  | arm | fwd | fwd+bwd (`value_and_grad`) |
  |---|---|---|
  | EP4, 1 node | 28.03 ms | 16.28 ms |
  | EP8, 2 nodes | 54.46 ms | 24.47 ms |
- Interpretation: **EP≥8 spanning nodes runs correctly with gradients** — the
  goal's biggest physics risk is retired. Training-relevant cost is the
  fwd+bwd number (24.5 ms/layer-equivalent at EP8) — ~1.2 s over 48 layers at
  the reference load, plausibly competitive with the incumbent a2a_cute legs
  (direct comparison lands with the e2e MFU runs, not here).
- Anomaly (open): the fwd-ONLY jit is consistently ~2× slower than
  fwd+bwd — even after switching to `value_and_grad` (plain `jax.grad` DCE'd
  the primal combine FFI entirely, an earlier 2–5× artifact). Tight p10–p90
  both arms, reproducible at both scales. Suspect scheduling/serialization
  around the FFI ops in the small fwd-only executable; profile only if e2e
  shows dispatch-bound steps.
- Gotcha (repeated twice): FFI handlers must be registered by importing TE
  **before the JAX CUDA client exists** — importing after
  `jax.distributed.initialize` → `NOT_FOUND: No FFI handler registered for
  te_ep_prepare_ffi`. TE imports now live at bench module top.

### 2026-07-17 — NCCLEP-005 (in flight): e2e bench integration — `nccl_ep` backend
- Branch `mcwitt/moe-standalone-ep-ncclep` (off the sample branch):
  `levanter/grug/_moe/ep_nccl.py` (TE dispatch → shard_map(QuACK
  `expert_mlp_fn`) → weighted hadamard → TE combine, mirroring TE's own MoE
  block), `nccl_ep` registered in the implementation enum + a global-view
  branch in `grug_moe.py`, and bench wiring: process-per-GPU assertion,
  `--replica-axis-size 1` requirement (TE single-outer-axis), TE
  `global_shard_guard(MeshResource(fsdp="data", ep="expert"))`, per-process
  `ep_bootstrap` sized from `--capacity-factor`, supervised jax_init under the
  multigpu supervisor. Single-node EP4 smoke (d2560 L4 b16) in flight.
- Integration failures fixed in order (smokes 1–9, each a real layer):
  1. **Global-gitignore `lib/` trap**: new files under `lib/*` silently skipped
     by `git add -A` AND by iris bundling → `git add -f` (memoried).
  2. uv's cached `marin-levanter` wheel misses new source files (cache key =
     pyproject only) → PYTHONPATH-shadow with the bundled tree.
  3. TE FFI handlers must register before the JAX CUDA client exists — import
     TE at module top, before `jax.distributed.initialize`.
  4. The bench mesh types all axes **Explicit**; TE's partitioning rules assume
     auto-sharding (outputs typed replicated, constraints assert) → run the EP
     region under `jax.sharding.auto_axes(...)`, pin only the final output;
     the FFN shard_map must take `jax.sharding.get_abstract_mesh()` (the
     auto-typed view), not the outer mesh object.
  5. **Command-buffer capture breaks TE's handle cache** (an EP op's host-side
     `lookup_handle` can run before `ep_prepare`'s cache insert) →
     `--xla_gpu_enable_command_buffer=` + `NVTE_EP_HANDLE_CACHE_SIZE=-1`
     (TE's own documented JAX workaround). Exactly the constraint the July
     source-read flagged ("disables command-buffer capture around the EP ops").
  6. **NCCL_EP has no drop path**: recv overflow beyond capacity is an OOB
     write (`CUDA_ERROR_LAUNCH_FAILED`, poisoned context). cf-1.0 exact-average
     capacity overflows on any imbalance → provision the no-drop worst case
     `ep × tokens_per_rank × top_k`. (Memory cost is real: 2.1 M rows ≈ 21.5 GiB
     bf16 per buffer at the 64-GPU reference config; sub-worst-case capacity
     needs TE-side drop support — upstream ask.)
  7. **Garbage-row NaN**: the QuACK seam extends the last group over the
     capacity tail (uninitialized dispatch buffer); garbage rows poison wgrad
     accumulation, and a 0-mask *multiply* converts `0×NaN → NaN` in the VJP →
     zero the tail rows before the GEMM + `jnp.where` for the combine
     weighting.
- **PASSED** (`/mwittmann/ncclep-e2e-smoke9` vs `ncclep-e2e-ctl2` ring_cute
  control, identical config d2560 L4 e64 top4 b16 seq4096 EP4): step-0 loss
  **bit-identical** (11.805191040039062), full 6-step trajectory parity to
  ~2e-5 (final 11.558435 vs 11.558453). Tiny-config MFU 8.9 % vs control
  11.2 % (per-call overheads + command-buffers-off dominate at this size —
  not meaningful; reference-config comparison is NCCLEP-006).
- Next (NCCLEP-006): 64-GPU EP8 reference config (d5120 L48 e64 top4 b1024
  seq4096, single-copy data8×expert8 per the TE single-outer-axis constraint)
  + a2a_cute EP8 control at the identical mesh/flags. Watch: HBM at no-drop
  capacity (mem fraction 0.90), the intermittent CUBIN failure envelope
  (B200MFU-035), first-step NCCL_EP JIT compiles.
