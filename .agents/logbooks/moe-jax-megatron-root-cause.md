# JAX vs Megatron MoE GPU Gap: Research Logbook

## Scope
- Goal: identify the concrete root causes behind the discrepancy between the negative JAX custom-call result in `#3665` and the more positive torch/Megatron results in `#3641` and `#3666`.
- Primary metric(s):
  - matched H100x8 forward/backward wall time
  - phase-level timing deltas where possible
  - attributable delta by hypothesis category
- Constraints:
  - Use CoreWeave Iris H100x8 via `~/llms/cw_ops_guide.md`.
  - Keep the experiment falsifiable: matched comparisons first, wider tuning second.
  - Include a direct JAX/Marin vs Megatron-LM head-to-head.
- Experiment issue: https://github.com/marin-community/marin/issues/3677

## Baseline
- Date: 2026-03-14
- Code refs:
  - `#3641` seal: `moe-deepep-hybrid-ep-seal-20260314`
  - `#3665` seal: `moe-deepep-jax-layout-ffi-h100-matrix-20260314`
  - `#3666` seal: `moe-megatron-qwen-scale-h100-matrix-20260314`
- Baseline numbers:
  - `#3665` fixed-shape H100x8:
    - `current` beat `deepep_layout_ragged_a2a` by about `1.48x` to `1.81x` on distributed cells
    - `deepep_layout_ragged_a2a` stayed roughly tied with `ragged_a2a`
  - `#3641` fixed-shape torch-side H100x8:
    - patched `hybrid_ep` / `hybrid_ep_permute` beat `deep_ep` on the replicated `runs` slice
  - `#3666` Megatron full-layer H100x8:
    - `deepep` won most `128`-expert Qwen-like points
    - `hybridep` won the authoritative `32` and `64` expert reruns

## Initial Hypotheses
- The dominant root cause is kernel coverage: `#3665` only replaced `get_dispatch_layout`, while `#3641` / `#3666` benefited from real dispatch/combine transport kernels.
- The next root cause is benchmark scope: `#3665` timed the sealed `#3633` fixed-shape path, while `#3666` timed a full `MoELayer` with grouped GEMM and different compute/communication balance.
- JAX-specific overheads may matter, but only after the first two causes are controlled for; otherwise they are easy to overstate.
- A direct same-shape torch dispatch/combine benchmark with optional JAX->Torch bridging should help isolate whether the missing gains are mostly in transport kernels or mostly in the broader Megatron methodology.

## Stop Criteria
- Produce a direct JAX/Marin vs Megatron-LM head-to-head on CoreWeave H100x8.
- Produce a root-cause table that identifies which differences are necessary and/or sufficient to explain the gap.
- Update the issue body with a concise conclusion that is more specific than "JAX is slower" or "Megatron is better."

## Experiment Log
### 2026-03-14 21:05 - Kick off the new root-cause thread
- Commands:
  ```bash
  git worktree add -b research/moe-jax-megatron-root-cause \
    /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    moe-megatron-qwen-scale-h100-matrix-20260314

  sed -n '1,220p' /Users/romain/dev/marin/.agents/skills/agent-research/SKILL.md
  sed -n '1,260p' /Users/romain/llms/cw_ops_guide.md
  gh issue view 3641 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3665 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3666 --repo marin-community/marin --json title,body,state,comments --comments
  ```
- Config:
  - branch: `research/moe-jax-megatron-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-root-cause`
  - CoreWeave kubeconfig: `~/.kube/coreweave-iris`
- Result:
  - Reloaded the `agent-research` workflow and the CoreWeave ops guide.
  - Re-established the three relevant sealed baselines:
    - torch-side DeepEP / Hybrid-EP wins in `#3641`
    - JAX layout-only negative result in `#3665`
    - Megatron full-layer ranking shift in `#3666`
  - Decided to structure the new thread around root-cause separation rather than another one-off benchmark.
- Initial root-cause matrix:
  - `methodology`: dispatch-only vs full `MoELayer`
  - `kernel coverage`: layout-only vs dispatch/combine transport
  - `backend`: JAX `ragged_all_to_all` / current ring EP vs DeepEP / Hybrid-EP
  - `interop`: optional JAX -> Torch bridge cost
  - `timing hygiene`: dummy GEMM, warmup length, fixed inputs, router dtype
- Next action:
  - Create the experiment issue and link it back here.
  - Audit the relevant JAX / torch / Megatron code paths before launching the first H100x8 comparison.

### 2026-03-14 22:00 - Matched same-shape JAX vs Megatron head-to-head
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_krt_bench.py \
    --repo-ref 4807bb3f3dbfe654977c978f57900a297cc421f5 \
    --task-id jax-megatron-root-cause-jax-forward-20260314-1445 \
    --skip-smoke \
    --shared-expert-dim 0 \
    --bench-pass forward \
    --warmup 5 \
    --iters 20 \
    --topk-list 2,8 \
    --distributions random,runs \
    --kernels current,ragged_a2a,deepep_layout_ragged_a2a \
    --ep-list 1,2,4,8

  uv run python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id jax-megatron-root-cause-megatron-20260314-1458 \
    --cases marin_3633_topk_2,marin_3633_topk_8 \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 5 \
    --measure-iters 20
  ```
- Result:
  - JAX/Marin fixed-shape forward-only control stayed negative even after removing the shared-expert branch and using longer timing windows.
  - On distributed JAX cells, `current / ragged_a2a` ranged from `1.73x` to `2.48x`.
  - `deepep_layout_ragged_a2a` stayed effectively tied with `ragged_a2a`; most cells were within `~1%`, worst-case about `+4.2%`.
  - Same-shape Megatron `MoELayer` runs were positive on both `topk=2` and `topk=8`:
    - `topk=2`: `deepep` `1.89x` forward / `3.26x` backward faster than `alltoall`; `hybridep` `2.32x` forward / `2.35x` backward faster
    - `topk=8`: `deepep` `2.33x` forward / `3.39x` backward faster than `alltoall`; `hybridep` `1.82x` forward / `2.00x` backward faster
  - So the cross-framework discrepancy reproduces on the same H100x8 shape; it is not just a reporting artifact from unrelated model scales.
- Additional signal:
  - JAX emitted repeated `gemm_fusion_autotuner` slow-kernel warnings on the large grouped expert matmuls for the `topk=8` slices.
  - This looks like an absolute JAX throughput factor, but it is not needed to explain the relative `deepep_layout_ragged_a2a ~= ragged_a2a` result.
- Next action:
  - Measure raw DeepEP dispatch/combine on the same shape and separate:
    - layout cost
    - steady-state transport cost
    - JAX -> Torch bridge cost
    - Torch -> JAX bridge cost

### 2026-03-14 22:20 - Direct DeepEP dispatch/combine isolation with explicit JAX bridge
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id jax-megatron-root-cause-dispatch-matrix-jax-20260314-1517 \
    --topk-list 2,8 \
    --distributions random,runs \
    --input-sources torch,jax \
    --return-to-jax \
    --warmup 5 \
    --iters 20
  ```
- Launcher/debug notes:
  - First dispatch attempt failed because `Buffer.dispatch` was called with `topk_idx` as the second positional argument, which DeepEP interprets as a cached `handle`.
  - Fixed the harness to use keyword arguments for the non-cached path and a real cached `handle` for the steady-state path.
  - Second attempt failed because the PyTorch image did not include JAX.
  - Fixed the launcher to install `jax[cuda12]==0.8.0` only when the matrix includes `input_source=jax`.
- Result:
  - Raw torch-side DeepEP dispatch/combine on the sealed shape is very fast:
    - `random, topk=2`: `full_s=0.000486` (`67.44M tokens/s`)
    - `random, topk=8`: `full_s=0.001062` (`30.85M tokens/s`)
    - `runs, topk=2`: `full_s=0.000913` (`35.90M tokens/s`)
    - `runs, topk=8`: `full_s=0.001298` (`25.25M tokens/s`)
  - Once tensors are already in Torch, the steady-state dispatch/combine cost for JAX-originating tensors is in the same ballpark:
    - `random, topk=2`: JAX/Torch `full_s` ratio `1.02x`
    - `random, topk=8`: JAX/Torch `full_s` ratio `1.36x`
    - `runs, topk=2`: JAX/Torch `full_s` ratio `0.96x`
    - `runs, topk=8`: JAX/Torch `full_s` ratio `0.92x`
  - The bridge itself is the expensive part:
    - `bridge_to_torch_s`: about `85 ms` to `105 ms`
    - `bridge_to_jax_s`: about `2 ms` to `12 ms`
  - Therefore:
    - a direct JAX -> Torch DLPack bridge every training step would dominate the sub-millisecond to low-millisecond transport kernel time
    - but the raw DeepEP transport kernel is not inherently incompatible with JAX-shaped inputs once those tensors are already on the Torch side
- Root-cause update:
  - This confirms that the missing JAX gains are primarily about missing transport kernel coverage, not about DeepEP transport itself being bad on the sealed shape.
  - It also shows that a naive per-step Torch interop path is not a viable substitute for a real JAX custom call, because the bridge cost would erase the gain.

### 2026-03-14 16:48 PDT - Pure-JAX transport custom-call bring-up: compile fixes and runtime-model mismatch
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1712 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --timeout-seconds 7200

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1721 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --timeout-seconds 7200

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1730 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added a pure-JAX transport wrapper at `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
  - Added the CUDA/XLA FFI transport binding at `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
  - Added a dedicated smoke/bench harness at `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
  - Added CoreWeave launcher `.agents/scripts/deepep_jax_transport_krt.py`
- Iteration results:
  - First remote compile failed because the FFI wrapper's `deep_ep::intranode::dispatch(...)` call did not match the pinned DeepEP ref `567632dd59810d77b3cc05553df953cc0f779799`.
  - After fixing that signature drift, the next remote compile failed with:
    - `atomicAdd_system` / `atomicSub_system` undefined in `runtime.cu`
  - Root cause for that step: the ad hoc raw `nvcc` build path was missing an explicit Hopper architecture target, so it was not compiling with the same arch assumptions as upstream DeepEP's `setup.py`.
  - After aligning the raw `nvcc` flags more closely with upstream (`sm_90`, `-rdc=true`, `--ptxas-options=--register-usage-level=10`, `DISABLE_AGGRESSIVE_PTX_INSTRS`), compilation succeeded.
- Runtime result after compilation succeeded:
  - The real runtime init failure surfaced as:
    - `cudaIpcOpenMemHandle: invalid device context`
  - This is more informative than the earlier `runtime.cu:29 illegal memory access`, which turned out to be teardown masking a partially initialized runtime.
- External references checked:
  - NVIDIA developer forum guidance says `cudaIpcOpenMemHandle` is for another process, and opening the exported handle in the creating process is wrong.
  - CUDA Programming Guide peer-access guidance says that in a single process with unified virtual addressing and peer access enabled, the same pointer can be used from both devices.
- Interpretation:
  - The pure-JAX runtime manager was incorrectly reusing DeepEP's inter-process CUDA IPC pattern inside JAX's single-process, multi-GPU execution model.
  - So the latest blocker is no longer "can the FFI compile?" but "the runtime model must change from inter-process IPC to same-process peer-access/UVA if this is going to work in standard JAX."
- Next action:
  - Replace the custom runtime manager's `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` path with same-process peer access (`cudaDeviceCanAccessPeer` / `cudaDeviceEnablePeerAccess`) and direct peer pointers, then rerun the same single-cell H100x8 smoke.

### 2026-03-14 17:58 PDT - Same-process peer-access runtime reaches DeepEP intranode launch, then fails in kernel-launch compatibility
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1751 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --timeout-seconds 7200

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1758 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --disable-sm90-features \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Replaced the experimental runtime manager's inter-process CUDA IPC setup with same-process peer access and direct peer pointers.
  - Fixed the FFI attribute typing bug by passing `num_experts` as `np.int32` under `JAX_ENABLE_X64=1`.
  - Added support for upstream DeepEP's `DISABLE_SM90_FEATURES` compatibility knob in the raw custom-call build path.
- Result:
  - The peer-access rewrite removed the previous `cudaIpcOpenMemHandle: invalid device context` failure and let the benchmark enter the actual DeepEP intranode dispatch path.
  - First pure peer-access run (`sm_90` path intact) failed at the real DeepEP launch site with:
    - assertion in `/tmp/DeepEP/csrc/kernels/intranode.cu:608`
    - `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess`
  - Compatibility rerun with `DISABLE_SM90_FEATURES=1` changed the failure mode:
    - repeated `named symbol not found` from the same dispatch site (`intranode.cu:608`)
    - followed by DeepEP timeout-check failures in the intranode barrier path and eventual `CUDA_ERROR_LAUNCH_FAILED`
- Interpretation:
  - This is progress: the failure frontier is no longer the JAX process model or the FFI wrapper typing, but DeepEP's own intranode dispatch launch assumptions under this pure-JAX custom-call build.
  - The two current launch-side blockers are:
    1. SM90/TMA path: dynamic-shared-memory attribute setup fails on H100 in this build mode.
    2. `DISABLE_SM90_FEATURES` fallback path: gets past that attribute but then fails at kernel launch / symbol resolution and eventually deadlocks the intranode barrier.
- Next action:
  - Search for public reports around these two exact launch failures (`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` on H100 and `named symbol not found` on the fallback path).

### 2026-03-14 18:55 PDT - Same-image Torch controls isolate an older-ref build defect from the deeper H100 launch blocker
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref 567632dd59810d77b3cc05553df953cc0f779799 \
    --task-id deepep-torch-control-20260314-1910 \
    --distributions random \
    --topk-list 2 \
    --input-sources torch \
    --warmup 1 \
    --iters 3

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-torch-control-hybrid-ep-20260314-1928 \
    --distributions random \
    --topk-list 2 \
    --input-sources torch \
    --warmup 1 \
    --iters 3

  curl -L --silent https://raw.githubusercontent.com/deepseek-ai/DeepEP/567632dd59810d77b3cc05553df953cc0f779799/setup.py | sed -n '1,260p'
  curl -L --silent https://raw.githubusercontent.com/deepseek-ai/DeepEP/hybrid-ep/setup.py | sed -n '1,260p'
  ```
- Result:
  - The pinned old ref failed in the same CUDA 12.8 / H100 pod before any JAX code ran:
    - `ImportError: ... deep_ep_cpp ... undefined symbol: __cudaRegisterLinkedBinary_...`
  - The paired `hybrid-ep` control succeeded in the same image and ran the fixed-shape torch slice:
    - `dispatch_combine_full_s=0.000502`
    - `tokens_per_s=65.23M`
  - Upstream `setup.py` diff explains the divergence:
    - the old pinned ref enables `-rdc=true` on the H100 path but does not force a device-link step when NVSHMEM is disabled
    - `hybrid-ep` adds exactly that guard and appends `-dlink` in the non-NVSHMEM H100 path
- Interpretation:
  - The old pinned ref was not a trustworthy pure-JAX baseline in this environment because it had a separate CUDA registration/device-link defect.
  - The environment itself is not broken, because the paired `hybrid-ep` torch control ran successfully in the same pod.
- Next action:
  - Port the pure-JAX wrapper to the `hybrid-ep` `intranode::dispatch(...)` signature and rerun the H100 smokes there.

### 2026-03-14 20:10 PDT - Pure-JAX wrapper ported to `hybrid-ep`; deeper H100 launch failures persist on the newer branch
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-jax-transport-hybrid-ep-20260314-2030 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-jax-transport-hybrid-ep-sm80-20260314-2038 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --disable-sm90-features
  ```
- Code changes under test:
  - Added retry logic to `.agents/scripts/deepep_jax_transport_krt.py` so transient `codeload.github.com` timeouts do not kill the pod.
  - Added a small compile-time compatibility shim in `transport_ffi.py` / `deepep_transport_ffi.cu` so the same wrapper can target the widened `hybrid-ep` `intranode::dispatch(...)` signature.
- Result:
  - Baseline SM90 path now compiles and reaches the real DeepEP intranode launch on `hybrid-ep`, but fails with:
    - `/tmp/DeepEP/csrc/kernels/intranode.cu:521`
    - `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess`
  - `DISABLE_SM90_FEATURES=1` on the newer branch flips the failure mode to:
    - `CUDA error ... /tmp/DeepEP/csrc/kernels/intranode.cu:521 'named symbol not found'`
    - followed by the same timeout / launch-failure cleanup tail
- Interpretation:
  - The old-ref device-link defect was real, but it was not the whole story.
  - After removing that factor and porting the wrapper to `hybrid-ep`, the pure-JAX path still lands on the same substantive DeepEP H100 launch blockers.
  - So the remaining problem is no longer old-ref drift or wrapper signature mismatch; it is the actual DeepEP intranode launch/runtime assumptions under this pure-JAX execution model.
- Next action:
  - Treat the remaining work as kernel-launch/runtime debugging, not build drift.
  - If continuing, instrument the failing H100 launch path directly (`smem_size`, kernel attributes, launch variant selection) rather than spending more time on older-ref compatibility.
  - If no exact fix appears upstream, inspect whether our raw `nvcc` shared-library build still differs materially from DeepEP's working PyTorch extension build in a way that affects launch registration or symbol resolution.

### 2026-03-14 18:33 PDT - Build-frontend diagnostic: PyTorch extension toolchain changes the failure mode again
- Command:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id deepep-jax-transport-smoke-20260314-1833 \
    --topk-list 2 \
    --distributions random \
    --warmup 1 \
    --iters 1 \
    --build-with-torch-extension \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added an opt-in build path that compiles the same wrapper plus DeepEP kernel sources through `torch.utils.cpp_extension.load(..., is_python_module=False)`.
  - Passed the missing JAX XLA FFI include directory into that build.
- Result:
  - The extension-based build gets materially further than the first raw-`nvcc` attempt:
    - compilation succeeds for `deepep_transport_ffi.cu`, `runtime.cu`, `layout.cu`, and `intranode.cu`
    - the final shared object link step also succeeds
  - But library load then fails with:
    - `undefined symbol: __cudaRegisterLinkedBinary_a14bd921_23_deepep_transport_ffi_cu_8a7790fa`
    - surfaced as `OSError: Could not load this library: .../deepep_transport_ffi_<digest>.so`
- Interpretation:
  - This is strong evidence that build/link mechanics are still part of the problem, not just runtime or kernel semantics.
  - The extension-toolchain experiment changed the failure from kernel-launch/runtime assertions into a CUDA registration / device-linking problem at shared-library load time.
  - So the current pure-JAX blockers split cleanly into two classes:
    1. Raw-`nvcc` build path can load, but then fails inside DeepEP launch/runtime assumptions.
    2. Extension-like build path gets a more upstream-faithful compile, but currently lacks the CUDA registration/device-link setup needed for the resulting `.so` to load cleanly in this custom-call flow.
- Next action:
  - Decide whether to keep pursuing the extension-style build by adding the missing CUDA device-link registration step, or treat this as enough evidence that the build frontend itself is a real integration barrier and fold that into the root-cause report.

### 2026-03-14 21:58 PDT - Deterministic interface probe shows Torch and JAX agree before the DeepEP launch boundary
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-interface-probe-torch-rerun-20260314-2141 \
    --tokens 16 \
    --hidden 8 \
    --experts 16 \
    --topk-list 2 \
    --distributions deterministic \
    --input-sources torch \
    --probe-only \
    --probe-max-elements 512 \
    --warmup 1 \
    --iters 1 \
    --timeout-seconds 7200

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-interface-probe-jax-20260314-2110 \
    --tokens 16 \
    --hidden 8 \
    --experts 16 \
    --topk-list 2 \
    --distributions deterministic \
    --probe-only \
    --probe-max-elements 512 \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added deterministic `--probe-only` support to:
    - `lib/levanter/scripts/bench/bench_deepep_dispatch.py`
    - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
  - Added launcher plumbing for the new probe flags in:
    - `.agents/scripts/deepep_dispatch_krt_bench.py`
    - `.agents/scripts/deepep_jax_transport_krt.py`
- Result:
  - Both frameworks printed a structured `PROBE_JSON` before entering the failing transport launch.
  - Common semantic fields matched exactly on rank 0:
    - `x`: same `bfloat16 [2, 8]` values `[[0.0, 0.125, ..., 0.875], [1.0, 1.125, ..., 1.875]]`
    - `topk_idx`: `[[0, 1], [1, 2]]`
    - `topk_weights`: `[[0.6666667, 0.33333334], [0.6666667, 0.33333334]]`
    - `num_tokens_per_rank`: `[2, 1, 0, 0, 0, 0, 0, 0]`
    - `num_tokens_per_expert`: `[1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
    - `is_token_in_rank` matched exactly for both tokens
    - dispatch config: `num_sms=20 num_max_send_tokens=6 num_max_recv_tokens=256`
    - combine config: `num_sms=20 num_max_send_tokens=4 num_max_recv_tokens=256`
  - Differences were only probe-format details:
    - Torch reports framework-prefixed dtype strings like `torch.int64`; JAX reports normalized dtype names
    - Torch includes extra bookkeeping fields (`bridge_to_torch_s`, `local_rank`, `local_world_size`, `num_nvl_bytes`, `num_rdma_bytes`)
- Interpretation:
  - This rules out a high-level boundary bug where JAX feeds obviously different routing/layout data into DeepEP.
  - The next debugging layer is the actual transport launch/runtime state after this point.
- Next action:
  - Instrument the selected intranode kernel variant plus requested dynamic shared memory and function attributes, then compare that low-level launch state between the working Torch path and failing pure-JAX path.

### 2026-03-14 19:56 PDT - Launch instrumentation shows the raw JAX FFI build fails at CUDA kernel symbol resolution, not at high-level dispatch intent
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-launch-debug-torch-20260314-2230 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --input-sources torch \
    --warmup 0 \
    --iters 1 \
    --launch-debug \
    --launch-debug-label torch \
    --timeout-seconds 7200

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-launch-debug-jax-20260314-2230 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --warmup 0 \
    --iters 1 \
    --launch-debug \
    --launch-debug-label jax \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added `.agents/scripts/patch_deepep_intranode_launch_debug.py` to patch `DeepEP/csrc/kernels/intranode.cu` inside the pod.
  - Added `--launch-debug` / `--launch-debug-label` plumbing to:
    - `.agents/scripts/deepep_dispatch_krt_bench.py`
    - `.agents/scripts/deepep_jax_transport_krt.py`
  - The patch injects env-gated `DEEPEP_LAUNCH_DEBUG {...}` records with:
    - dispatch specialization metadata
    - requested dynamic shared memory
    - `cudaFuncGetAttributes(...)`
    - a probe `cudaFuncSetAttribute(...)`
- Result:
  - Torch control produced valid H100 dispatch metadata on the same shape:
    - `attr_status_code=0`, `set_attr_status_code=0`
    - `binary_version=90`, `ptx_version=90`
    - `requested_smem_bytes=196608`
    - `num_sms=20`, `num_threads=768`, `cluster_dim_x=2`
    - `num_regs=80`, `max_threads_per_block=768`
  - The pure-JAX raw FFI build reached the same host dispatch site with the same host-side intent:
    - `num_tokens=4096`
    - `hidden_int4=256`
    - `num_topk=2`
    - `num_experts=128`
    - `num_max_send_tokens=6`
    - `num_recv_buffer_tokens=256`
    - `requested_smem_bytes=196608`
  - But its kernel symbol lookup already failed at the CUDA runtime boundary:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - all function-attribute fields came back zero (`binary_version=0`, `ptx_version=0`, `max_dynamic_smem_bytes=0`, ...)
  - The later JAX assertion at `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)` is therefore downstream of this earlier missing-symbol condition.
- Interpretation:
  - This is stronger than “some low-level JAX runtime issue.”
  - The raw custom-call `.so` is not presenting a resolvable CUDA dispatch kernel symbol with valid attributes, while the Torch build of the same DeepEP source does.
  - So the current primary blocker is build / device-link / runtime registration in the pure-JAX FFI shared library, not routing data, not the chosen dispatch config, and not an obvious stream mismatch.
- Next action:
  - Focus the next debug step on raw-build CUDA registration/device-link mechanics. A meaningful next test is explicit device-link/runtime registration for the raw JAX FFI build rather than more shape sweeps.

### 2026-03-14 20:25 PDT - Explicit raw `-dlink` plus an arch-pinned final link removed the stale `sm_52` fallback, but the DeepEP dispatch symbol is still unresolved
- Commands:
  ```bash
  python -m py_compile \
    lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
    .agents/scripts/deepep_jax_transport_krt.py \
    .agents/scripts/patch_deepep_intranode_launch_debug.py

  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-launch-debug-jax-dlink-20260314-2358 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --warmup 0 \
    --iters 1 \
    --launch-debug \
    --launch-debug-label jax_dlink_archfix \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Refactored `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py` raw build path to:
    - compile each `.cu` source to an object with RDC enabled
    - run an explicit `nvcc -dlink` step
    - perform the final shared-library link with `--cudart=shared` and the explicit H100 `-gencode`
  - Added the Python builder source plus a cache schema version to the shared-library hash so build-pipeline edits force a fresh `.so`.
- Result:
  - The rerun rebuilt from scratch and no longer emitted the earlier:
    - `nvlink warning : SM Arch ('sm_52') not found ...`
  - The actual launch debug line on the rebuilt raw `.so` remained unchanged at the DeepEP dispatch boundary:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - `requested_smem_bytes=196608`
    - all function-attribute fields still zero
  - JAX still aborts at the downstream DeepEP assertion:
    - `/tmp/DeepEP/csrc/kernels/intranode.cu:642 'cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess'`
- Interpretation:
  - The missing explicit arch on the final raw link was a real secondary build bug; fixing it removed the stale `sm_52` fallback warning.
  - But that was not the primary cause of the pure-JAX failure. Even after an explicit object-compile + `-dlink` pipeline and a guaranteed fresh rebuild, the raw custom-call `.so` still does not expose a resolvable DeepEP dispatch kernel symbol with valid CUDA function attributes.
  - That narrows the remaining problem further: the unresolved symbol is now very likely in raw CUDA device registration / fatbin packaging mechanics, not in launch intent, not in cache staleness, and not in the final-link arch selection itself.
- Next action:
  - Inspect the rebuilt raw `.so` against a working Torch-extension artifact at the binary level (fatbin sections, registration symbols, and device-link outputs) before trying more kernel-launch tweaks.

### 2026-03-14 20:38 PDT - Artifact inspection rules out a gross raw-packaging failure; `RTLD_GLOBAL` also does not fix the JAX launch
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_transport_build_inspect_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-transport-build-inspect-upstream-20260314-2045 \
    --gpus 1 \
    --cpu 16 \
    --memory-gib 96 \
    --disk-gib 128 \
    --timeout-seconds 7200 \
    --build-modes upstream

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-launch-debug-jax-rtld-global-20260314-2052 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --warmup 0 \
    --iters 1 \
    --launch-debug \
    --launch-debug-label jax_rtld_global \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added `.agents/scripts/deepep_transport_build_inspect_krt.py` to inspect built DeepEP/JAX transport artifacts on CoreWeave.
  - Changed `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py` to load the raw shared library with `ctypes.CDLL(..., RTLD_NOW | RTLD_GLOBAL)` instead of `ctypes.cdll.LoadLibrary(...)`.
- Result:
  - The raw custom-call `.so` looks structurally healthy on disk:
    - dynamic deps: `libcudart.so.12`, `libstdc++.so.6`, `libgcc_s.so.1`, `libc.so.6`
    - sections include `.nv_fatbin`
    - `cuobjdump --list-elf` shows two embedded `sm_90` cubins
    - exported symbols include:
      - `__cudaRegisterLinkedBinary_*`
      - `__fatbinwrap_*`
      - the expected DeepEP intranode dispatch/combine specializations
    - the explicit raw `deepep_transport_ffi.dlink.o` also contains `.nv_fatbin` and its own `sm_90` cubin
  - The installed upstream `deep_ep_cpp` artifact is also structurally healthy:
    - `.nv_fatbin`
    - one embedded `sm_90` cubin
    - the expected `__cudaRegisterLinkedBinary_*`, `__fatbinwrap_*`, and `deep_ep::intranode::*` symbols
    - dynamic deps include `libcudart.so.12` plus `libc10.so`, `libtorch_cpu.so`, `libtorch_python.so`, `libc10_cuda.so`
  - The explicit `RTLD_GLOBAL` load-mode test did not change the JAX failure:
    - launch debug still reports `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - all function-attribute fields remain zero
- Interpretation:
  - The remaining blocker is no longer well-described as “the raw `.so` forgot to include the DeepEP kernels” or “ctypes loaded the library too locally.”
  - The raw custom-call artifact on disk contains the expected cubins and registration symbols, and forcing `RTLD_GLOBAL` does not help.
  - That pushes the remaining suspicion further toward a runtime execution-model issue: something about how the JAX/XLA custom-call path reaches `cudaFuncGetAttributes` / `cudaFuncSetAttribute` still differs materially from the working upstream DeepEP/Torch execution path after the library is already loaded.
- Next action:
  - Add a tiny exported host-side probe in the custom-call `.so` that calls `cudaFuncGetAttributes` on the same dispatch specialization immediately after load / runtime init, before entering the XLA custom call, to separate “library loaded incorrectly” from “custom-call execution context is wrong.”

### 2026-03-14 20:44 PDT - A host-side dispatch probe reproduces the same `named symbol not found` failure before XLA execution
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref refs/heads/hybrid-ep \
    --task-id deepep-host-kernel-probe-launch-debug-20260314-2111 \
    --tokens 32768 \
    --hidden 2048 \
    --experts 128 \
    --topk-list 2 \
    --distributions random \
    --host-kernel-probe-only \
    --launch-debug \
    --launch-debug-label host_probe \
    --warmup 0 \
    --iters 1 \
    --timeout-seconds 7200
  ```
- Code changes under test:
  - Added a host-side probe entrypoint to `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`.
  - Added a Python wrapper plus `--host-kernel-probe-only` path in:
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
    - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
    - `.agents/scripts/deepep_jax_transport_krt.py`
  - The probe initializes the existing intranode runtime, allocates dummy device buffers, and calls `deep_ep::intranode::dispatch(...)` directly, without entering `jax.jit` / XLA execution.
- Result:
  - The patched DeepEP source emitted the same launch-debug line from the host probe path:
    - `label="host_probe"`
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - `requested_smem_bytes=196608`
    - all function-attribute fields zero
  - The host probe then failed with:
    - `Failed: Assertion error /tmp/DeepEP/csrc/kernels/intranode.cu:642 'cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess'`
- Interpretation:
  - This cleanly rules out the idea that the remaining failure is specific to XLA custom-call execution, `shard_map`, or an XLA-owned stream/context.
  - The failure is already present in the raw custom-call library + runtime-init path before XLA executes anything.
  - Combined with the artifact inspection, the remaining problem is now best framed as a process/runtime registration issue in the pure-JAX raw shared library path:
    - the cubins and `__cudaRegisterLinkedBinary_*` symbols are present on disk
    - but the raw library still cannot resolve the DeepEP dispatch kernel symbol through CUDA runtime queries, even from a plain host-side call in the same process
- Next action:
  - Compare the raw library load/init path against a working Python-extension-style load path even more directly, for example by loading the same raw `.so` through CPython extension import machinery or by probing immediately after load but before runtime init, to isolate whether the remaining divergence is in module load/registration timing rather than XLA.

### 2026-03-14 21:45 PDT - Python-extension load alone does not help, but a CUDAExtension-style build/load finally clears the kernel-symbol blocker and exposes deeper execution issues
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref 7febc6e25660af0f54d95dd781ecdcd62265ecca \
    --task-id deepep-host-probe-pyext-20260314-2125 \
    --host-kernel-probe-only \
    --launch-debug \
    --launch-debug-label host_probe_pyext \
    --load-as-python-module

  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref 7febc6e25660af0f54d95dd781ecdcd62265ecca \
    --task-id deepep-host-probe-cudaext-pyext-preload-20260314-2145 \
    --host-kernel-probe-only \
    --launch-debug \
    --launch-debug-label host_probe_cudaext_pyext_preload \
    --build-with-torch-extension \
    --load-as-python-module

  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref 7febc6e25660af0f54d95dd781ecdcd62265ecca \
    --task-id deepep-full-jax-cudaext-pyext-20260314-2150 \
    --launch-debug \
    --launch-debug-label jax_cudaext_pyext \
    --build-with-torch-extension \
    --load-as-python-module \
    --warmup 0 \
    --iters 1

  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --deepep-ref 7febc6e25660af0f54d95dd781ecdcd62265ecca \
    --task-id deepep-full-jax-cudaext-pyext-pmap-20260314-2200 \
    --launch-debug \
    --launch-debug-label jax_cudaext_pyext_pmap \
    --build-with-torch-extension \
    --load-as-python-module \
    --execution-model pmap \
    --warmup 0 \
    --iters 1
  ```
- Code changes under test:
  - Added a minimal CPython shim source at `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_pyext.cc`.
  - Added a raw CPython-extension load mode in `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`.
  - Added a stronger `CUDAExtension` build/load path in `transport_ffi.py` that:
    - builds the JAX transport wrapper as a Python extension with explicit RDC + `nvcc` device-link,
    - preloads Torch shared libraries before import,
    - then exposes the same JAX FFI symbols through the imported extension module.
  - Added an alternate `pmap` execution mode in `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py` and `.agents/scripts/deepep_jax_transport_krt.py`.
- Result:
  - Raw CPython-extension loading alone did **not** change the original failure:
    - `host_probe_pyext` still logged:
      - `attr_status_code=500`, `attr_status="named symbol not found"`
      - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
  - The first Torch-extension Python-module attempt failed too early to be useful:
    - import failed with `undefined symbol: __cudaRegisterLinkedBinary...`
  - Replacing that with a real `CUDAExtension` build plus Torch-library preload changed the situation materially:
    - the host probe now logs a healthy H100 dispatch kernel:
      - `attr_status_code=0`, `attr_status="no error"`
      - `set_attr_status_code=0`, `set_attr_status="no error"`
      - `binary_version=90`, `ptx_version=90`
      - `num_regs=80`, `max_threads_per_block=768`
      - `requested_smem_bytes=196608`
    - this is the first pure-JAX-path build/load variant that clears the earlier kernel-symbol-resolution blocker
  - The full `shard_map` JAX path still does not complete:
    - it now gets past launch-attribute setup, but later fails during live transport with:
      - repeated DeepEP receiver timeouts (`tokens remained: ...`)
      - `cudaMemcpyAsync(read num_recv_tokens): unspecified launch failure`
  - The `pmap` alternative fails in a different place:
    - before transport launch, JAX aborts while sharding inputs with:
      - `cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- Interpretation:
  - The old “kernel symbol cannot be resolved from the pure-JAX artifact” blocker is now no longer fundamental. A closer-to-upstream `CUDAExtension` build/load path can make that part work.
  - The surviving failures are deeper and different:
    - `shard_map` reaches real dispatch and then fails in multi-GPU progress / completion,
    - `pmap` collides with peer-access state before launch.
  - So the remaining problem is no longer just CUDA device-registration / fatbin packaging. It is now a mix of:
    - JAX execution/progress model vs DeepEP's expected intranode rendezvous behavior, and
    - runtime ownership of peer-access state.
- Next action:
  - Treat the `CUDAExtension` + preload path as the new working baseline for further pure-JAX bring-up.
  - Move runtime initialization / peer-access setup later so JAX input placement can finish before we touch peer access.
  - Add a non-JAX host control that drives all local ranks concurrently from the same process, to check whether the post-launch timeout is a JAX scheduling issue or a same-process DeepEP issue more generally.

### 2026-03-15 17:31 PDT - Same-process host-only dispatch round on isolated CW H100x8 lane

- Hypothesis:
  - If a host-only all-ranks dispatch round still hangs after the `CUDAExtension` / Python-extension path clears kernel symbol resolution, then the remaining blocker is not specific to `jax.jit`, `shard_map`, or `pmap`.
- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --build-with-torch-extension \
    --load-as-python-module \
    --host-dispatch-round-only \
    --launch-debug \
    --launch-debug-label host-round \
    --topk-list 2 \
    --distributions random
  ```
- Config:
  - Isolated CoreWeave lane: namespace `iris-3677-jax`, label prefix `i3677jax`
  - Hardware: one warmed `H100x8` node
  - DeepEP ref: `7febc6e25660af0f54d95dd781ecdcd62265ecca`
  - Repo ref: `2b2d2e36b6efbde8624cee237479642421e24b4e`
  - Dispatch shape inside the host round: `num_tokens=4096` per rank, `hidden=2048`, `num_experts=128`, `num_topk=2`
- Result:
  - The launcher-side packaging blocker is gone; the pod now downloads the branch snapshot directly and reaches benchmark execution.
  - The `CUDAExtension` transport build now succeeds under JAX `0.8.0` after fixing bf16 FFI pointer casts in `deepep_transport_ffi.cu`.
  - The host-only control reaches a real DeepEP H100 dispatch launch and logs:
    - `attr_status_code=0`, `set_attr_status_code=0`
    - `binary_version=90`, `ptx_version=90`
    - `requested_smem_bytes=196608`
  - After launch, the host-only control still hangs in distributed progress:
    - repeated `DeepEP timeout for dispatch receivers ... tokens remained: ...`
    - repeated `DeepEP timeout for dispatch senders ...`
    - final `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
- Interpretation:
  - This is the cleanest evidence so far that the remaining blocker is below the JAX execution model.
  - The failure now reproduces in a same-process, host-only, all-ranks dispatch round using the same singleton runtime manager, before any `jax.jit`, `shard_map`, or `pmap` transport step runs.
  - So the frontier has moved again:
    - no longer raw symbol resolution
    - no longer just “maybe XLA stream / custom-call semantics”
    - now most likely same-process DeepEP runtime assumptions, rendezvous/progress ordering, or how our runtime manager drives those kernels
- Next action:
  - Shrink the host-only control toward the smallest deterministic case (`topk=1`, `num_experts=8`, smaller token count) to test whether the current hang is shape-independent or triggered by the larger sealed shape.
  - Instrument the host-only round more precisely around `notify_dispatch`, `WaitForRecvCounts`, and `dispatch` completion so we can see whether one rank or channel is the first to stop making progress.
  - Compare the same-process host-only control against a process-per-rank control if the minimal same-process case still hangs.
