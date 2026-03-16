# Debugging log for deepep-jax-transport

Pure-JAX DeepEP intranode dispatch/combine bring-up on CoreWeave H100x8.

## Initial status

The earlier root-cause thread established that `deepep_layout_ragged_a2a` in JAX never exercised the winning DeepEP transport kernels, while Torch/Megatron did. This branch is testing a fully Torch-free path:

- JAX/XLA custom call for DeepEP intranode `dispatch` / `combine`
- same fixed-shape H100x8 routed-token regime used in `#3633`
- no per-step JAX -> Torch bridge

The current failures have already moved below the original JAX integration layer:

- initial wrapper/API drift was fixed
- raw `nvcc` compile drift on Hopper was fixed
- same-process CUDA IPC misuse was fixed by switching to peer access/UVA
- current blockers are now:
  - raw-build H100 launch failure at `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`
  - fallback `DISABLE_SM90_FEATURES=1` launch failure with `named symbol not found`
  - extension-style build load failure with undefined `__cudaRegisterLinkedBinary...`

## Hypothesis 1

The remaining failures are not primarily JAX semantic bugs anymore; they are in DeepEP/CUDA compatibility or build/link mechanics.

## Changes to make

- Keep the current pure-JAX FFI/runtime wiring in place.
- Add a same-image Torch DeepEP control using the exact same container image and DeepEP ref as the JAX transport harness.
- Use that control to answer a narrower question:
  - does unmodified Torch-side DeepEP dispatch/combine still work in this exact H100/CUDA 12.8 pod and pinned DeepEP ref?

## Future Work

- [ ] Add explicit same-image Torch control results for the pinned DeepEP ref
- [ ] Compare raw `nvcc` build flags against DeepEP `CUDAExtension` device-link behavior
- [ ] Add kernel-launch instrumentation for shared-memory attribute and function attributes
- [ ] Decide whether `--cudart=shared` or explicit device-link is required in the raw build path

## Results

- Upstream DeepEP `setup.py` confirms that the normal working path is a PyTorch `CUDAExtension` build with RDC and linker-managed CUDA registration, not a minimal `nvcc -shared` command.
- NVIDIA guidance confirms the original CUDA IPC usage was wrong for the standard single-process JAX multi-GPU model, and the peer-access/UVA rewrite was the correct fix for that layer.
- Same-image Torch controls split two causes apart:
  - old pinned ref `567632dd...` fails at import with `undefined symbol: __cudaRegisterLinkedBinary...`
  - `refs/heads/hybrid-ep` succeeds in the same image and runs the fixed-shape torch dispatch/combine slice (`65.23M tokens/s`)
- Upstream `setup.py` explains that contrast:
  - older ref enables `-rdc=true` without forcing a device-link step in the non-NVSHMEM H100 path
  - newer `hybrid-ep` adds the missing `-dlink` guard
- Conclusion from this hypothesis:
  - the old pinned ref had a real build/device-link defect in this environment
  - but that is not the only blocker for the pure-JAX path

## Hypothesis 2

The raw `nvcc` and extension-style build failures may still collapse to one shared root cause: our custom-call shared-library build is not yet reproducing DeepEP's expected CUDA device-link/runtime-registration model.

## Changes to make

- After the same-image Torch control, compare the successful and failing build paths directly.
- If Torch control passes, treat the remaining gap as build/link plus JAX-runtime integration work.
- If Torch control fails in the same environment, treat the pod/image/toolchain or pinned DeepEP ref as a first-class blocker.

## Future Work

- [ ] Capture exact same-image Torch control logs in the research logbook
- [ ] Search for public reports of `__cudaRegisterLinkedBinary` and H100 `MaxDynamicSharedMemorySize` failures matching this build mode
- [ ] Decide whether to continue with the pure-JAX path or stop with a stronger root-cause write-up

## Results

- The pure-JAX wrapper now compiles against `refs/heads/hybrid-ep` after a small API-compatibility shim for the widened `intranode::dispatch(...)` signature.
- On the newer branch, the pure-JAX path reaches the same substantive runtime failures seen before:
  - baseline SM90 path: `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`
  - `DISABLE_SM90_FEATURES=1`: `named symbol not found`
- That means the deeper H100 launch/runtime incompatibility survives even after the old-ref build defect is removed.

## Hypothesis 3

The remaining blocker is in DeepEP intranode launch assumptions under this pure-JAX execution/build mode, not in old-ref drift or wrapper API mismatch.

## Changes to make

- Stop treating older-ref compatibility as the main problem.
- If continuing, instrument the actual H100 launch path:
  - requested dynamic shared memory size
  - function attributes for the selected kernel
  - exact launch path chosen under baseline vs `DISABLE_SM90_FEATURES=1`

## Future Work

- [ ] Log `cudaFuncGetAttributes` and the requested `smem_size` at the failing intranode launch site
- [ ] Check whether the pure-JAX build mode is selecting a different kernel variant than the working Torch path
- [ ] Decide whether the remaining issue is fixable locally or should be written up as an upstream/kernel-level blocker

## Results

- Current state:
  - old ref broken by device-link/registration
  - new branch buildable and runnable from pure JAX
  - both branches converge to the same substantive H100 launch blocker family once the old-ref defect is removed

## Hypothesis 4

The remaining mismatch might still be above the actual DeepEP launch site: JAX and Torch could be constructing different inputs, layouts, or runtime configs before calling dispatch.

## Changes to make

- Add a deterministic `--probe-only` mode to the Torch and JAX dispatch benches.
- Print a structured `PROBE_JSON` payload before entering dispatch/combine:
  - local inputs (`x`, `topk_idx`, `topk_weights`)
  - layout outputs (`num_tokens_per_rank`, `num_tokens_per_expert`, `is_token_in_rank`)
  - runtime dispatch/combine configs
- Run the same tiny H100x8 `tokens=16 hidden=8 experts=16 topk=2` case against `refs/heads/hybrid-ep` for both frameworks.

## Future Work

- [ ] Instrument the actual intranode launch site with kernel attributes and requested `smem_size`
- [ ] Compare the selected kernel variant in the working Torch path vs the failing pure-JAX path
- [ ] Decide whether the next fix belongs in Marin's JAX custom call glue or upstream DeepEP launch/runtime code

## Results

- Deterministic interface probes completed successfully for both:
  - Torch: `deepep-interface-probe-torch-rerun-20260314-2141`
  - JAX: `deepep-interface-probe-jax-20260314-2110`
- Same-image common payload fields matched at rank 0:
  - identical `x` values and `bfloat16 [2, 8]` shape/strides
  - identical `topk_idx=[[0,1],[1,2]]`
  - identical `topk_weights=[[0.6666667, 0.33333334], ...]`
  - identical `num_tokens_per_rank=[2,1,0,0,0,0,0,0]`
  - identical `num_tokens_per_expert=[1,2,1,0,...]`
  - identical `is_token_in_rank` membership matrix
  - identical runtime configs:
    - dispatch: `num_sms=20 num_max_send_tokens=6 num_max_recv_tokens=256`
    - combine: `num_sms=20 num_max_send_tokens=4 num_max_recv_tokens=256`
- The remaining differences were instrumentation-only, not semantic:
  - Torch prints framework-prefixed dtype names like `torch.int64`; JAX prints normalized names like `int64`
  - Torch probe includes extra bookkeeping fields (`bridge_to_torch_s`, `local_rank`, `local_world_size`, `num_nvl_bytes`, `num_rdma_bytes`)
- Conclusion:
  - the pure-JAX path is not misrouting tokens or constructing the wrong layout/config at the high-level DeepEP boundary
  - the remaining blocker sits below this boundary, in the actual kernel-launch/runtime layer

## Hypothesis 5

The remaining divergence may already be visible at CUDA kernel symbol resolution itself: Torch and pure JAX might be reaching the same dispatch host function but not the same usable compiled kernel symbol / function metadata.

## Changes to make

- Add an env-gated DeepEP source patch that logs dispatch launch metadata from inside `csrc/kernels/intranode.cu`:
  - selected dispatch specialization (`num_ranks`, `num_threads`, `num_tma_bytes_per_warp`)
  - requested dynamic shared memory
  - `cudaFuncGetAttributes(...)`
  - a probe `cudaFuncSetAttribute(...)`
- Stage that patch into both CoreWeave launchers:
  - `.agents/scripts/deepep_dispatch_krt_bench.py`
  - `.agents/scripts/deepep_jax_transport_krt.py`
- Run the same H100x8 `tokens=32768 hidden=2048 experts=128 topk=2 distribution=random warmup=0 iters=1` case for both:
  - Torch control: `deepep-launch-debug-torch-20260314-2230`
  - pure JAX raw FFI build: `deepep-launch-debug-jax-20260314-2230`

## Future Work

- [ ] Test whether explicit CUDA device-link / runtime registration in the raw JAX FFI build makes `cudaFuncGetAttributes(...)` succeed
- [ ] Revisit the extension-style build path with a cleaner load/registration strategy now that the failure domain is narrower
- [ ] Only after symbol resolution works, compare stream/launch behavior again

## Results

- Torch control on the patched DeepEP source logged a healthy dispatch kernel on rank 0:
  - `attr_status_code=0`, `set_attr_status_code=0`
  - `binary_version=90`, `ptx_version=90`
  - `requested_smem_bytes=196608`
  - `num_sms=20`, `num_threads=768`, `cluster_dim_x=2`
  - Example debug line also showed real function attributes (`num_regs=80`, `max_threads_per_block=768`)
- The pure-JAX raw FFI build reached the same host dispatch site on the same shape, but the kernel symbol was not usable:
  - `attr_status_code=500`, `attr_status="named symbol not found"`
  - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
  - all returned function-attribute fields were zero (`binary_version=0`, `ptx_version=0`, `max_dynamic_smem_bytes=0`, ...)
  - the later assertion at `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)` is therefore downstream of the earlier missing-symbol condition
- Both launch debug lines agreed on the host-side launch intent:
  - `num_tokens=4096`
  - `hidden_int4=256`
  - `num_topk=2`
  - `num_experts=128`
  - `num_max_send_tokens=6`
  - `num_recv_buffer_tokens=256`
  - `requested_smem_bytes=196608`
- Conclusion:
  - this is no longer best described as a vague runtime incompatibility
  - the immediate failure in the pure-JAX path is that the raw custom-call `.so` does not expose a resolvable CUDA dispatch kernel symbol with valid function attributes, while the Torch build does
  - that points back to build / device-link / runtime-registration mechanics in the pure-JAX FFI shared library as the current primary blocker

## Hypothesis 6

The earlier `sm_52` `nvlink` warning from the first raw `-dlink` attempt was a real bug in our final raw link command, but only a secondary one. After fixing that and forcing a fresh rebuild, the core failure may still remain unchanged, which would imply the real blocker is deeper in raw CUDA device registration / fatbin packaging.

## Changes to make

- Update `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py` so the raw build path:
  - compiles each CUDA source to an object with RDC enabled
  - runs an explicit `nvcc -dlink`
  - performs the final `nvcc -shared` link with the explicit `_cuda_arch_flag()`
- Add the Python builder source plus a cache schema version to the shared-library hash so build-pipeline changes invalidate the cached `.so`.
- Rerun the H100x8 launch-debug JAX job on `refs/heads/hybrid-ep` and check both:
  - whether the `sm_52` warning disappears
  - whether `cudaFuncGetAttributes(...)` starts returning real function metadata

## Future Work

- [ ] Inspect the rebuilt raw `.so` and its device-link object against a working Torch-extension build
- [ ] Check whether the raw artifact is missing fatbin / registration sections even after the corrected `-dlink` flow
- [ ] Search for upstream CUDA guidance or reports specific to `cudaErrorSharedObjectSymbolNotFound` in separable-compilation shared libraries

## Results

- The corrected raw build rerun completed on H100x8:
  - task: `deepep-launch-debug-jax-dlink-20260314-2358`
- The stale final-link arch bug was real and is now fixed:
  - the rerun no longer emitted `nvlink warning : SM Arch ('sm_52') not found ...`
- But the primary failure did not move:
  - DeepEP launch debug still reported:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - `requested_smem_bytes=196608`
    - all function-attribute fields still zero
  - JAX still failed later at the downstream DeepEP assertion on `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`
- Conclusion:
  - fixing the final-link arch selection removed one genuine raw-build defect, but it did not solve the pure-JAX transport blocker
  - the remaining failure stays centered on raw CUDA device registration / fatbin packaging for the custom-call `.so`, not on stale caching or wrong launch intent

## Hypothesis 7

The raw custom-call `.so` may actually be structurally fine on disk, and the remaining failure may instead come from how the loaded library interacts with the JAX/XLA execution path at runtime. If so, a binary inspection should show normal cubins and registration symbols, and changing the `dlopen` scope to `RTLD_GLOBAL` still may not fix the failing `cudaFuncGetAttributes(...)`.

## Changes to make

- Add a CoreWeave build-inspection helper that:
  - locates the raw custom-call `.so`
  - inspects its dynamic deps, sections, symbols, and embedded cubins
  - inspects the installed upstream `deep_ep_cpp` artifact in the same image
- Change the raw library loader from `ctypes.cdll.LoadLibrary(...)` to `ctypes.CDLL(..., RTLD_NOW | RTLD_GLOBAL)`.
- Rerun the H100x8 launch-debug job on the same `tokens=32768 hidden=2048 experts=128 topk=2` case.

## Future Work

- [ ] Add an exported host-side probe that calls `cudaFuncGetAttributes(...)` immediately after library load / runtime init, before entering XLA custom-call execution
- [ ] Check whether the same probe succeeds when the raw library is loaded and called from a plain Python host path outside JAX execution
- [ ] If the host probe succeeds but the custom call still fails, focus on XLA stream / execution-context interaction rather than build packaging

## Results

- Artifact inspection on H100 showed the raw custom-call library is not obviously malformed:
  - `.nv_fatbin` is present in the final `.so`
  - `cuobjdump --list-elf` reports two embedded `sm_90` cubins
  - the final `.so` exports the expected `__cudaRegisterLinkedBinary_*`, `__fatbinwrap_*`, and DeepEP intranode dispatch/combine symbols
  - the explicit raw `deepep_transport_ffi.dlink.o` also contains `.nv_fatbin` and its own `sm_90` cubin
- The installed upstream `deep_ep_cpp` artifact is also structurally healthy in the same image:
  - `.nv_fatbin`
  - one embedded `sm_90` cubin
  - the expected `__cudaRegisterLinkedBinary_*`, `__fatbinwrap_*`, and `deep_ep::intranode::*` symbols
- The loader-scope experiment did not change the JAX launch result:
  - after switching to `ctypes.CDLL(..., RTLD_NOW | RTLD_GLOBAL)`, launch debug still reported:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - all function-attribute fields zero
- Conclusion:
  - the current blocker is not just “the raw custom-call `.so` is missing cubins / registration symbols”
  - it is also not explained by `ctypes` loading the library too locally
  - the remaining failure is now best framed as a runtime interaction problem between the otherwise-populated raw custom-call library and the JAX/XLA execution path that reaches the DeepEP intranode dispatch kernel

## Hypothesis 8

If the same DeepEP launch debug failure reproduces from a plain host-side call that never enters `jax.jit` / XLA execution, then the remaining bug is not XLA-specific after all. It would instead live in the raw library load / runtime-init path itself.

## Changes to make

- Add an exported host-side probe to the raw custom-call library that:
  - initializes the existing intranode runtime
  - allocates dummy device buffers matching the sealed `#3633` local shape
  - calls `deep_ep::intranode::dispatch(...)` directly on `runtime.aux_stream`
- Add a Python wrapper plus a `--host-kernel-probe-only` path in the JAX bench/launcher so the probe can run on H100x8 without entering the XLA custom call.
- Run that probe with launch-debug enabled.

## Future Work

- [ ] Compare the raw library load/init path against a CPython extension-module load path even more directly
- [ ] Probe immediately after library load but before intranode runtime init to see whether init timing matters
- [ ] If needed, test whether loading the same raw `.so` via a Python extension module instead of `ctypes` changes CUDA symbol resolution

## Results

- The host-side probe reproduced the same DeepEP dispatch launch-debug line before any XLA execution:
  - `label="host_probe"`
  - `attr_status_code=500`, `attr_status="named symbol not found"`
  - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
  - `requested_smem_bytes=196608`
  - all function-attribute fields zero
- The probe then failed with the same downstream DeepEP assertion on `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`.
- Conclusion:
  - this rules out the remaining “maybe it is only XLA/custom-call execution context” explanation
  - the failure is already present in the raw shared-library path before XLA executes anything
  - combined with the earlier artifact inspection, the remaining blocker is now best framed as a raw library load / runtime-registration problem in-process, not an XLA-specific execution bug and not a gross packaging omission

## Hypothesis 9

Loading the same raw code as a CPython extension module instead of bare `ctypes` may change CUDA registration behavior enough to resolve the missing-symbol failure.

## Changes to make

- Add a minimal CPython module shim source:
  - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_pyext.cc`
- Add an opt-in `DEEPEP_LOAD_AS_PYTHON_MODULE=1` mode in:
  - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
- Re-run the H100x8 host probe with:
  - raw build + Python extension load

## Future Work

- [ ] If raw Python-extension load still fails, try a closer-to-upstream build frontend rather than another loader tweak
- [ ] Keep the raw and extension load paths both available until the failure domain is clear

## Results

- Raw Python-extension loading did not change the failing launch debug line:
  - `attr_status_code=500`, `attr_status="named symbol not found"`
  - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
- Conclusion:
  - loader semantics alone are not sufficient
  - the remaining divergence still depends on build/link behavior, not just `ctypes` vs extension import

## Hypothesis 10

A closer-to-upstream `CUDAExtension` build plus Python-extension import may clear the kernel-symbol-resolution blocker even if the raw build cannot.

## Changes to make

- Replace the experimental `cpp_extension.load(..., is_python_module=True)` path with a generated `setup.py` that builds the transport wrapper through:
  - `torch.utils.cpp_extension.CUDAExtension`
  - explicit RDC compile flags
  - explicit `nvcc` device-link flags
- Preload Torch shared libraries before importing the built extension module.
- Re-run:
  - host probe
  - full `shard_map` JAX path
  - full `pmap` JAX path

## Future Work

- [ ] Move peer-access setup later so JAX input placement can finish before runtime init
- [ ] Add a same-process multi-rank host control to separate “JAX scheduling” from “DeepEP same-process progress”
- [ ] If needed, strip the extension build down further after behavior is understood

## Results

- The generated `CUDAExtension` path finally changed the core signal:
  - the host probe now logs:
    - `attr_status_code=0`, `attr_status="no error"`
    - `set_attr_status_code=0`, `set_attr_status="no error"`
    - real H100 function attributes (`binary_version=90`, `ptx_version=90`, `num_regs=80`, `max_threads_per_block=768`)
- This means the earlier kernel-symbol-resolution blocker is no longer fundamental; it was specific to the raw shared-library build/load path.
- The full `shard_map` JAX path now fails later, during live transport:
  - repeated DeepEP receiver timeouts
  - `cudaMemcpyAsync(read num_recv_tokens): unspecified launch failure`
- The `pmap` path fails even earlier, during JAX input placement:
  - `cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- Conclusion:
  - the current frontier has moved
  - build/load mechanics are no longer the only blocker
  - the remaining problems are:
    - real multi-GPU progress / rendezvous under `shard_map`
    - peer-access-state ownership / ordering under `pmap`

## Hypothesis 11

If a host-only all-ranks dispatch round still hangs after the `CUDAExtension` / Python-extension path clears kernel symbol resolution, then the remaining failure is not specific to `jax.jit`, `shard_map`, or `pmap`.

## Changes to make

- Add a host-only exported control to `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`:
  - initialize once through the existing singleton `RuntimeManager`
  - spawn one host thread per GPU/rank
  - on each thread, allocate dummy device buffers, build deterministic routing metadata, and call the real `notify_dispatch + WaitForRecvCounts + dispatch` path
- Wire that symbol through:
  - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
  - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
  - `.agents/scripts/deepep_jax_transport_krt.py`
- Run it on the isolated CoreWeave H100x8 lane with:
  - `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`
  - `DEEPEP_LOAD_AS_PYTHON_MODULE=1`
  - launch debug enabled

## Future Work

- [ ] Reduce the host-only control to the smallest deterministic case (`topk=1`, `num_experts=8`, smaller token count)
- [ ] Add per-rank host logs around `notify_dispatch`, `WaitForRecvCounts`, and `dispatch` completion
- [ ] Test whether a process-per-rank control succeeds where the current same-process control fails

## Results

- First, the launcher surfaced a separate operational bug:
  - the previous pod launch path failed with `exec /usr/bin/bash: argument list too long`
  - root cause: inlining base64-staged source files into `bash -lc`
  - fix: commit/push the runnable experiment code and make the launcher fetch the branch snapshot directly instead of staging large local files into the command line
- Second, once the pod reached execution, the host-only control exposed a JAX `0.8.0` bring-up mismatch:
  - `ffi::Buffer<ffi::BF16>::typed_data()` now materializes as `unsigned short*`
  - our transport wrapper still passed those pointers into `DispatchOnCurrentDevice(...)` as `nv_bfloat16*` without an explicit cast
  - fixing those casts in `deepep_transport_ffi.cu` was enough to make the `CUDAExtension` transport build succeed on H100
- Third, the actual host-only dispatch control reached a real H100 launch:
  - `DEEPEP_LAUNCH_DEBUG` logged:
    - `attr_status_code=0`, `attr_status="no error"`
    - `set_attr_status_code=0`, `set_attr_status="no error"`
    - `binary_version=90`, `ptx_version=90`
    - `requested_smem_bytes=196608`
- But the host-only control still failed after launch:
  - repeated `DeepEP timeout for dispatch receivers ... tokens remained: ...`
  - repeated `DeepEP timeout for dispatch senders ...`
  - final `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
- Conclusion:
  - this removes the last strong “maybe this is still a JAX/XLA execution-context bug” explanation
  - with the extension-style build, the kernel symbol is valid and launch attributes are healthy
  - the remaining failure reproduces in a same-process, host-only, all-ranks dispatch round using the singleton runtime manager
  - so the new fault domain is most likely:
    - same-process DeepEP runtime assumptions,
    - rendezvous/progress ordering,
    - or runtime-manager ownership of the multi-rank dispatch lifecycle

## Hypothesis 12

If the same-process host-only control still hangs on a tiny deterministic case, then the bug is not specific to the larger sealed `#3633` payload shape.

## Changes to make

- Reuse the existing host-only control with a much smaller shape:
  - `tokens=128` globally (`16` per rank)
  - `hidden=128`
  - `num_experts=8`
  - `topk=1`
- Keep the same:
  - isolated CoreWeave H100x8 lane
  - `CUDAExtension` / Python-extension load path
  - launch-debug instrumentation

## Future Work

- [ ] Add stage-by-stage host logs for each rank around `notify_dispatch`, `WaitForRecvCounts`, and `dispatch`
- [ ] Check whether channel `0` is consistently the outlier on the tiny case
- [ ] Decide between a process-per-rank control and a stricter same-process debug mode as the next fork

## Results

- The tiny deterministic control still reaches a healthy H100 dispatch launch:
  - `attr_status_code=0`, `set_attr_status_code=0`
  - `binary_version=90`, `ptx_version=90`
- It still hangs after launch:
  - repeated receiver timeout lines with `tokens remained: 1`
  - several ranks also show `tokens remained: 11` on channel `0`
  - final surfaced failure:
    - `rank 2: cudaStreamSynchronize(dispatch): unspecified launch failure`
- Conclusion:
  - the post-launch hang is not a large-shape artifact
  - the same-process failure survives even on the smallest deterministic case tried so far
  - that strengthens the case that the remaining bug lives in same-process runtime/progress behavior rather than in specific routing payloads or the larger benchmark shape

## Hypothesis 13

The shared dispatch helper is still imposing the wrong host-side synchronization discipline: it waits on `cudaStreamSynchronize(dispatch)` inside `DispatchOnCurrentDevice(...)` even though the receive-count metadata is already ready before dispatch launch.

## Changes to make

- Refactor `DispatchOnCurrentDevice(...)` in `deepep_transport_ffi.cu` so that:
  - `local_expert_counts` is copied asynchronously after `WaitForRecvCounts(...)`
  - the helper can skip `cudaStreamSynchronize(dispatch)` after launch
- Update the host-only control to:
  - enqueue dispatch on all ranks
  - wait at a second host barrier
  - only then synchronize each per-rank stream

## Future Work

- [ ] Re-run the tiny deterministic host-only control with the same no-eager-sync path and higher `dispatch_num_max_send_tokens`
- [ ] Carry the no-eager-sync discipline into the real JAX `shard_map` path once the host-only control is cleaner
- [ ] If needed, split `dispatch` and completion/readback into distinct debug helpers instead of one shared wrapper

## Results

- Commit `2994b3d15d6dfc7ebbd76a1bf303a119050c800a` applies the no-eager-sync refactor.
- On the default tiny deterministic host-only control (`tokens=128 hidden=128 experts=8 topk=1`, `dispatch_num_sms=20`):
  - all `8/8` ranks now reach `after_dispatch_launch`
  - all `8/8` ranks also reach the new `after_launch_sync_barrier`
  - only after that do they start timing out and fail in `cudaStreamSynchronize(dispatch)`
- This is a real behavior change:
  - the earlier failure shape looked like ranks were collapsing while launch was still being entered
  - the new failure shape is clearly inside live dispatch completion after all ranks have already launched
- On the same tiny control with `dispatch_num_sms=2`:
  - all `8/8` ranks again reach `after_dispatch_launch`
  - all `8/8` ranks reach `after_launch_sync_barrier`
  - rank `0` is now the first rank to fully reach:
    - `after_dispatch_sync`
    - `thread_done`
  - the remaining failing ranks still die in `cudaStreamSynchronize(dispatch)`
  - their timeout lines are now concentrated on `responsible_channel = 0`
  - the stuck token counts are `8` or `16`, which is larger than the default `dispatch_num_max_send_tokens=6`
- Conclusion:
  - the eager per-rank host sync in the shared helper was part of the problem, but not the whole problem
  - the remaining same-process host-only bug now looks even more like a live channel/progress/configuration problem than a generic pre-launch or XLA-ordering problem

## Hypothesis 14

The real JAX `shard_map` path should succeed on the tiny deterministic shape once the wrapper stops corrupting `notify_dispatch(...)`, returns the receive-side channel-prefix handle expected by combine, and actually propagates the debug dispatch config into the JAX bench.

## Changes to make

- Fix the `notify_dispatch(...)` wrapper bug:
  - pass `runtime.dispatch_num_channels()` instead of `runtime.dispatch_config.num_sms`
- Propagate CLI dispatch/combine overrides through `bench_deepep_dispatch_jax.py`
- Return `recv_channel_prefix_matrix` from the dispatch FFI and pass it into combine
- Repair the benchmark-side correctness check to compare host arrays rather than subtract differently sharded JAX arrays

## Future Work

- [ ] Explain the post-result hang in the JAX benchmark process
- [ ] Re-run the tiny deterministic case without launch-debug spam once teardown is understood
- [ ] Scale to a slightly larger shape before jumping back to the sealed `#3633` regime

## Results

- The tiny real `shard_map` rerun on the isolated H100x8 lane completed the transport step successfully:
  - task: `deepep-jax-transport-krt-20260315-184150`
  - pod: `iris-task-03e0519dbd58`
  - repo ref: `2b86a3f6448481c600d311be42ae0b0c8f014a3d`
- Launch debug on the real JAX path was healthy:
  - `attr_status_code=0`
  - `set_attr_status_code=0`
  - `num_sms=2`
  - `num_max_send_tokens=32`
  - `max_dynamic_smem_bytes=196608`
- All ranks reached the expected live-dispatch stages and the benchmark returned to Python cleanly enough to run the correctness check.
- The correctness check passed exactly:
  - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
- The benchmark produced the first clean pure-JAX timing line:
  - `RESULT step_s=0.000480 tokens_per_s=266519.71`
- The remaining issue has shifted again:
  - after printing the result, the Python benchmark process stayed alive inside the pod instead of exiting
  - `ps` inside the live container still showed `/opt/conda/bin/python lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py ... --check`

This means the current blocker is no longer “transport fails to run.” It is now a smaller teardown/lifecycle issue after a successful pure-JAX transport step.
  - the next concrete hypothesis is that the reduced-channel deterministic case is also underprovisioned by the default `dispatch_num_max_send_tokens` cap

## Hypothesis 15

The apparent post-result "hang" is not a transport-execution failure anymore. The pure-JAX path should also complete on medium and full sealed-shape runs, with the remaining problem confined to teardown noise after a successful result.

## Changes to make

- Re-run the real pure-JAX transport path on:
  - a medium random case (`tokens=1024 hidden=2048 experts=128 topk=2`)
  - the sealed-shape random case (`tokens=32768 hidden=2048 experts=128 topk=2`)
- Keep the same working bring-up settings:
  - `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`
  - `DEEPEP_LOAD_AS_PYTHON_MODULE=1`
  - `dispatch_num_sms=2`
  - larger send/recv token caps
- Watch for:
  - exact correctness-check lines
  - timing lines
  - final pod exit code
  - whether the late CUDA/XLA error flood still appears after a successful result

## Future Work

- [ ] Compare the now-working pure-JAX transport path directly against the Torch/Megatron-style transport microbench on the same sealed shape
- [ ] Reduce teardown noise to a minimal reproducer once performance comparisons are captured
- [ ] Determine whether the cleanup failures come from outstanding DeepEP work, XLA device shutdown order, or both

## Results

- Medium random pure-JAX case:
  - command:
    - `tokens=1024 hidden=2048 experts=128 topk=2 distribution=random`
    - `dispatch_num_sms=2`
    - all send/recv caps set to `256`
    - `warmup=0`, `iters=1`
  - result:
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.000552 tokens_per_s=1855976.92`
    - `EXIT_CODE=0`
- Full sealed-shape random pure-JAX case:
  - command:
    - `tokens=32768 hidden=2048 experts=128 topk=2 distribution=random`
    - `dispatch_num_sms=2`
    - all send/recv caps set to `8192`
    - `warmup=0`, `iters=1`
  - result:
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.005023 tokens_per_s=6523372.00`
    - `EXIT_CODE=0`
- Both runs still end with a large late error flood:
  - `DeepEP timeout check failed: rank = ...`
  - `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
  - XLA/JAX CUDA stream, event, module-unload, and memory-free errors during shutdown

## Conclusion

- The remaining blocker has moved again.
- It is no longer accurate to describe the pure-JAX path as "still hanging before a usable run."
- The pure-JAX DeepEP transport step now executes correctly on:
  - the tiny deterministic smoke
  - a medium random case
  - the full `#3633` token regime
- The surviving issue is a teardown / device-shutdown problem after a successful transport step, not a front-half correctness or launch blocker.
- That is enough progress to move back to the original goal: direct JAX-vs-Torch transport comparison on the sealed shape, while treating cleanup noise as a separate follow-up bug.

## Hypothesis 16

The earlier `num_sms=20` crash was at least partly caused by using oversized debug send/recv caps instead of the actual DeepEP world-size `8` defaults.

## Changes to make

- Re-run the sealed `#3633`-shape pure-JAX transport cell with the same config family the Torch transport microbench uses for world-size `8`:
  - dispatch: `num_sms=20`, `num_max_send_tokens=6`, `num_max_recv_tokens=256`
  - combine: `num_sms=20`, `num_max_send_tokens=4`, `num_max_recv_tokens=256`
- Keep the same now-working extension-style bring-up:
  - `DEEPEP_BUILD_WITH_TORCH_EXTENSION=1`
  - `DEEPEP_LOAD_AS_PYTHON_MODULE=1`
- Use the same authoritative sealed-shape comparison cell:
  - `tokens=32768 hidden=2048 experts=128 topk=2 distribution=random`

## Future Work

- [ ] Verify whether the wrapper defaults (without explicit CLI overrides) reproduce the same `20`-SM result
- [ ] Expand the head-to-head beyond `random, topk=2` once the `20`-SM path is confirmed stable
- [ ] Revisit the earlier `8192/8192` crash only if it still matters after the apples-to-apples comparisons are complete

## Results

- The Torch-matched `20`-SM config works on the pure-JAX sealed-shape cell.
- Command family:
  - `uv run .agents/scripts/deepep_jax_transport_krt.py ... --dispatch-num-sms 20 --dispatch-num-max-send-tokens 6 --dispatch-num-max-recv-tokens 256 --combine-num-sms 20 --combine-num-max-send-tokens 4 --combine-num-max-recv-tokens 256 --warmup 1 --iters 3`
- Authoritative result:
  - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
  - `RESULT step_s=0.000777 tokens_per_s=42173236.13`
  - pod exit: `0 Completed`
- The same late teardown/XLA CUDA shutdown noise still appears after the result line, but it no longer changes the outcome of the actual benchmark cell.
- Most importantly, this overturns the earlier interpretation of the `20`-SM frontier:
  - `num_sms=20` is not inherently broken on the JAX path
  - the earlier failing `20`-SM run used the oversized debug caps (`8192/8192`)
  - the clean Torch-matched config runs correctly and is much faster than the earlier debug settings
- Updated sealed-shape transport ladder:
  - JAX `num_sms=2`: `6.64M tokens/s`
  - JAX `num_sms=4`: `12.51M tokens/s`
  - JAX `num_sms=8`: `21.95M tokens/s`
  - JAX `num_sms=20` with Torch-matched defaults: `42.17M tokens/s`
  - Torch transport baseline: `64.89M tokens/s`

This means the live question is no longer “why does JAX fall over at `20` SMs?” The live question is now “what explains the remaining ~`1.54x` gap once JAX also uses the real DeepEP-style `20`-SM transport config?”

## Hypothesis 17

The corrected pure-JAX transport path is now broadly competitive across the same-shape four-cell matrix; the remaining issues are bounded to small residual overhead plus a top-`8` bf16 reconstruction drift and the known teardown noise.

## Changes to make

- Run the remaining same-shape cells one pod at a time under the wrapper defaults:
  - `random, topk=2`
  - `runs, topk=2`
  - `random, topk=8`
  - `runs, topk=8`
- Compare them directly against the existing Torch transport matrix already captured in the thread.
- Treat the post-result teardown hang as an execution nuisance, not as a reason to avoid capturing the single-cell results.

## Future Work

- [ ] Decide whether the consistent `topk=8` `x_max_abs=1.785707e-02` drift is an acceptable bf16 accumulation artifact or deserves a targeted follow-up
- [ ] Determine whether to seal the experiment at the transport-root-cause layer or keep it open for teardown cleanup
- [ ] If continuing, isolate the `topk=8` drift with a smaller deterministic reproducer

## Results

- Single-cell pure-JAX results under the corrected default `20`-SM regime:
  - `random, topk=2`
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.000748 tokens_per_s=43787427.16`
  - `runs, topk=2`
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.001124 tokens_per_s=29145790.33`
  - `random, topk=8`
    - `CHECK x_max_abs=1.785707e-02 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.001298 tokens_per_s=25254084.80`
  - `runs, topk=8`
    - `CHECK x_max_abs=1.785707e-02 topk_max_abs=0.000000e+00`
    - `RESULT step_s=0.001429 tokens_per_s=22935953.20`
- Direct comparison to the existing Torch transport matrix on the same shape:
  - `random, topk=2`: Torch/JAX `1.54x`
  - `runs, topk=2`: Torch/JAX `1.23x`
  - `random, topk=8`: Torch/JAX `1.22x`
  - `runs, topk=8`: Torch/JAX `1.10x`

This is now a fundamentally different state from the earlier debug-config comparison. The pure-JAX transport path is no longer “far behind Torch” in a vague sense. On the corrected same-shape matrix, it is within about `10%` to `54%` depending on the cell, with the tightest gaps on the `topk=8` transport-heavy cells.

## Hypothesis 18

The `topk=8` `x_max_abs=1.785707e-02` drift is likely consistent with bf16 fan-in noise rather than a transport-specific corruption bug.

## Changes to make

- Run a tiny local sanity check outside the transport path:
  - create a random bf16 tensor
  - sum it with itself `2` and `8` times
  - divide back down in float32
  - compare against the original bf16 tensor in the same way the benchmark’s `CHECK` logic does

## Future Work

- [ ] If someone still cares about the `topk=8` drift after sealing, build a smaller deterministic GPU reproducer to check whether the exact transport path and naive bf16 fan-in align more closely
- [ ] Split teardown noise into its own follow-up if needed

## Results

- Local CPU/JAX bf16 fan-in sanity check:
  - `err2=0.000000000`
  - `err8=0.031250000`
- The observed transport-side `topk=8` drift is:
  - `x_max_abs=1.785707e-02`
- Interpretation:
  - the observed drift is smaller than the naive bf16 fan-in toy result but clearly in the same order of magnitude
  - that makes the transport-side `topk=8` drift much more plausibly “expected bf16 accumulation noise under larger fan-in” than “evidence of token corruption or a broken transport handle”

At this point the remaining uncertainty is too small to overturn the transport-root-cause conclusion.
