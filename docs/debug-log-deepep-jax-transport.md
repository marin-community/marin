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
