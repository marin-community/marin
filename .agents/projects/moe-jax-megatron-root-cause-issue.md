## Description

This experiment starts from three sealed GPU MoE threads that currently point in different directions:

- `#3641`: torch-side DeepEP / Hybrid-EP looked promising on H100x8
- `#3665`: the first JAX custom-call DeepEP path was runnable but did not improve the sealed `#3633` benchmark
- `#3666`: Megatron-LM's full `MoELayer` benchmark changed the ranking again and favored `deepep` or `hybridep` depending on scale

The new question is not "is DeepEP good?" in the abstract. The question is:

> what are the concrete root causes of the performance gap between the positive torch/Megatron results and the negative JAX/Marin result?

This thread is specifically about separating:

1. benchmark methodology differences
2. kernel coverage differences
3. transport / communication backend differences
4. JAX-specific overheads or missing fusions

The goal is to produce a falsifiable explanation rather than another isolated timing table.

## Hypothesis or Goal

- Hypothesis: `#3665` stayed tied with `ragged_a2a` because it only replaced JAX-side dispatch-layout metadata production, while the positive results in `#3641` / `#3666` came from exercising materially different transport and full-layer kernels.
- Hypothesis: part of the apparent gap is methodology, not just implementation. `#3665` reused the fixed-shape `#3633` dispatch benchmark, while `#3666` measured a full Megatron `MoELayer` with different timing hygiene and compute/communication balance.
- Goal: build a rigorous root-cause matrix that explains which differences matter, how much they matter, and which ones are sufficient to reproduce the torch/Megatron gains.
- Goal: include a direct head-to-head between JAX/Marin and Megatron-LM on the same CoreWeave H100x8 cluster.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior JAX custom-call issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior torch-side seal tag: https://github.com/marin-community/marin/tree/moe-deepep-hybrid-ep-seal-20260314
* Prior JAX custom-call seal tag: https://github.com/marin-community/marin/tree/moe-deepep-jax-layout-ffi-h100-matrix-20260314
* Prior Megatron seal tag: https://github.com/marin-community/marin/tree/moe-megatron-qwen-scale-h100-matrix-20260314
* Research logbook: `.agents/logbooks/moe-jax-megatron-root-cause.md`

## Results

Current state as of 2026-03-14:
- The same H100x8 shape now has three matched views:
  1. JAX/Marin fixed-shape forward-only control on the sealed `#3633` harness
  2. Megatron-LM full `MoELayer` timing on a same-shape `marin_3633_topk_{2,8}` case
  3. direct DeepEP dispatch/combine isolation with explicit JAX -> Torch and Torch -> JAX bridge timing
- There is now also a fourth layer of evidence:
  4. a pure-JAX DeepEP transport bring-up on the same H100x8 environment, including same-image Torch controls and deterministic pre-launch interface probes

Matched JAX/Marin result:
- Shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 shared_expert_dim=0`
- Timing hygiene: `warmup=5 iters=20`
- `current` still wins every distributed point (`EP > 1`)
- `current / ragged_a2a` ranges from `1.73x` to `2.48x`
- `deepep_layout_ragged_a2a` stays effectively tied with `ragged_a2a`, usually within `~1%`

Matched Megatron result on the same shape:
- `topk=2`
  - `alltoall`: `forward 14.62 ms`, `backward 19.57 ms`
  - `deepep`: `forward 7.74 ms`, `backward 6.00 ms`
  - `hybridep`: `forward 6.31 ms`, `backward 8.35 ms`
- `topk=8`
  - `alltoall`: `forward 11.26 ms`, `backward 20.86 ms`
  - `deepep`: `forward 4.84 ms`, `backward 6.15 ms`
  - `hybridep`: `forward 6.19 ms`, `backward 10.43 ms`

Direct DeepEP dispatch/combine isolation:
- Raw torch-side DeepEP transport on the sealed shape is strong:
  - `random, topk=2`: `67.44M tokens/s`
  - `random, topk=8`: `30.85M tokens/s`
  - `runs, topk=2`: `35.90M tokens/s`
  - `runs, topk=8`: `25.25M tokens/s`
- Once tensors are already in Torch, JAX-originating tensors see similar steady-state transport cost:
  - JAX/Torch `dispatch_combine_full_s` ratio ranges from `0.92x` to `1.36x`
- But the bridge itself is expensive:
  - `bridge_to_torch_s`: about `85 ms` to `105 ms`
  - `bridge_to_jax_s`: about `2 ms` to `12 ms`

Pure-JAX DeepEP transport bring-up:
- The older pinned DeepEP ref (`567632dd...`) was not a clean JAX baseline on CUDA 12.8 / H100:
  - same-image Torch control failed before any JAX code with `undefined symbol: __cudaRegisterLinkedBinary...`
- The newer `refs/heads/hybrid-ep` branch does run in the same image from Torch:
  - same-image Torch control reached `65.23M tokens/s`
- After porting the JAX wrapper to the newer branch, the pure-JAX path reaches the real intranode DeepEP launch and fails lower in the stack:
  - baseline SM90 path: `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`
  - `DISABLE_SM90_FEATURES=1`: `named symbol not found`
- A deterministic Torch-vs-JAX interface probe on `tokens=16 hidden=8 experts=16 topk=2` shows the pre-launch boundary matches on the common semantic fields:
  - same local `x`, `topk_idx`, and `topk_weights`
  - same layout outputs (`num_tokens_per_rank`, `num_tokens_per_expert`, `is_token_in_rank`)
  - same runtime dispatch/combine configs
  - remaining differences are only probe-format details like dtype string spelling and Torch-only bookkeeping fields
- Launch instrumentation on the real `tokens=32768 hidden=2048 experts=128 topk=2` shape sharpens the current blocker:
  - Torch control logs a healthy dispatch kernel:
    - `attr_status_code=0`, `set_attr_status_code=0`
    - `binary_version=90`, `ptx_version=90`
    - real H100 function attributes and `requested_smem_bytes=196608`
  - The pure-JAX raw FFI build reaches the same host dispatch call with the same host-side launch intent, but:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - all returned function-attribute fields are zero
  - So the JAX failure is already present at CUDA kernel symbol resolution / function-attribute lookup, before any successful dispatch launch
- A follow-up raw-build fix tested whether this was just a bad final link:
  - the raw build now compiles each `.cu` to an object, runs explicit `nvcc -dlink`, and links the final `.so` with the explicit H100 `-gencode`
  - the shared-library cache key now includes the Python builder logic so those build-pipeline changes force a fresh rebuild
  - that removed the earlier `nvlink warning : SM Arch ('sm_52') not found ...`
  - but the launch-debug result stayed the same:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - all function-attribute fields still zero
  - so the remaining blocker is narrower than “wrong arch flags” or “stale cached build”: it still points at raw CUDA device-registration / fatbin-packaging mechanics in the pure-JAX custom-call `.so`
- Artifact inspection and an explicit loader-scope test narrowed that again:
  - the raw custom-call `.so` is structurally populated on disk:
    - `.nv_fatbin`
    - two embedded `sm_90` cubins
    - `__cudaRegisterLinkedBinary_*`
    - `__fatbinwrap_*`
    - the expected DeepEP intranode dispatch/combine symbols
    - the explicit `deepep_transport_ffi.dlink.o` also contains `.nv_fatbin` and an `sm_90` cubin
  - the installed upstream `deep_ep_cpp` artifact in the same image is also structurally healthy:
    - `.nv_fatbin`
    - one embedded `sm_90` cubin
    - the expected `__cudaRegisterLinkedBinary_*`, `__fatbinwrap_*`, and `deep_ep::intranode::*` symbols
  - switching the raw loader from `ctypes.cdll.LoadLibrary(...)` to `ctypes.CDLL(..., RTLD_NOW | RTLD_GLOBAL)` did not change the failing JAX launch:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
  - so the remaining blocker is no longer well-described as “the raw custom-call `.so` forgot to package the DeepEP kernels” or “the library was loaded too locally”
- A host-side probe removed the last major XLA-specific hypothesis:
  - the custom-call library now exposes a host probe that initializes the intranode runtime, allocates dummy device buffers, and calls `deep_ep::intranode::dispatch(...)` directly without entering `jax.jit` / XLA execution
  - with launch-debug enabled, that host probe emits the same line:
    - `label="host_probe"`
    - `attr_status_code=500`, `attr_status="named symbol not found"`
    - `set_attr_status_code=500`, `set_attr_status="named symbol not found"`
    - `requested_smem_bytes=196608`
  - it then fails with the same downstream DeepEP assertion on `cudaFuncSetAttribute(... MaxDynamicSharedMemorySize ...)`
  - so the failure is already present before XLA custom-call execution begins
- A new extension-style bring-up changed that part of the story:
  - raw CPython-extension loading alone still did **not** help:
    - `attr_status_code=500`, `attr_status="named symbol not found"`
  - but a closer-to-upstream `CUDAExtension` build plus Python-extension import and Torch-library preload **did** clear the original kernel-symbol blocker:
    - host probe now logs:
      - `attr_status_code=0`, `attr_status="no error"`
      - `set_attr_status_code=0`, `set_attr_status="no error"`
      - `binary_version=90`, `ptx_version=90`
      - real H100 function attributes (`num_regs=80`, `max_threads_per_block=768`)
  - that means the earlier missing-symbol failure was specific to the raw pure-JAX shared-library build/load path, not an unavoidable JAX limitation
  - however, the full JAX execution still does not complete:
    - `shard_map` now gets past launch-attribute setup but fails later with DeepEP receiver timeouts and `cudaMemcpyAsync(read num_recv_tokens): unspecified launch failure`
    - `pmap` fails earlier during JAX input placement with `cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- A new host-only same-process control on the isolated `iris-3677-jax` H100x8 lane moves the frontier again:
  - the pod launcher itself is no longer a blocker; it now fetches the pushed branch snapshot directly instead of inlining large staged files into `bash -lc`
  - the `CUDAExtension` transport build now succeeds under JAX `0.8.0` after explicit bf16 pointer casts in `deepep_transport_ffi.cu`
  - the host-only all-ranks dispatch round reaches a real H100 dispatch launch and logs:
    - `attr_status_code=0`, `set_attr_status_code=0`
    - `binary_version=90`, `ptx_version=90`
    - `requested_smem_bytes=196608`
  - but it still fails after launch, outside `jax.jit` and outside `shard_map` / `pmap`:
    - repeated `DeepEP timeout for dispatch receivers ... tokens remained: ...`
    - repeated `DeepEP timeout for dispatch senders ...`
    - final `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
  - so the remaining blocker is no longer well-described as an XLA-specific execution bug; it reproduces in a same-process host-only dispatch round using the same singleton runtime manager
- The smallest deterministic same-process host round also fails:
  - shape: `tokens=128` globally (`16` per rank), `hidden=128`, `num_experts=8`, `topk=1`
  - launch debug is still healthy:
    - `attr_status_code=0`, `set_attr_status_code=0`
    - `binary_version=90`, `ptx_version=90`
  - but post-launch progress still stalls:
    - repeated receiver timeout lines with `tokens remained: 1`
    - several ranks also show `tokens remained: 11` on channel `0`
    - final surfaced failure: `rank 2: cudaStreamSynchronize(dispatch): unspecified launch failure`
  - so the remaining same-process host-round bug is not specific to the larger sealed `#3633` shape either
- A new no-eager-sync refactor changed the failure shape again:
  - commit: `2994b3d15d6dfc7ebbd76a1bf303a119050c800a`
  - change:
    - stop forcing `cudaStreamSynchronize(dispatch)` inside the shared `DispatchOnCurrentDevice(...)` helper
    - copy `local_expert_counts` asynchronously once `WaitForRecvCounts(...)` has already produced stable metadata
    - for the host-only control, enqueue dispatch on every rank first, then use a second host barrier, then synchronize each per-rank stream
  - on the default tiny deterministic host-only control (`dispatch_num_sms=20`):
    - all `8/8` ranks now reach `after_dispatch_launch`
    - all `8/8` ranks also reach the new `after_launch_sync_barrier`
    - only after that do they begin timing out and fail during `cudaStreamSynchronize(dispatch)`
  - on the same tiny control with `dispatch_num_sms=2`:
    - all `8/8` ranks again reach `after_dispatch_launch`
    - all `8/8` ranks reach `after_launch_sync_barrier`
    - rank `0` now fully reaches `after_dispatch_sync` and `thread_done`
    - the remaining failing ranks still fail in `cudaStreamSynchronize(dispatch)`
    - their timeout lines are now concentrated on `responsible_channel = 0`
    - the stuck token counts are `8` or `16`, which is larger than the default `dispatch_num_max_send_tokens=6`
  - this is the strongest partial success so far:
    - the wrapper's own eager host sync was part of the problem
    - the remaining failure is now more sharply localized to live dispatch progress / configuration on channel `0`

Root-cause summary:
1. `#3665` never exercised the winning DeepEP / Hybrid-EP transport kernels. It only replaced dispatch-layout metadata and then still used JAX `ragged_all_to_all`.
2. That kernel-coverage difference is sufficient to explain the main discrepancy: on the same shape, Megatron using real DeepEP / Hybrid-EP is positive while JAX `deepep_layout_ragged_a2a` remains tied with plain ragged.
3. A naive JAX -> Torch bridge each training step would erase the gain. The direct bridge cost is tens of milliseconds, far larger than the sub-millisecond to low-millisecond transport kernel time.
4. JAX likely also has a separate absolute-performance issue in grouped expert GEMMs on this shape; the JAX run emitted repeated XLA slow-kernel warnings on `topk=8`. That helps explain absolute JAX throughput, but it is not needed to explain why `deepep_layout_ragged_a2a` failed to beat `ragged_a2a`.
5. The current pure-JAX blocker is not an obvious high-level interface mismatch. On the newer DeepEP branch, JAX reaches the real transport launch and the deterministic probe shows JAX and Torch agree at the pre-launch boundary.
6. The actual current divergence is even narrower than “runtime behavior”: Torch resolves a valid dispatch kernel symbol with real H100 function attributes, while the raw JAX FFI build returns `named symbol not found` from `cudaFuncGetAttributes(...)` / `cudaFuncSetAttribute(...)` on that same kernel specialization. That points to build / device-link / runtime-registration mechanics in the pure-JAX custom-call `.so`.
7. Fixing the stale `sm_52` fallback in the raw link command was necessary but not sufficient. After an explicit object-compile + `-dlink` pipeline and a guaranteed fresh rebuild, the symbol-resolution failure still remains, so the unresolved piece is deeper in raw CUDA device registration / fatbin packaging rather than just arch selection or cache staleness.
8. Artifact inspection weakens that packaging-only story: the raw custom-call `.so` on disk already contains `.nv_fatbin`, `sm_90` cubins, `__cudaRegisterLinkedBinary_*`, `__fatbinwrap_*`, and the expected DeepEP intranode symbols.
9. Explicit `RTLD_GLOBAL` loading also does not fix the failure.
10. A plain host-side probe reproduces the same `named symbol not found` line before any XLA execution, so the remaining divergence is not specific to `jax.jit`, `shard_map`, or XLA-owned streams. The failure is already present in the raw library load / runtime-init path itself.
11. A closer-to-upstream `CUDAExtension` build/load path clears that specific kernel-symbol-resolution failure, so the remaining blockers are now deeper:
   - `shard_map`: multi-GPU progress / rendezvous during live transport
   - `pmap`: peer-access-state ownership and ordering during JAX input placement
12. The new host-only same-process dispatch control shows that the post-launch hang is not confined to JAX execution APIs. With the extension-style build, the remaining failure now reproduces even outside `jax.jit`, `shard_map`, and `pmap`, which points at same-process DeepEP runtime assumptions or rendezvous/progress ordering rather than XLA custom-call semantics alone.
13. The smallest deterministic same-process host round still hangs, so the remaining failure is not a large-shape artifact. That further weakens “bad payload for the sealed shape” and strengthens the case for a shape-independent same-process runtime/progress bug.
14. The latest no-eager-sync rerun is a real narrowing step:
    - all ranks now launch before failure
    - with `dispatch_num_sms=2`, at least one rank can complete fully
    - the remaining failures are concentrated on `responsible_channel = 0`
    - the stuck token counts exceed the default `dispatch_num_max_send_tokens` cap
15. So the next live hypothesis is no longer just “same-process is hard.” It is more specific:
    - the wrapper's old host sync discipline was wrong
    - and the surviving tiny-case failures may also reflect underprovisioned dispatch token-cap configuration on the hot channel

## Decision Log

- 2026-03-14: start a new experiment rather than reopening `#3665` or `#3666`, because the new question is explanatory and cross-methodological rather than just another benchmark variant.
- 2026-03-14: branch from the sealed Megatron snapshot so the new thread starts from the previously working JAX custom-call and Megatron harness state.
- 2026-03-14: make the first milestone a root-cause matrix, not an optimization sprint.

## Negative Results

- `#3665` is already a sealed negative result for the narrow claim that replacing only the JAX dispatch-layout metadata producer improves the `#3633` fixed-shape H100x8 benchmark.
- The first direct dispatch-isolation attempt failed because the harness misused the DeepEP cached-handle API; fixing that did not change the final conclusion.
- The second dispatch-isolation attempt failed because the PyTorch benchmark image did not have JAX installed; adding `jax[cuda12]==0.8.0` resolved that setup issue.
- The older pinned DeepEP ref introduced a separate device-link / CUDA registration failure, so it is not a trustworthy baseline for judging the pure-JAX transport path in this environment.
- The first explicit raw `-dlink` rerun still inherited a stale-cache / bad-final-link path; fixing that removed the `sm_52` warning, but it did not change the actual DeepEP dispatch symbol-resolution failure.
- Changing the raw loader from `ctypes.cdll.LoadLibrary(...)` to `ctypes.CDLL(..., RTLD_NOW | RTLD_GLOBAL)` did not change the `named symbol not found` failure.
- Calling `deep_ep::intranode::dispatch(...)` from a plain host-side probe in the raw custom-call library still reproduces the same `named symbol not found` launch-debug line and downstream assertion.

## Conclusion

The root cause is now specific:

- The positive H100x8 gains from `#3641` / `#3666` do exist on the sealed `#3633` shape, but only when the benchmark actually uses DeepEP / Hybrid-EP dispatch-combine transport.
- The negative JAX result in `#3665` was not a faithful test of that path. It swapped in DeepEP layout metadata, but still paid JAX `ragged_all_to_all` transport costs, so it stayed tied with plain `ragged_a2a`.
- Direct Torch interop is not an acceptable substitute for a real JAX integration because the JAX -> Torch bridge cost is too large to pay per step.
- The current pure-JAX transport prototype has moved beyond those earlier conceptual blockers: on `refs/heads/hybrid-ep`, it reaches the actual DeepEP transport launch, and a deterministic Torch-vs-JAX probe shows the frameworks agree on the high-level inputs reaching that boundary.
- The newest launch instrumentation makes that still more specific: the raw pure-JAX FFI build fails at CUDA kernel symbol resolution / function-attribute lookup for DeepEP dispatch, while the Torch build resolves the same kernel cleanly on the same H100x8 machine.
- A corrected explicit `-dlink` rebuild removed the stale `sm_52` fallback warning but left the same `named symbol not found` failure in place, so the remaining work is now squarely about raw CUDA device registration / fatbin packaging in the pure-JAX custom-call artifact.
- Artifact inspection and the `RTLD_GLOBAL` rerun narrow that further: the raw artifact already looks structurally sane on disk, so the remaining blocker likely lives in how the JAX/XLA custom-call execution path interacts with CUDA symbol lookup at runtime, not just in what got compiled into the library.
- The host-side probe sharpens that again: the same failure appears before XLA executes anything, so the remaining blocker is now best described as a raw library load / runtime-registration problem in-process, not an XLA-specific execution bug.
- The latest extension-style bring-up changes that conclusion in one important way: a `CUDAExtension` build plus Python-extension import and Torch-library preload can resolve the dispatch kernel correctly from the pure-JAX path. So the frontier has moved again.
- The new remaining blockers are execution-model issues rather than raw symbol resolution:
  - `shard_map` reaches real transport and then times out / launches fail during receiver progress
  - `pmap` collides with peer-access setup before transport launch
- The newest same-process host-only control sharpens that further: once the extension-style build clears symbol resolution, the dispatch hang reproduces even without JAX execution APIs. That suggests the next debugging layer is the same-process runtime/rendezvous model itself, not just XLA custom-call sequencing.
- The tiny deterministic rerun sharpens it again: the hang survives even on `tokens_per_rank=16, hidden=128, experts=8, topk=1`, so the next step should focus on runtime/progress instrumentation or changing the process model, not on large-shape-specific routing data.
- The latest no-eager-sync rerun sharpens it again:
  - the wrapper-level eager host sync was genuinely part of the problem
  - after removing it, all ranks now enqueue dispatch before failure
  - with `dispatch_num_sms=2`, one rank can complete fully and the remaining failures concentrate on channel `0`
  - the next concrete check should be whether the hot channel is simply underprovisioned by the default `dispatch_num_max_send_tokens` cap before we conclude that the remaining blocker is purely architectural

The next meaningful follow-up is therefore not “retune `deepep_layout_ragged_a2a`”. It is:
1. re-run the tiny deterministic host-only control with the new no-eager-sync path and a higher `dispatch_num_max_send_tokens`
2. if that clears the remaining channel-`0` failures, propagate the same synchronization discipline into the real JAX `shard_map` path
3. only if it does not, escalate from “config/progress bug” back to “same-process runtime-model mismatch” and continue with the process-model fork

## New milestone: tiny pure-JAX `shard_map` success

`2026-03-15`

The tiny deterministic H100x8 `shard_map` case is now a real end-to-end success for the pure-JAX transport path.

- Command:
  ```bash
  KUBECONFIG=/Users/romain/.kube/coreweave-iris \
  uv run .agents/scripts/deepep_jax_transport_krt.py \
    --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --build-with-torch-extension \
    --load-as-python-module \
    --launch-debug \
    --launch-debug-label shard-map-min-sms2-fix6 \
    --tokens 128 \
    --hidden 128 \
    --experts 8 \
    --topk-list 1 \
    --distributions deterministic \
    --dispatch-num-sms 2 \
    --dispatch-num-max-send-tokens 32 \
    --timeout-seconds 3600
  ```
- Result:
  - dispatch launch resolved and executed on all `8` H100s
  - correctness passed exactly:
    - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
  - timing:
    - `RESULT step_s=0.000480 tokens_per_s=266519.71`

This is the first direct evidence that the pure-JAX DeepEP transport path can execute correctly on H100x8 with zero Torch in the step path.

The remaining bug from this run is now smaller and later:
- the benchmark process stayed alive after printing the final result instead of exiting cleanly

So the thread has moved from “can pure-JAX transport run?” to “why does the successful benchmark process hang during teardown?”
