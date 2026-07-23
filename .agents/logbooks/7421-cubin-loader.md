---
topic: 7421-cubin-loader
issue: https://github.com/marin-community/marin/issues/7421
description: Instrument the production CUDA module-loader boundary.
author: mcwitt
---

# Issue #7421 CUDA module loader: Task Logbook

## Scope

- Goal: classify the L48/B512 CUBIN failure as a prior asynchronous error, loader API mismatch, pointer or ordering behavior, resource pressure, or image/context state.
- Primary metrics: per-rank loader outcome sequence, synchronization result, retry result, HBM/pool snapshot, and first-step outcome.
- Constraints: preserve the confirmed 16xGB200x4 graph and process topology; no Iris restart; region-local artifacts only.
- Coordinating issue: https://github.com/marin-community/marin/issues/7421

## Hypothesis Queue

### Active

- `H-ASYNC`: a preceding asynchronous CUDA operation failed.
- `H-API`: raw ELF CUBINs fail specifically through `cuModuleLoadFatBinary`.
- `H-ORDER`: concurrent XLA/CuTe loader ordering changes driver state.
- `H-INPUT`: pointer lifetime, mutation, or alignment changes the result.
- `H-PRESSURE`: a live memory or module resource is exhausted.

## Entry Log

### 2026-07-21 - CUBIN7421-001 probe implementation start

- Hypothesis: module-boundary outcomes can distinguish the active hypotheses.
- Commit Hash: `0a3785463`
- Command: local fake-driver integration tests, then one-GPU JAX smoke.
- Config: trace, sync, Data-direct, and pressure profiles.
- Result: implementation in progress.
- Interpretation: no new runtime evidence yet.
- Next action: make private-handle `dlsym` interception fail under test.

### 2026-07-21 - CUBIN7421-001 local instrumentation checkpoint

- Hypothesis: a `dlsym` interposer can observe XLA's private-driver-handle lookup and preserve enough loader state to distinguish async errors, API behavior, input ownership, overlap, and a paired pressure treatment.
- Commit Hash: based on `0a3785463`; checkpoint commit pending.
- Commands:
  - `uv run ... pytest experiments/grug/test_dispatch.py experiments/grug/moe/standalone/test_cuda_module_probe.py experiments/grug/moe/standalone/test_cuda_module_probe_tools.py`
  - `./infra/pre-commit.py <instrumentation files>`
  - `uv run --with pyrefly pyrefly check <instrumentation Python files>`
  - `g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared ...`
- Config: private-handle fake CUDA driver; trace, sync, Data-direct, and pressure profiles; two-thread overlap; task-zero content-addressed capture.
- Result: 16 focused tests passed in 14.83 seconds. Scoped repository lint passed. Pyrefly reported zero errors. The probe compiled with warnings treated as errors. Source SHA-256: `f79b8e1d5090f6a4c42605cca2557ff335f70dc1482df9e8417f9b10d86c2100`.
- Interpretation: the local harness validates the event and treatment contracts. It does not establish that the interposer reaches JAX/XLA in the production CUDA environment.
- Next action: run a one-GB200 JAX smoke and require both a FatBinary symbol redirect and a raw ELF load before the 16-host diagnostic.

### 2026-07-21 - CUBIN7421-002 one-GB200 smoke attempt 1

- Hypothesis: the locally validated interposer reaches JAX/XLA unchanged on an aarch64 GB200 worker.
- Commit Hash: `db61c4a4f`.
- Job: `/mwittmann/cubin7421-probe-smoke-001`.
- Config: one GB200; JAX 0.10.1; trace profile; required FatBinary and raw ELF coverage; task-zero capture.
- Result: failed before JAX output with no probe events. The worker built the expected source hash and binary, then the child returned before the coverage check.
- Interpretation: `real_dlsym` requested only the x86-64 symbol version `GLIBC_2.2.5`; the aarch64 worker uses `GLIBC_2.17`. This is an instrumentation portability defect, not evidence about the CUDA failure.
- Next action: resolve both glibc symbol versions, rerun local tests, and repeat the same smoke.

### 2026-07-21 - CUBIN7421-002b one-GB200 smoke attempt 2

- Hypothesis: after resolving the aarch64 glibc symbol version, the interposer reaches the actual raw-ELF loader path used by JAX 0.10.1 on GB200.
- Commit Hash: `3ec1f64cc`.
- Job: `/mwittmann/cubin7421-probe-smoke-002`.
- Config: one GB200; JAX 0.10.1; trace profile; required FatBinary and raw ELF coverage; task-zero capture.
- Result: succeeded. JAX returned `cuda:0` and sum `1048575.875`. The probe observed 10 `cuModuleLoadFatBinary` calls on raw ELF inputs, all with original result 0; eight unique CUBIN hashes; maximum in-flight loads 1. Source SHA-256: `974d807b893db6642522bb15c16a40e72173947a3f9d031ea09c199f52cdae21`. Binary SHA-256: `253648ba9e9b4b46a318982cb777bb5c0068bcf169670fb527b759fef201f436`.
- Artifacts: `s3://marin-us-east-02a/marin/scratch/mwittmann/cubin-diag/CUBIN7421-002b/task-0/`.
- Interpretation: the probe builds and loads on the target architecture, intercepts the production XLA loader API, parses the raw ELF inputs, and preserves a successful JIT. The instrumentation is ready for the multi-host positive control.
- Next action: run the exact archived L48/B512 configuration with task-index split trace/sync profiles and automatic retries disabled.

### 2026-07-21 - CUBIN7421-003 split trace/sync diagnostic

- Hypothesis: comparing the same failing graph across trace and pre-load synchronization ranks will distinguish a loader-local failure from an earlier asynchronous CUDA error.
- Commit Hash: `7ae727a82`.
- Jobs: `/mwittmann/cubin7421-trace-sync-l48-b512-coord` and `/mwittmann/cubin7421-trace-sync-l48-b512-002-coord`.
- Config: 16 GB200x4 hosts; d5120, L48, sequence 4096, 128 experts, top-5, batch 512; intermediate and shared intermediate 2560; ring EP8, replica axis 1; `gpu_fa4_cute`; scanned layers; CuTe MXFP8 producer; liger CE; MuonH with all three distribution knobs; `cuda_async`; local checkpoints; one training step. Even task indices use trace, odd task indices synchronize immediately before each load. Probe and Iris failure retries are zero.
- Result: the first coordinator failed before creating a GPU child because `bash -lc` reset the virtual-environment path and selected a Python without Fray. The corrected coordinator launched all 16 hosts and reproduced `CUDA_ERROR_INVALID_VALUE` at `jit_train_step`. Current Iris exports `IRIS_TASK_ID`, not `IRIS_TASK_INDEX`; every worker therefore defaulted to trace, and artifact upload raised `KeyError` before preserving the local files.
- Interpretation: the positive control remains reproducible with the probe loaded, but this attempt cannot compare trace with sync or classify the CUDA failure. The missing task identity is an instrumentation integration defect.
- Next action: derive the task index from canonical `IRIS_TASK_ID` in both the preload library and uploader, verify parity and task-zero capture in the fake-driver harness, then repeat after the GPU cooldown.

### 2026-07-21 - CUBIN7421-003b task-identity correction

- Hypothesis: canonical Iris task IDs provide the stable rank index required for profile parity and per-task artifact paths.
- Commit Hash: `30fdb93e7`.
- Commands: focused fake-driver and tooling tests; strict C++ compilation; scoped repository lint.
- Config: `IRIS_TASK_ID=/user/job/1:0` selects sync; `IRIS_TASK_ID=/user/job/0:0` enables task-zero capture; legacy `IRIS_TASK_INDEX` remains accepted for direct tools.
- Result: 19 focused tests passed. The C++ source compiled with `-Wall -Wextra -Werror`; scoped repository lint passed.
- Interpretation: both the interposer and Python uploader now derive the same index from the runtime identity current Iris actually exports.
- Next action: commit the correction and repeat the exact CUBIN7421-003 split run after cooldown.

### 2026-07-21 - CUBIN7421-003c corrected split trace/sync diagnostic

- Hypothesis: if a preceding asynchronous CUDA operation failed, odd task indices will surface that error from `cuCtxSynchronize` before entering the module loader.
- Commit Hash: `30fdb93e7`.
- Jobs: `/mwittmann/cubin7421-trace-sync-l48-b512-003-coord`, child `/mwittmann/cubin7421-trace-sync-l48-b512-003-coord/grug-train-cubin7421-trace-sync-l48-b512-003`, and collector `/mwittmann/cubin7421-probe-collect-003`.
- Config: the same 16-host L48/B512 positive control as CUBIN7421-003. Even task indices used trace; odd task indices synchronized before every module load. Retries remained disabled.
- Result: all 16 tasks uploaded complete artifacts. The probe recorded 94,116 FatBinary calls: 94,052 succeeded and 64 returned `CUDA_ERROR_INVALID_VALUE`, with four failures per task. All 47,024 sync-profile calls returned `CUDA_SUCCESS` from the pre-load synchronization. The 64 failures used an unknown, page-aligned input and occurred once per local device. Post-failure context and device queries succeeded, while `cuMemGetInfo` returned 201, so no HBM snapshot was available at the failing boundary. Maximum observed loader concurrency was four.
- Artifacts: `s3://marin-us-east-02a/marin/users/root/experiments/grug-moe-cw/grug-moe-cw-d5120-L48-e128-r16-cubin7421-trace-sync-l48-b512-003/dev/cuda-module-probe/`.
- Interpretation: a stale asynchronous CUDA error is not the direct source of the reported loader error. The original FatBinary call itself returns invalid value after a successful synchronization. OOM remains possible only through an upstream mechanism that produces an absent or malformed image, or through loader-local resource state; this run did not measure free memory at the failure.
- Next action: record whether unknown inputs are null. If they are non-null, add a bounded prefix identity before selecting Data-direct or paired-pressure treatments.

### 2026-07-21 - CUBIN7421-004 null-input classification

- Hypothesis: the 64 unknown inputs are null, which would locate the immediate defect before the CUDA loader rather than in CUBIN architecture or pointer handling.
- Commit Hash: `b77c2c283`.
- Jobs: `/mwittmann/cubin7421-null-classify-l48-b512-004-coord`, child `/mwittmann/cubin7421-null-classify-l48-b512-004-coord/grug-train-cubin7421-null-classify-l48-b512-004`, and collector `/mwittmann/cubin7421-null-collect-004`.
- Config: the same 16-host L48/B512 graph and split trace/sync profiles as CUBIN7421-003c. The only probe change was an explicit `image_is_null` field on each load entry.
- Result: the failure reproduced at `jit_train_step`, and all 16 task artifacts uploaded. Of 94,244 FatBinary calls, all 94,180 successful calls had non-null images and all 64 failed calls had null images. Every failed call returned `CUDA_ERROR_INVALID_VALUE`; its immediately preceding synchronization returned `CUDA_SUCCESS`. Each task again recorded four failures, one per local GPU.
- Artifacts: `s3://marin-us-east-02a/marin/users/root/experiments/grug-moe-cw/grug-moe-cw-d5120-L48-e128-r16-cubin7421-null-classify-l48-b512-004/dev/cuda-module-probe/`.
- Interpretation: the immediate failure is not a wrong-architecture or malformed CUBIN rejected by the driver. The caller passes a null module image, and the driver reports invalid argument; XLA then renders that as the generic “compiled for a different GPU?” message. An upstream OOM remains consistent with the observations if it causes compilation or assembly to yield no image, but this evidence does not identify why the image is null. Data-direct, pointer-copy, and loader-pressure treatments cannot be informative for a null input.
- Next action: instrument the producer of the module byte span so that an empty compiler/assembler result is paired with its originating status and memory state before the loader call.

### 2026-07-22 - CUBIN7421-005a NVIDIA JAX 26.06 identity smoke

- Hypothesis: NVIDIA's current JAX container runs on the GB200 fleet without Iris creating a virtual environment or replacing the image's JAX stack.
- Commit Hash: `d9dcb4d58` (no code changes required for the direct image smoke).
- Job: `/mwittmann/cubin7421-ngc-identity-2606-001`.
- Config: one GB200; `nvcr.io/nvidia/jax:26.06-py3`; Iris `--no-sync`; direct image Python; JAX/JAXLIB path and binary hashes; one GPU JIT.
- Result: succeeded. Python was `/usr/bin/python`; JAX `0.10.1.dev20260605+10439788c` loaded from `/opt/jax`; JAXLIB loaded from `/opt/jaxlibs`; PJRT reported CUDA 13.3; driver 595.71.05; the JIT completed on `cuda:0`. `/app/.venv` was absent from `sys.path`.
- Interpretation: the unmodified container is compatible with the target node and driver. This does not establish that Marin can run without overlay dependencies.
- Next action: install only missing Marin dependencies into a system-site-packages overlay and require the container JAX/JAXLIB hashes and import paths to remain unchanged.

### 2026-07-22 - CUBIN7421-005b guarded Marin overlay

- Hypothesis: a system-site-packages venv can add Marin without installing or shadowing the NGC JAX, JAXLIB, CUDA, or NVIDIA Python packages.
- Commit Hash: `cf174fee8`.
- Jobs: `/mwittmann/cubin7421-ngc-overlay-2606-001` through `-004`.
- Config: one GB200; `nvcr.io/nvidia/jax:26.06-py3`; temporary `uv` bootstrap; root and Levanter-GPU syncs with explicit accelerator-package exclusions; pre/post hashes for JAX, JAXLIB, `_jax.so`, and `libjax_common.so`; full scale-launcher import; one GPU JIT.
- Result: `-001` stopped because the image omits `ensurepip`; `-002` proved both syncs and hash checks but exposed the missing workspace-root `marin-dupekit` dependency; `-003` stopped because `--no-group dev` is invalid for `marin-root`; corrected `-004` succeeded. The four hashes were identical before and after sync, neither `jax` nor `jaxlib` existed under `/app/.venv`, imports still resolved to `/opt`, the full launcher imported, and the JIT completed.
- Interpretation: the overlay preserves the container accelerator stack while supplying the repository runtime. The staged failures were setup-contract failures before training, not CUDA-loader evidence.
- Next action: run the exact 16-host L48/B512 one-step graph with the NGC image and zero retries, initially without the loader probe.

### 2026-07-22 - CUBIN7421-005c exact NVIDIA JAX comparison

- Hypothesis: if the null-image failure is specific to the original JAX/container stack, the exact positive-control graph will complete under NVIDIA's current JAX image while its JAX and JAXLIB remain unchanged.
- Commit Hash: `66fa9cba2`.
- Jobs: `/mwittmann/cubin7421-ngc2606-l48-b512-006-coord`, `/mwittmann/cubin7421-ngc-cutlass-id-2606-007`, `/mwittmann/cubin7421-ngc-cutlass-smoke-2606-009`, `/mwittmann/cubin7421-ngc2606-l48-b512-010-coord`, and confirmation `/mwittmann/cubin7421-ngc2606-l48-b512-011-coord`.
- Config: the same 16-host, four-GB200-per-host L48/B512 graph as CUBIN7421-004, using `nvcr.io/nvidia/jax:26.06-py3`; JAX `0.10.1.dev20260605+10439788c` and JAXLIB `0.10.1.dev20260722` loaded from `/opt`; CUDA 13.3; a guarded system-site overlay supplying the repository and the locked CUDA 13 CUTLASS DSL; zero retries; one training step.
- Result: the first exact attempt reached compilation but failed with `NVVM_ERROR_COMPILATION: unsupported operation`, not the target loader error. The image contained both `nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13`, which install the same `_cutlass_ir` files. Excluding the base wheel and overlaying the locked CUDA 13 CUTLASS packages fixed that independent collision; the full dual-quantizer smoke then passed. Both corrected exact runs completed the training step on all 16 ranks: 2,097,152 tokens and loss `11.804323196411133`, with no `cuModuleLoadFatBinary`, NVVM, CUDA, coordination, or OOM error during compilation or execution. After the successful step, each run force-saved the 248-billion-parameter checkpoint to local `/tmp`; rank 0 exited 137 during that separate checkpoint phase and Iris killed its peers.
- Interpretation: the original null-image loader failure did not reproduce in two exact model-step runs under the preserved NVIDIA JAX/JAXLIB stack. This is evidence that the failure depends on some difference in the original software image or compilation stack, but it neither proves which component differs nor disproves an upstream OOM mechanism in that original stack. The post-step exit 137 is a distinct local-checkpoint capacity failure and is not evidence about the module loader.
- Next action: compare producer-side compilation status and memory state in the original stack; the loader probe is unnecessary for another NGC run unless the null-image signature reappears.

### 2026-07-22 - CUBIN7421-006 cross-reproducer NVIDIA JAX 26.06 matrix

- Hypothesis: if the NVIDIA JAX image resolves the loader failure generally, independent full-size graphs previously associated with the failure will complete model steps under the preserved NGC stack, with zero retries.
- Commits: `6c089ba5e` (#7407 launcher), `2b87a6d37` (#7279 launcher and allocator forwarding), and `ced1afa90` (original B1024 launcher).
- Jobs:
  - #7407 heterogeneous-KV nested-scan BF16: `/mwittmann/cubin-ngc2606-7407-01/grug-train-cubin-ngc2606-7407-01-20260722`.
  - #7279 pure-XLA ragged-all-to-all EP16: default-BFC `/mwittmann/cubin-ngc2606-7279-01/grug-train-cubin-ngc2606-7279-01`, then `cuda_async` `/mwittmann/cubin-ngc2606-7279-01a/grug-train-cubin-ngc2606-7279-01a`.
  - Original CuTe/MXFP8/ring B1024: `/mwittmann/cubin-ngc2606-b1024-01/grug-train-cubin-ngc2606-b1024-01-20260722`.
- Config: `nvcr.io/nvidia/jax:26.06-py3`; guarded overlay preserving JAX `0.10.1.dev20260605+10439788c` from `/opt/jax` and JAXLIB `0.10.1.dev20260722` from `/opt/jaxlibs`; 16 four-GB200 hosts; one model step; no final checkpoint; zero retries. A one-host #7279 model smoke completed with loss `11.802978515625` before the full run.
- Result: all three full-size graph families reproduced the exact target failure on task 0: `Failed to load in-memory CUBIN (compiled for a different GPU?).: CUDA_ERROR_INVALID_VALUE: invalid argument [executable_name='jit_train_step']`; coscheduled siblings then exited as cascades. The first #7279 run, using the default BFC allocator, instead explicitly failed while allocating 82.53 GiB in `jit_train_step`. Forwarding `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` avoided that BFC failure and exposed the target CUBIN loader error. The pre-agreed acceptance criterion required every trial to succeed; because each distinct full-size family failed, duplicate trials were stopped as non-decision-relevant.
- Interpretation: NGC 26.06 is not a general fix for the CUBIN loader failure. Its two earlier B512 successes and the successful smokes show that the image can execute related graphs, but the outcome is graph- and allocator-sensitive. The #7279 A/B result strengthens the possibility that memory pressure can alter which surface error is reported, but it does not prove that OOM causes the null-image loader failures. Producer-side instrumentation remains necessary to capture why compilation returns no image.
- Next action: retain the NGC launcher as a comparison environment, but do not close #7421 as solved; instrument the module-image producer status and memory state upstream of `cuModuleLoadFatBinary`.

### 2026-07-22 - CUBIN7421-007 memory-headroom and producer treatments

- Hypothesis: if the NGC reproductions are masked device OOMs, reducing memory pressure while retaining the important graph structure will remove the CUBIN surface and permit an actual training step. A separate producer treatment tests whether the CuTe/MXFP8 family depends on producer-specific codegen rather than only persistent memory.
- Commits: `6c089ba5e` (#7407 launcher), `2b87a6d37` (#7279 launcher), and `ced1afa90` (original CuTe/MXFP8 launcher).
- Config: all counted runs used `nvcr.io/nvidia/jax:26.06-py3`, JAX from `/opt/jax`, JAXLIB from `/opt/jaxlibs`, the guarded dependency overlay, 16 four-GB200 hosts, one model step, no final checkpoint, and zero retries.
- #7407 results:
  - The B1024/L48 baseline had an XLA plan of 175.08 GiB before and 171.62 GiB after rematerialization, versus a roughly 133.93 GiB target, then failed with the CUBIN error.
  - Changing only parameter storage from FP32 to BF16 reduced the post-rematerialization plan to 146.66 GiB, a measured 24.96 GiB/device reduction close to the predicted 26.159 GiB/device, but still failed with the same CUBIN error.
  - Retaining BF16 parameter storage and reducing L48 to L30 reduced parameters from 359,842,007,040 to 225,493,442,304. `/mwittmann/cubin-ngc2606-7407-bf16-l30-01/grug-train-cubin-ngc2606-7407-bf16-l30-01-20260722` completed on all 16 ranks with loss `11.811906814575195` and no CUBIN or explicit OOM surface. The attempted L32 treatment was invalid because heterogeneous KV requires the layer count to be divisible by six and is not counted.
- #7279 results:
  - The B1024/L48 default-BFC baseline explicitly failed on an 82.53 GiB allocation. The same graph with `cuda_async` instead surfaced the target CUBIN error.
  - L40 removed the CUBIN surface but exposed a separate command-buffer `SIGSEGV` in `xla::gpu::GetKernelAnnotation`. Disabling command buffers allowed execution to advance further, then a `cublasLtMatmul` launch failed with CUDA containment cleanup errors; cuda_async reported 131.62 GiB maximum in use against a 138.22 GiB limit and a 74.65 GiB maximum allocation.
  - Reducing to L32 with command buffers disabled gave enough additional headroom. `/mwittmann/cubin-ngc2606-7279-l32-nocb-01/grug-train-cubin-ngc2606-7279-l32-nocb-01` completed on all 16 ranks with loss `11.806449890136719` and no CUBIN surface.
- CuTe/MXFP8/ring results:
  - Reducing L48 to L24 at fixed B1024 removed about 16.52 GiB/device of expected persistent state and eliminated the XLA memory-planner warning, but `/mwittmann/cubin-ngc2606-b1024-l24-01/grug-train-cubin-ngc2606-b1024-l24-01-20260722` still failed with the CUBIN error on all 16 ranks.
  - At the same B1024/L24 graph, changing only `SCALE_FP8_PRODUCER=cute` to `xla` made `/mwittmann/cubin-ngc2606-b1024-l24-xla-02/grug-train-cubin-ngc2606-b1024-l24-xla-02-20260722` complete with loss `11.806529998779297`. A preceding `-01` attempt failed during coordinator setup before GPU submission and is not counted.
  - With the CuTe producer retained, reducing only global batch to B768 made `/mwittmann/cubin-ngc2606-b768-l24-cute-01/grug-train-cubin-ngc2606-b768-l24-cute-01-20260722` complete with loss `11.806591033935547`.
  - B832 also completed, with loss `11.806538581848145`, even though its estimated ring all-gather contains 2,181,038,080 BF16 elements, just above 2^31. This rules out a sharp failure at that element-count boundary. The CuTe threshold remains somewhere above B832 and at or below B1024, if batch-dependent memory is the cause.
- Interpretation: #7407 and #7279 provide strong experimental evidence that the CUBIN surface can mask insufficient device-memory headroom: in each family, progressively reducing memory pressure removes the loader error and eventually permits a real training step. The allocator-dependent #7279 transition from an explicit 82.53 GiB BFC OOM to CUBIN under cuda_async is especially direct. These results do not justify treating every CUBIN failure as OOM. The CuTe/MXFP8/ring family is batch- and producer-dependent: B768 and B832 CuTe pass, B1024 CuTe fails, and B1024 XLA-producer passes. That remains consistent with transient producer-specific memory pressure, but it also fits a CuTe codegen or runtime defect at larger shapes. Prior MXFP8 experiments showed the same producer split repeatedly, so this family should remain a distinct open mechanism until producer-side status and peak-memory instrumentation separate those explanations.
- Next action: use the NGC image as the controlled comparison environment, treat OOM/headroom as the leading explanation for the #7407 and #7279 graphs, and capture producer-side failure status plus peak/pool memory for the B1024 CuTe/MXFP8 boundary before generalizing the OOM diagnosis.

### 2026-07-22 - CUBIN7421-008 one-process-per-GPU topology treatment

- Hypothesis: the four simultaneous loader calls observed by CUBIN7421-003c race because the default topology places four CUDA contexts and executors in one JAX process. Moving from one process per four-GPU node to one process per GPU should remove process-local loader concurrency and eliminate the CUBIN failure if that is the cause.
- Commit Hash: `8d2af76f5`.
- Baseline: `/mwittmann/cubin-ngc2606-b1024-l24-01/grug-train-cubin-ngc2606-b1024-l24-01-20260722` emitted `processes_per_task: 1`: 16 JAX processes, four local GB200s each. At B1024/L24 with the CuTe MXFP8 producer it failed with the target CUBIN error on all 16 tasks. The earlier loader probe measured maximum in-process load concurrency four and one null-image failure per local GPU.
- Treatment: `/mwittmann/cubin-ngc2606-b1024-l24-procgpu-01/grug-train-cubin-ngc2606-b1024-l24-procgpu-01-20260722`. The only intended graph/runtime change was `processes_per_task: 1` to `4`. All 16 Iris tasks logged `supervising 4 process(es) x 1 device(s) each`, giving 64 JAX processes with one GPU each and the same global 64-device mesh. Model, batch, sharding, parameter count (124,712,317,952), CuTe producer, allocator, NGC image, and one-step target were retained. Every worker again loaded JAX from `/opt/jax` and JAXLIB from `/opt/jaxlibs`.
- Result: all 64 process ranks surfaced the exact `Failed to load in-memory CUBIN ... CUDA_ERROR_INVALID_VALUE ... jit_train_step` error within about one second of each other; no training step completed. Iris recorded the first task failure and terminated the 15 coscheduled sibling tasks.
- Interpretation: this refutes the narrow process-topology hypothesis that the failure requires four concurrent CUDA contexts or XLA executors inside one process. The error is observable with one process per GPU. A driver-global race among separate processes that all load simultaneously remains logically possible, so this run does not refute every concurrency hypothesis. It is less consistent with the existing module-boundary evidence, however: all failed loader calls received a null image from the caller after a successful pre-load synchronization, so the immediate `CUDA_ERROR_INVALID_VALUE` is explained by the null argument rather than a driver rejection of a valid image during a race. Any remaining concurrency test should serialize producer completion and module loading across processes on a node, not only switch process topology.
- Next action: keep memory/producer-side failure as the leading investigation path. If NVIDIA wants a direct race test, add a node-shared serialization treatment around producer completion and module loading and compare it against the same deterministic B1024/L24 graph.
