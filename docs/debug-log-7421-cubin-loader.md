# Debugging log for issue #7421 CUBIN loader

Instrument the production loader boundary without changing the positive-control graph.

## Initial status

L48/B512 with the CuTe MXFP8 producer fails at the first `jit_train_step` module load. Direct module replay succeeds; OOM, prior asynchronous error, loader API, pointer ownership, and concurrent loader state remain distinguishable hypotheses.

## Hypothesis 1

XLA's private driver lookup can be redirected by wrapping `dlsym`, while ordinary non-target lookups remain unchanged.

## Changes

Added an ABI-only loader probe and fake-driver integration harness under `experiments/grug/moe/standalone/`. The probe redirects loader lookups made through private handles, identifies bounded ELF64 inputs, and records ordered loader attempts without issuing CUDA telemetry before the trace attempts. Grug workers build and preload the library before importing JAX and upload per-task artifacts on failure.

## Results

The fake driver confirms:

- non-target private-handle lookups pass through unchanged;
- trace distinguishes original, same-pointer, aligned-owned-copy, and Data retries;
- sync errors return before any module load;
- Data-direct bypasses FatBinary for raw ELF inputs;
- the pressure treatment releases its reserve only after the ordinary sequence fails;
- concurrent loads receive unique sequence numbers and record overlap;
- task 0 captures content-addressed CUBINs while other tasks retain hashes only.

The focused suite passed 16 tests. Scoped lint and type checks passed, and the C++ source compiled with `-Wall -Wextra -Werror`. This is local contract evidence only; the real XLA interception path remains to be established by the one-GPU smoke.

The first GB200 smoke failed before JAX emitted output because the interposer resolved the real `dlsym` using only x86-64's `GLIBC_2.2.5` version. GB200 workers are aarch64 and use `GLIBC_2.17`. The corrected resolver tries both versions; the failed smoke carries no CUDA-loader evidence.

The corrected GB200 smoke succeeded under JAX 0.10.1. It intercepted 10 real `cuModuleLoadFatBinary` calls on raw ELF inputs, all of which returned success through the original API, while the JIT completed with the expected result. This establishes production-path coverage and permits the multi-host diagnostic; it does not yet test the failing graph or classify the failure.

NVIDIA's `nvcr.io/nvidia/jax:26.06-py3` image also passed a direct GB200 identity/JIT smoke. Its JAX and JAXLIB load from `/opt/jax` and `/opt/jaxlibs`, respectively, under CUDA 13.3 and driver 595.71.05. A guarded system-site-packages overlay then installed the Marin workspace while excluding JAX, JAXLIB, their CUDA plugins, and CUDA/NVIDIA Python packages. Hashes of JAX, JAXLIB, `_jax.so`, and `libjax_common.so` were identical before and after the overlay; the full scale launcher imported and a GPU JIT completed. The exact L48/B512 graph is the remaining container comparison.

The first exact NGC-image run reached graph compilation but failed with `NVVM_ERROR_COMPILATION: unsupported operation`, a different signature from the null-image loader failure. The image installed both the base and CUDA 13 CUTLASS DSL library wheels, which claim the same `_cutlass_ir` files. A package-identity probe confirmed that collision. The guarded overlay now excludes the base wheel and supplies the repository's locked CUDA 13 CUTLASS DSL packages while retaining the container's JAX and JAXLIB. The full dual-quantizer smoke, including the L48/B512 tensor shapes, then passed.

Two corrected exact runs completed the full one-step graph on all 16 ranks. Each processed 2,097,152 tokens with loss `11.804323196411133`; neither emitted a FatBinary loader, NVVM, CUDA, coordination, or OOM error during compilation or execution. Both subsequently attempted to force-save the 248-billion-parameter model checkpoint to local `/tmp`, where rank 0 exited 137 and Iris terminated the remaining ranks. That post-step checkpoint-capacity failure is separate from the successful model step and from the target loader signature.

The exact graph therefore did not reproduce the original null-image failure in two runs under the preserved NVIDIA JAX/JAXLIB stack. This supports a software-image or compilation-stack dependency, but does not isolate the responsible component and does not rule out an upstream OOM in the original stack.

The first multi-host probe attempt reproduced the original `CUDA_ERROR_INVALID_VALUE` at `jit_train_step`, but current Iris provides rank identity through `IRIS_TASK_ID` rather than `IRIS_TASK_INDEX`. All workers consequently used the trace fallback, and the Python uploader raised while resolving its task path. No per-task artifacts survived, so this attempt establishes only that the probed positive control still fails. Task-index derivation now parses the canonical Iris task ID in both the interposer and uploader, with regression coverage for odd-rank sync selection and task-zero CUBIN capture.

The corrected split run produced all 16 task artifacts. Eight trace tasks and eight sync tasks recorded 94,116 `cuModuleLoadFatBinary` calls. Exactly four calls per task returned `CUDA_ERROR_INVALID_VALUE`; the other 94,052 calls succeeded. Every pre-load synchronization on the sync tasks returned success, including the 32 failing sync-profile calls. This rules out an earlier asynchronous CUDA error surfacing at the loader boundary.

The 64 failing calls differ from the successful raw-ELF loads. Their input kind is unknown, their address is page-aligned, and no bounded size or hash is available, so the safe retry sequence did not run. Each task failed once per local GPU. Context and device queries succeeded after each failure, but `cuMemGetInfo` returned CUDA code 201 and no memory counts. The run therefore does not distinguish a null or malformed compiler output from loader-local resource state. The probe now records whether the input pointer is null; a fake-driver integration test covers that event field.

The null-classification rerun reproduced the failure and uploaded all 16 task artifacts. Its 94,180 successful FatBinary calls all received non-null images. Its 64 failed calls all received null images, returned `CUDA_ERROR_INVALID_VALUE`, and followed a successful immediate synchronization. The failures again occurred four times per task, once per local GPU.

This locates the immediate defect before the CUDA loader. The driver is not rejecting a real CUBIN for architecture, ownership, or pressure reasons; it is rejecting a null argument. XLA's “compiled for a different GPU?” suffix is therefore misleading for this failure. OOM remains a viable upstream cause only if it makes compilation or assembly produce no module image, but the loader-boundary probe cannot establish that causal step. Loader retry treatments are not meaningful for a null image; the next diagnostic boundary is the producer of the module byte span and its compilation status.

## Future work

- [ ] Instrument the module-byte producer to pair an empty result with its compiler or assembler status and memory state.
- [ ] Compare the original stack's module-byte producer and memory telemetry with the passing NVIDIA JAX stack.
