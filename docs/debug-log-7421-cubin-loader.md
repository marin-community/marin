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

## Future work

- [ ] Trace the immediately preceding CUDA operation if pre-load synchronization fails.
