# Debugging log for CUDA 13 CCCL headers

Ensure the Linux GPU extras install the CCCL headers required to compile
Shuttle's generated CUDA typed-FFI handlers before another H100 allocation.

## Initial status

The attention training gate reached the final NVCC handler build with NVCC
13.3.73 and CUDA runtime headers 13.0.96, then failed when `cuda_fp16.h`
included a missing `<nv/target>` header. The environment installed
`jax[cuda13]==0.10.1`, Torch 2.11.0, and Triton 3.6.0, but it did not install
`nvidia-cuda-cccl`.

## Hypothesis 1

CUDA 13 distributes CCCL separately from the runtime and compiler packages.
Adding the unqualified `nvidia-cuda-cccl` distribution to both Linux GPU
extras should supply `<nv/target>` without changing Shuttle's JAX or Torch
boundaries.

## Changes to make

Pin `nvidia-cuda-cccl==13.3.3.4.1` in the Marin and Levanter Linux GPU extras,
update the workspace lock without upgrading unrelated packages, and compile a
translation unit that includes `cuda_fp16.h`, `cuda_bf16.h`, and `nv/target`
on a CPU-only Linux worker.

## Results

PyPI publishes `nvidia-cuda-cccl==13.3.3.4.1` for Linux x86-64 and
AArch64. The x86-64 wheel has SHA-256
`cc0adc188d570b09f4d606c7dc05a42aa3d8aa082e0d60f7bbfc5b6435f627c6`
and contains both `nvidia/cu13/include/nv/target` and
`nvidia/cu13/include/cccl/nv/target`. An independent Event Tensor CPU-only
Linux preflight used the same CCCL version with NVCC 13.3.73 and JAX 0.10.1.

A normal `uv lock` with the workstation's uv 0.8.13 changed 1,401 lock lines
and re-resolved unrelated CUDA, Torch, and vLLM packages. That output was
discarded. The targeted lock update adds 14 lines: four workspace dependency
records and the exact three-wheel CCCL package record. `uv lock --check` with
uv 0.8.13 accepts the result without re-resolution.

The CPU-only Iris job
`/dlwh/shuttle-attention-cccl-preflight-20260809` installed the exact locked
`marin-levanter:gpu` environment and succeeded with one CPU, 3 GB of memory,
9 GB of disk, and no accelerator. The resolved toolchain contained:

- `nvidia-cuda-cccl==13.3.3.4.1`;
- `nvidia-cuda-nvcc==13.2.78`;
- `nvidia-cuda-runtime==13.0.96`;
- JAX and JAXLIB 0.10.1.

NVCC 13.2.78 compiled `cuda_fp16.h`, `cuda_bf16.h`, and `<nv/target>` for
`compute_90` into a 10,288-byte object without warnings or errors. The job
completed in 6.38 seconds after scheduling. This clears the missing-CCCL
toolchain defect that stopped the previous attention gate before correctness.
A fresh H100 allocation is admissible; it must still rerun the unchanged
forward-plus-backward correctness and performance gate.

The dependency change does not add Torch to the default install or to the
compiler-facing smoke. The existing Levanter GPU extra still resolves Torch
transitively through its pre-existing FlashAttention dependency; this patch
does not use Torch or widen that boundary.

## Future work

- [ ] Re-run the unchanged attention forward-plus-backward H100 gate after the
  CPU-only NVCC smoke passes.
