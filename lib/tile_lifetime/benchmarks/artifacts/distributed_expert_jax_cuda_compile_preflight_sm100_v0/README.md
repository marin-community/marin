# Distributed expert JAX CUDA compile/register preflight

The CPU-only Iris job `/dlwh/shuttle-moe-ffi-cpu-preflight` compiled all five
generated typed-FFI families for `sm_100a`, loaded every DSO with `ctypes`,
resolved each handler symbol, and registered each target with JAX typed FFI.
The successful task completed in 39.61 seconds. It requested two CPUs, 8 GB of
memory, 20 GB of disk, and no accelerator. The script does not call
`jax.devices()` or execute a kernel.

The preflight uses the package versions pinned in the repository lock: JAX and
jaxlib 0.10.1, CUDA NVCC/CRT/NVVM 13.2.78, NVRTC 13.0.88, runtime 13.0.96,
CCCL 13.3.3.4.1, and cuBLAS 13.4.1.1. A preceding diagnostic run allowed
NVCC's dependencies to float and failed because the compiler emitted PTX 9.3
for a PTX 9.2 assembler. Pinning the complete locked compiler stack fixed that
environment error. The generated sources did not change between attempts.

`result.json` records source and DSO hashes, the compile command template,
dependency versions, successful load/registration states, and the no-device
boundary. `sources/manifest.json` maps the five exact generated sources to their
semantic digests and typed-FFI targets.

This artifact proves host-side source compilation, linking, loading, and target
registration. It does not prove device compilation at JAX runtime, execution,
numerical correctness, determinism, or latency.
