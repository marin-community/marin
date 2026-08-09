# Generated Contract/Map H100 fixed-layout replay

## Result

The generated two-Contract scalar-Map training component is correct after
converting recovered minor-to-major layouts to the major-to-minor order expected
by JAX typed FFI. The single fixed replay used source revision
`239372d31dc8e52b63c42c0321f38483e6d66034`.

The generated two-call path measures `0.05889887 ms`; matched natural JAX
forward plus `jax.vjp` measures `0.03779946 ms`. The ratio is `1.558193x` over
30 counterbalanced samples with 1,000 iterations per sample. This misses the
1.20 component target and remains performance-unaccepted.

This is a standalone `8 x 32`, rank-128 component. It does not measure or
validate the 23-call natural-Grug composition.

## Correctness

The generated output is bitwise equal to natural JAX. Maximum/mean errors are:

| Output | Natural JAX maximum | Natural JAX mean | Ordered CPU maximum |
| --- | ---: | ---: | ---: |
| input adjoint | 0.0009765625 | 0.0000694394 | 0 |
| first weight adjoint | 0.0001220703 | 0.0000135183 | 0 |
| second weight adjoint | 0.0001831055 | 0.0000093663 | 0 |

Both dW buffers use the recovered `{0,1}` minor-to-major physical layout. Their
successful natural-JAX and ordered-CPU comparisons validate the wrapper's JAX
layout conversion.

Three repeated generated executions produced identical hashes for output, dX,
dW0, and dW1. Forward and reverse handler counts are both exactly `30,013`,
matching the fixed protocol.

## Environment and boundary

The request used one H100, one CPU, 32 GB host memory, 50 GB disk, and batch
priority. JAX, JAXLIB, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` are all
0.11.0. CUDA compilation used NVCC 13.3.73 for `sm_90a`.

Compile/link/load preflight passed with fresh handler counts at zero. The source
audit found two generated kernels, explicit BF16 Contract boundaries, generated
forward/reverse Maps, no atomics, and no opaque semantic dependency. JAX owns
AD, and the runtime has no Torch dependency.

The exact source archive SHA-256 is
`d63b9afff367c0345d99fc6d4a68f787cddbd7f07a0619e62a5a4e30c7f2c9ed`.

## Release

The holder job `/dlwh/dev-gpu-dlwh-shuttle-contract-map-chain-h100-239` was
explicitly terminated after evidence copy. The controller reports no active
matching job, the local holder state is absent, and the exact task-label pod
query is empty. No retry, tuning, or additional GPU invocation followed.

