# Generated Contract/Map H100 component replay

## Result

The single authorized H100 replay failed its natural-JAX correctness guard
before timing. Generated `input_adjoint` differs from `jax.vjp` by maximum
absolute error `0.47265625` and mean absolute error `0.1155403256`; the fixed
limits are `0.0078125` and `0.0005`.

This is a component failure, not a latency result. The run produced no
counterbalanced samples, determinism hashes, final handler counts, ordered-CPU
input-adjoint comparison, or dW parity comparison. It makes no whole-Grug claim.

## Fixed boundary

The measured source is
`d9e8990e878def454d683bc7ae36d0ed39510c25`. The benchmark harness, generated
Contract/Map implementation, JAX typed-FFI wrapper, recovery code, and frozen
HLO fixture are byte-identical to `bcafcc5ab13677146af36644b4f12b008790b676`.
The source archive SHA-256 is
`a5243468bedd30b69b3aa99146bfbed2f53e8d30fbfbaf64f96d6c53704ef84b`.

The request used one H100, one CPU, 32 GB host memory, 50 GB disk, and batch
priority. JAX, JAXLIB, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` are all
0.11.0. CUDA compilation used NVCC 13.3.73 for `sm_90a`.

Compile/link/load preflight passed. Both fresh handler counters were zero, the
source audit found two kernels, no atomics, explicit BF16 Contract boundaries,
generated forward and reverse Maps, and no opaque semantic dependency.

The generated dW stores encode both recovered `{0,1}` minor-to-major layouts.
The GPU produced the dW buffers, but the harness checks outputs in order and
aborted on `input_adjoint`; dW parity therefore remains unvalidated.

## Release

The holder job `/dlwh/dev-gpu-dlwh-shuttle-contract-map-chain-h100-d9` was
explicitly terminated after copying the evidence. The controller reports no
active matching job, the local holder state is absent, and the exact task-label
pod query is empty.

No retry, tuning change, or diagnostic GPU invocation followed the failure.

