# Streaming-attention training H100 fixed-cause gate

This artifact preserves the environment preflight and the only authorized
fixed-cause H100 validation of Shuttle's generated streaming-attention training
boundary at source revision `e23b25dcdb`. No tile, kernel, schedule, shape, or
benchmark parameter changed from the statically reviewed plan.

## CPU preflight

Iris job `/dlwh/shuttle-attn-environment-preflight-v1-20260809` succeeded in
10.72 seconds with zero failures or preemptions. It resolved and imported:

* JAX, JAXLIB, CUDA plugin, and PJRT 0.10.1;
* Torch 2.11.0+cu130;
* Triton 3.6.0;
* NVCC/PTXAS 13.3.73 and CUDA runtime 13.0.96.

The preflight verified the wheel CUDA paths, runtime header/library, natural
JAX export and recovery, three generated AOT kernel plans, result ABI, and the
requested `(3,1,2,0)` layouts. It did not compile a host CUDA translation unit,
which was the missing check exposed by the H100 run.

## H100 result

Iris job `/dlwh/shuttle-attn-training-h100-fixed-20260809` requested one
preemptible H100, one CPU, 32 GB host memory, 50 GB disk, batch priority,
zero retries, and a 3,600-second timeout. It terminal-failed after 16.89 seconds
with zero preemptions before correctness or timing execution.

The generated forward, dQ, and dK/dV Triton AOT sources compiled far enough to
enter the final NVCC typed-FFI handler build. CUDA's `cuda_fp16.h` then included
`<nv/target>`, but the repository-direct `jax[cuda13]==0.10.1` dependency set
did not install the CCCL headers:

```text
fatal error: nv/target: No such file or directory
```

This is dependency/bootstrap evidence, not a kernel correctness or performance
result. The valid PyPI dependency is the unqualified `nvidia-cuda-cccl`
package; successful neighboring artifacts record version 13.3.3.4.1. It is
not present in the current repository GPU dependency set or lock. A future run
must first add and pin that dependency, then pass a CPU NVCC header smoke. The
no-retry rule was respected and no GB200 replay was launched.

The pod selector returned no resources after the terminal state, proving the
H100 allocation was released. The combined O+dQ+dK+dV boundary, even when it
runs, is component parity evidence only: a real train step needs an early
generated forward producing O and generic Fold state, followed later by a
generated reverse that consumes saved state or recomputes after dO exists.

The minimal submission bundle ended at `f86737f3deea6302003804c8a2de2abb150f4d62`
with tree `270c8349d1660bdadfec482417b2d8ac98e890cf`. Exact scripts and raw Iris logs
are preserved here.
