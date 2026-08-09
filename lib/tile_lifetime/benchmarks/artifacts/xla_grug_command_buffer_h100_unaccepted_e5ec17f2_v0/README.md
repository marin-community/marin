# Command-buffer H100 bootstrap failure

This artifact preserves the only authorized H100 invocation of the opt-in
`normalized_exp_pair` command-buffer candidate at Shuttle revision
`e5ec17f21c3375c477ac9c53319cc52d18cfd6d3`. It is an unaccepted
infrastructure result, not a correctness or performance measurement.

The source-only preflight passed 17 focused tests. The in-pod preflight then
confirmed one physical H100, JAX/JAXLIB/CUDA plugin/PJRT 0.11.0, the public HLO
rewrite API, XLA's default command-buffer selection, and capture-safe generated
forward and reverse sources. Both sources carry
`ffi::Traits::kCmdBufferCompatible` and pass the static rejection audit for
scratch/runtime allocation, lazy handles, autotuning, launch-status queries,
and synchronization.

The benchmark process started once with the required
`shared_map_fused_reverses` composition, four warmups, and 30 counterbalanced
samples. It failed during compilation before warmup or measurement because the
fresh environment did not contain Triton:

```text
ModuleNotFoundError: No module named 'triton'
CalledProcessError: python -m triton.tools.compile ... returned non-zero exit status 1
```

The process produced zero timed samples, no correctness or determinism result,
no target-multiplicity result, and no capture-aware handler counts. The
command-buffer candidate is therefore **unaccepted and unmeasured**. The run was
not retried, and no profiler was invoked.

## Corrected pre-allocation prerequisite

Before another allocation, the exact Python environment must prove all of the
following on CPU:

1. JAX, JAXLIB, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` resolve to 0.11.0.
2. `triton.tools.compile` imports and a minimal AOT compile succeeds.
3. The pinned generated-attention dependency set imports together, including
   the Torch/Triton versions used by the existing FA4-derived AOT path.
4. The complete composition performs a compile-only dependency preflight,
   rather than checking only the two newly eligible handlers.

The intended next H100 run remains exactly one four-warmup, 30-sample replay.
It must require correctness, determinism, exact target multiplicity, and
capture-aware handler evidence before accepting any timing.

## Allocation and release

- Allocation: one H100, one requested CPU, 32 GB host memory, 50 GB disk,
  batch priority.
- Physical device: NVIDIA H100 80GB HBM3, driver 595.71.05, 700 W power limit.
- Benchmark start: 2026-08-09T23:08:44Z.
- Compile failure: 2026-08-09T23:10:03Z, exit code 1.
- Final Iris state: killed by explicit release, zero task failures.
- Local session: absent after release.
- Kubernetes pod selector: no resources after release.

`benchmark.log.gz` is the complete captured process log. The two generated CUDA
files are the exact command-buffer candidate sources produced before the
unrelated attention-backward AOT compile failed.

