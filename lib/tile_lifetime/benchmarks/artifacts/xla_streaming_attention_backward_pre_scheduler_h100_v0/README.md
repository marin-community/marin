# Natural-JAX streaming-attention reverse replacement on H100

This artifact is the first live GPU proof that Shuttle replaced a natural,
JAX-differentiated streaming-attention reverse entry inside XLA rather than
only executing the generated handler through a direct benchmark wrapper.

The source is ordinary JAX tensor algebra for causal BF16 grouped-query
attention. JAX owns automatic differentiation. The configuration is batch 1,
sequence 2,048, 32 query heads, eight K/V heads, and head dimension 128.
Shuttle revision `ab6c6493f14f447cb723640dd04a4a7bfa0a5562` recovered five
generic Contracts, four additive Folds, the normalized-exponential state, and
the causal DomainRestriction. It rewrote the complete differentiated entry at
XLA `PRE_SCHEDULER` to one typed-FFI custom call.

The callback proof is preserved in three independent forms:

- `source-vjp-stablehlo.mlir.bc` is the portable natural-JAX differentiated
  program.
- `original-pre-scheduler-hlo.*` and `transformed-pre-scheduler-hlo.*` are the
  exact before/after GPU HLO modules. The transformed text contains exactly one
  Shuttle target.
- `result.json` records 160 executions of the generated typed-FFI handler.

Thirty counterbalanced samples contain five executions per sample:

| Path | Median latency |
| --- | ---: |
| Stock natural JAX/XLA reverse | 4.466521 ms |
| Shuttle `PRE_SCHEDULER` replacement | 0.880135 ms |
| Shuttle / stock JAX/XLA | 0.197052x |

This is an integration comparison against stock XLA for the natural semantic
boundary, not the clean-synthesis expert-performance acceptance result. In
particular, it must not be substituted for the separately matched expert
recompute comparison. The closest preserved direct H100 measurement is
0.679229 ms for the handler-only JAX typed-FFI path and 0.634584 ms for the
matched expert recompute oracle, under a prior revision/toolchain. The live
integrated result is therefore approximately 1.30x that direct Shuttle path
and 1.387x the expert number. Those ratios identify the remaining integration
gap but are not a fresh counterbalanced expert comparison.

The transformed HLO has no input copies. Its query cotangent already matches
the root layout, but key and value cotangents each require one explicit output
`copy`: two copies over 8 MiB of payload, or 16 MiB of nominal read-plus-write
traffic. The current generated handler hard-codes contiguous output strides,
so XLA cannot simply assign the physical root layouts to the FFI results. The
smallest next experiment is a physical-layout-native FFI binding: pass the
selected output layouts/strides into generic reverse generation, emit dK and
dV directly in their requested layouts, and verify that both HLO copies erase.
This is preferable to further attention tile tuning until the roughly 0.20 ms
direct-to-integrated gap is isolated.

Maximum absolute error against the unmodified natural JAX executable is
0.03125 for dQ, dK, and dV. Mean absolute errors are at most 0.000146. Both
executables were bitwise stable across the five explicit determinism repeats,
and each implementation produced one output hash across all timing samples.

The GPU was one NVIDIA H100 80GB HBM3 with driver 595.71.05, a 700 W power
limit, and unpinned clocks sampled at 1,830 MHz SM and 2,619 MHz memory. The
allocation requested one H100, two CPUs, 32 GB host memory, and batch priority.
The runtime used Python 3.12.13 and JAX/JAXlib/CUDA plugin 0.11.0. The handler
was linked with NVCC 13.3.73. `result.json` contains the raw timing distribution,
toolchain output, GPU telemetry, revisions, HLO hashes, and generated-source
hashes.

Torch and Triton were absent from the runtime process. The copied generic AOT
input module does contain a top-level `import torch` at line 29, so Torch 2.11
was installed as a build-only dependency even though the generated kernels do
not use a Torch operator. Triton 3.6 was the AOT compiler. This incidental
import edge should be removed separately; it was not changed during the live
replay. `handler-ldd.txt` confirms that the resulting DSO links only CUDA and
system libraries, with no Torch or Triton dependency.

The exact Python argument vector is recorded in `result.json`. The equivalent
invocation was:

```text
PYTHONPATH=lib/tile_lifetime/src /tmp/attn-hlo-venv/bin/python \
  lib/tile_lifetime/benchmarks/xla_streaming_attention_backward_gpu_custom_call.py \
  --repository . \
  --build-directory /tmp/attn-hlo-h100-build \
  --artifact-directory /tmp/attn-hlo-h100-artifact \
  --nvcc /tmp/attn-hlo-venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --sequence 2048 \
  --query-heads 32 \
  --key-value-heads 8 \
  --head-dimension 128 \
  --block-m 32 \
  --block-n 32 \
  --num-warps 8 \
  --num-stages 3 \
  --warmups 4 \
  --repeats 30 \
  --iterations 5 \
  --determinism-repeats 5 \
  --shuttle-revision ab6c6493f14f447cb723640dd04a4a7bfa0a5562 \
  --holder-revision eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb \
  --allocation-cpu 2 \
  --allocation-memory 32GB \
  --allocation-disk 50GB \
  --allocation-priority batch
```
