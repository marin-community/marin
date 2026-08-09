# JAX typed-FFI streaming-attention reverse on H100

This artifact measures Shuttle revision
`e96c1cfba79b47b0b5a158c08fb969545e6d2726` on one NVIDIA H100 80GB
HBM3. Ordinary JAX owns automatic differentiation. Shuttle imports the
serialized JAX VJP, recovers generic Contract, Fold, Map, and
DomainRestriction structure, applies the normalized-exponential maximum-VJP
rewrite under `allow_rounding_reorder`, and emits a typed-FFI execution path.

The accepted boundary is Q, K, V, and output cotangent to Q, K, and V
cotangents. Neither implementation receives saved forward output or log-sum-exp
state. Shuttle and the Flash-SDPA oracle both recompute that state from Q, K,
and V before executing the reverse pass.

The causal BF16 GQA shape is batch 1, sequence 2,048, 32 query heads, eight
K/V heads, and head dimension 128. Thirty counterbalanced samples contain five
iterations each:

| Implementation | Median latency |
| --- | ---: |
| Shuttle JAX typed FFI | 0.679229 ms |
| Torch Flash-SDPA recompute oracle | 0.634584 ms |
| Shuttle / oracle | 1.070353x |

This comparison is deliberately different from the previously sealed
0.462000 ms Flash-SDPA number. That number measures backward from saved forward
state. It is a useful lower-bound component result, but it is not the natural
Q/K/V/output-cotangent boundary measured here.

The generated output is bitwise deterministic across repeated execution.
Maximum absolute errors against the natural JAX VJP are 0.03125 for dQ, dK,
and dV; mean absolute errors are at most 0.000146. The independently evaluated
Flash-SDPA oracle also agrees with the JAX semantic reference.

Triton 3.6.0 is used only as an AOT build compiler. The generated C launchers
embed CUBINs and are linked into the typed-FFI DSO as separate translation
units. Before the benchmark-only expert oracle is imported, neither Torch nor
Triton is present in the process. The generated handler source contains neither
name, and `ldd` reports only CUDA and system runtime libraries. The benchmark
observed 157 typed-FFI handler invocations.

`result.json` preserves every counterbalanced sample, the semantic fingerprint,
the StableHLO fixture digest, the generated handler and DSO digests, the AOT
input-source and launcher-header digests, each emitted C launcher digest, and
the SHA256 of each embedded CUBIN. `source_vjp_stablehlo.mlir.bc` is the exact
portable StableHLO input. `generated_handler.cu` and `launchers/` preserve the
small compiler-generated interfaces; the large generated C files are identified
by their exact commands and hashes in `result.json`.

The H100 used driver 595.71.05, a 700 W power limit, and unpinned clocks sampled
at 1,830 MHz SM and 2,619 MHz memory. The runtime was JAX/JAXlib 0.11.0 and
Python 3.12.13. Triton produced the embedded CUBINs for the detected H100. The
typed-FFI DSO was linked with NVCC 13.2.78 and CUDA runtime/CCCL 13.2.86.

The exact benchmark command was:

```text
PYTHONPATH=lib/tile_lifetime/src python \
  lib/tile_lifetime/benchmarks/jax_streaming_attention_backward_ffi_gpu.py \
  --repository . \
  --build-directory /tmp/attn-ffi-h100-final \
  --nvcc /tmp/cuda132-toolkit/bin/nvcc \
  --json-output /tmp/attn-ffi-h100-final.json \
  --sequence 2048 \
  --query-heads 32 \
  --key-value-heads 8 \
  --head-dimension 128 \
  --oracle torch_flash_recompute \
  --repeats 30 \
  --iterations 5 \
  --warmups 5 \
  --shuttle-revision e96c1cfba79b47b0b5a158c08fb969545e6d2726
```
