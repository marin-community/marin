# JAX typed-FFI streaming-attention reverse on GB200

This artifact confirms the H100 typed-FFI result on an actual NVIDIA GB200,
compute capability 10.0, at Shuttle revision
`e96c1cfba79b47b0b5a158c08fb969545e6d2726`. It uses the same ordinary JAX
VJP recovery, generic Contract/Fold/Map/DomainRestriction reverse program, and
32x32 schedule as the H100 run. Triton compiled fresh SM100 CUBINs on the GB200;
no H100 binary was reused or relabeled.

The accepted boundary is Q, K, V, and output cotangent to Q, K, and V
cotangents. Both Shuttle and the Flash-SDPA expert recompute forward output and
log-sum-exp state. The causal BF16 GQA shape is batch 1, sequence 2,048, 32
query heads, eight K/V heads, and head dimension 128.

Thirty counterbalanced samples contain five iterations each:

| Implementation | Median latency |
| --- | ---: |
| Shuttle JAX typed FFI | 0.580810 ms |
| Torch Flash-SDPA recompute oracle | 0.555089 ms |
| Shuttle / oracle | 1.046338x |

The generated output is bitwise deterministic. Maximum absolute errors against
the natural JAX VJP are 0.03125 for dQ, dK, and dV; mean absolute errors are at
most 0.000146. The expert oracle independently agrees with the semantic
reference.

Before the benchmark-only expert oracle is imported, neither Torch nor Triton
is present in the process. The generated handler contains neither dependency,
and the DSO links only CUDA and system runtime libraries. `result.json`
preserves all samples and hashes of the SM100 embedded CUBINs, generated C
launchers, launcher headers, generated handler, final DSO, AOT input sources,
and StableHLO fixture. The small generated interfaces and exact StableHLO input
are copied alongside the result.

The GB200 UUID is `GPU-78bf2ae1-552e-d7f3-cccd-3e522bcb9887`. It used driver
595.71.05, a 1,200 W power limit, and unpinned clocks sampled at 1,950 MHz SM and
3,996 MHz memory. The runtime was JAX/JAXlib 0.11.0, Python 3.12.13, and Triton
3.6.0 as an AOT compiler. The typed-FFI DSO was linked for `sm_100a` with NVCC
13.2.78 and CUDA runtime/CCCL 13.2.86.

The exact benchmark command was:

```text
PYTHONPATH=lib/tile_lifetime/src python \
  lib/tile_lifetime/benchmarks/jax_streaming_attention_backward_ffi_gpu.py \
  --repository . \
  --build-directory /tmp/attn-ffi-gb200-final \
  --nvcc /tmp/cuda132-toolkit/bin/nvcc \
  --architecture sm_100a \
  --json-output /tmp/attn-ffi-gb200-final.json \
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
