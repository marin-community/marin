# Physical-layout-native attention reverse on H100

This artifact measures one bounded attempt to remove the two output copies from
Shuttle's natural-JAX streaming-attention reverse replacement. The source is
ordinary JAX tensor algebra for causal BF16 grouped-query attention. JAX owns
automatic differentiation. Shuttle recovers the generic Contract, Fold, Map,
and DomainRestriction structure, then replaces the complete reverse entry at
XLA `PRE_SCHEDULER` with one generated typed-FFI target.

The configuration is batch 1, sequence 2,048, 32 query heads, eight K/V heads,
and head dimension 128 on one NVIDIA H100 80GB HBM3. The allocation requested
one H100, two CPUs, 32 GB host memory, and batch priority. Driver 595.71.05 and
NVCC 13.3.73 were used; the GPU reported a 700 W power limit and unpinned
1,830 MHz SM / 2,619 MHz memory clocks around the replay.

## Result

The compiler derives each result buffer's physical permutation from the exact
captured HLO and specializes the generated output strides. The default
contiguous FFI contract would require copies for dK and dV. The new contract
emits all three cotangents directly in their requested layouts:

| Output | XLA minor-to-major | Logical strides `(B,S,H,D)` |
| --- | --- | --- |
| dQ | `{3,2,1,0}` | `(8388608,4096,128,1)` |
| dK | `{3,1,2,0}` | `(2097152,128,262144,1)` |
| dV | `{1,3,2,0}` | `(2097152,1,262144,2048)` |

The transformed HLO contains zero Shuttle boundary copies, down from two over
8 MiB of payload (16 MiB nominal read-plus-write traffic). Thirty balanced
samples contain five executions per sample:

| Path | Median latency |
| --- | ---: |
| Stock natural JAX/XLA reverse | 4.433139 ms |
| Direct layout-native typed FFI | 0.838209 ms |
| Layout-native PRE_SCHEDULER replacement | 0.829328 ms |

The previous canonical-output integrated result was 0.880135 ms, so removing
the copies improves the integrated boundary by about 5.8% (0.050807 ms). This
does not recover the full prior direct-to-integrated difference. The closest
preserved canonical-output direct result is 0.679229 ms, implying that direct
noncanonical dK/dV stores cost about 0.159 ms under the prior measurement. The
closest preserved matched expert recompute result is 0.634584 ms; the new
integrated result is approximately 1.307x that oracle. Since those comparison
numbers came from a prior revision/toolchain rather than this counterbalanced
process, they diagnose the remaining gap but are not a fresh acceptance replay.

Physical-layout-native generation is therefore a legal plan candidate, not a
universal rule. The planner should compare it against canonical output plus a
copy or against a consumer-negotiated layout.

## Bounded tile-store experiment

`bounded_coalesced_store/` preserves one generic alternative. It replaces the
explicit arbitrary-stride output stores with layout-selected Triton block
pointers and transposes the register tile when sequence is the minor physical
axis. It remains correct and copy-free, but regresses the integrated median to
0.868246 ms, 4.7% slower than the selected explicit-stride candidate. The
source and full 30-sample distribution are preserved; the candidate is not in
the active implementation.

## Correctness and audit

Direct and integrated layout-native executions are bitwise identical. Each
path is bitwise stable across five explicit determinism repeats and all timing
samples. Against stock natural JAX, maximum absolute error is 0.03125 for dQ,
dK, and dV; mean absolute error is at most 0.000146.

The runtime process imported neither Torch nor Triton. Torch 2.11 and Triton
3.6 remained build-only dependencies of the AOT generator; the generated DSO
has no Torch or Triton linkage. `handler-ldd.txt`, `python-packages.txt`, the
generated CUDA handler, all three AOT C sources, before/after HLO, raw timing
distributions, telemetry, and exact revisions are included. The active source
revision is `c6a4244052a2ca808c70b3402ba6dbadbf9f5c1f`.

The selected result was produced with:

```text
PYTHONPATH=lib/tile_lifetime/src /tmp/attn-layout-venv/bin/python \
  lib/tile_lifetime/benchmarks/xla_streaming_attention_backward_gpu_custom_call.py \
  --repository . \
  --build-directory /tmp/attn-layout-h100-build \
  --artifact-directory /tmp/attn-layout-h100-artifact \
  --nvcc /tmp/attn-layout-venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --sequence 2048 --query-heads 32 --key-value-heads 8 --head-dimension 128 \
  --block-m 32 --block-n 32 --num-warps 8 --num-stages 3 \
  --warmups 6 --repeats 30 --iterations 5 --determinism-repeats 5 \
  --shuttle-revision c6a4244052a2ca808c70b3402ba6dbadbf9f5c1f \
  --holder-revision eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb \
  --allocation-cpu 2 --allocation-memory 32GB --allocation-disk 50GB \
  --allocation-priority batch
```
