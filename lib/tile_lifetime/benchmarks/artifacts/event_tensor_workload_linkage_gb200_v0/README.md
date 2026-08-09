# Event Tensor workload-linkage GB200 replay

This artifact records a hardened GB200 execution of mechanically derived Event
Tensor plans over real tensor payloads. The generated source revision is
`57cca04dc7fb4b58daee044e558f1476aa624cc2`, with a clean worktree.

Device and allocation provenance:

| Field | Value |
| --- | --- |
| GPU | NVIDIA GB200 |
| GPU UUID | `GPU-0f80290e-c058-9dc3-c1d6-7dca086f0c79` |
| Compute capability | 10.0 |
| Driver | 595.71.05 |
| Compile target | `sm_100a` |
| Observed/max SM clock | 1,950 / 2,062 MHz |
| Observed/max memory clock | 3,996 / 3,996 MHz |
| Power policy | P0, persistence enabled, 1,200 W limit |
| Resource request | 1 GB200, 1 CPU, 32 GB host memory, 50 GB disk, batch priority |

JAX, JAXlib, the CUDA plugin/PJRT, NVCC, and every installed NVIDIA CUDA
component are versioned individually in `result.json`.

The timing boundary is host dispatch through JAX typed FFI and output
completion. Each case has 10 warmups and 30 retained samples. These are small
FP32 generated payload kernels. They validate real tensor/CSR consumption,
physical Event Tensor realization, source mutation, and the Torch-free FFI
boundary. They do not establish expert grouped-GEMM throughput or tensor-core
streaming-attention throughput.

The replay command was:

```bash
python lib/tile_lifetime/benchmarks/sm100_event_tensor_workload_linkage.py \
  --output /tmp/event-results/event_tensor_workload_linkage.json \
  --build-directory /tmp/event-build \
  --nvcc /path/to/cuda-13.3/bin/nvcc \
  --architecture sm_100a \
  --warmups 10 \
  --repeats 30 \
  --requested-gpu-model GB200 \
  --requested-gpu-count 1 \
  --requested-cpu 1 \
  --requested-host-memory-gb 32 \
  --requested-disk-gb 50 \
  --requested-priority batch
```

Results:

| Case | Median | Mean | Maximum absolute error | Deterministic |
| --- | ---: | ---: | ---: | --- |
| Segmented Contract | 0.073328 ms | 0.076578 ms | 0 | yes |
| Segmented Contract relation mutation | 0.072800 ms | 0.075187 ms | 0 | yes |
| Streaming Contract/Fold | 0.074672 ms | 0.078443 ms | 2.384e-7 | yes |
| Streaming Contract/Fold depth/partition mutation | 0.074224 ms | 0.075200 ms | 1.192e-7 | yes |

`result.json` contains all raw samples, typed-FFI signatures, HLO custom-call
records, source and plan hashes, event-realization audits, correctness errors,
and deterministic output hashes. `generated/` preserves each emitted CUDA
source. No compiled binary is checked in.

`SHA256SUMS` covers this README, `result.json`, and all four generated CUDA
sources. Verify it from this directory with `sha256sum -c SHA256SUMS`.
