# Event Tensor workload-linkage GB200 replay

This artifact records the first GB200 execution of mechanically derived Event
Tensor plans over real tensor payloads. The run used one low-priority GB200,
one host CPU, JAX 0.10.1, CUDA/NVCC 13.3.73, driver 595.71.05, and `sm_100a`.
The reported hardware string in `result.json` records the observed 1,950 MHz
GPU clock, 3,996 MHz memory clock, and 1,200 W power limit.

The timing boundary is host dispatch through JAX typed FFI and output
completion. Each case has 10 warmups and 30 retained samples. These tiny FP32
reference bodies validate physical event realization, source mutation, and the
Torch-free FFI boundary; they are not throughput comparisons against expert
GEMM or attention kernels.

The replay command was:

```bash
python lib/tile_lifetime/benchmarks/sm100_event_tensor_workload_linkage.py \
  --output /tmp/event-results/event_tensor_workload_linkage.json \
  --build-directory /tmp/event-build \
  --nvcc /path/to/cuda-13.3/bin/nvcc \
  --architecture sm_100a \
  --warmups 10 \
  --repeats 30
```

Results:

| Case | Median | Mean | Maximum absolute error | Deterministic |
| --- | ---: | ---: | ---: | --- |
| Segmented Contract | 0.112208 ms | 0.112420 ms | 0 | yes |
| Segmented Contract relation mutation | 0.121696 ms | 0.119842 ms | 0 | yes |
| Streaming Contract/Fold | 0.122144 ms | 0.122180 ms | 2.384e-7 | yes |
| Streaming Contract/Fold depth/partition mutation | 0.121584 ms | 0.120498 ms | 1.192e-7 | yes |

`result.json` contains all raw samples, typed-FFI signatures, HLO custom-call
records, source and plan hashes, event-realization audits, correctness errors,
and deterministic output hashes. `generated/` preserves each emitted CUDA
source. No compiled binary is checked in.
