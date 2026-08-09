# Event Tensor JAX typed-FFI GB200 replay

This artifact records the Torch-free Event Tensor replay at Shuttle revision
`1a04930ecd4008588e215703688a390042e1b9d4` on one NVIDIA GB200. The batch
holder requested one GPU, one CPU, 32 GB host memory, and 50 GB ephemeral disk.
The observed stack was driver 595.71.05, JAX 0.10.1 with CUDA 13, and NVCC
13.3.73 targeting `sm_100a`.

The runtime-relation primary and mutation both match their source-ordered
references bitwise. The phased primary and mutation have maximum absolute
errors `8.9407e-8` and `1.1921e-7`. All four paths are bitwise deterministic
over five repeated executions.

| Path | Median (ms) | Minimum (ms) | Maximum (ms) |
|---|---:|---:|---:|
| Runtime relation | 0.061314 | 0.059035 | 0.066563 |
| Runtime relation mutation | 0.061152 | 0.059504 | 0.064824 |
| Phased pipeline | 0.169697 | 0.168770 | 0.170714 |
| Phased pipeline mutation | 0.146477 | 0.146363 | 0.146777 |

Each timing distribution contains 30 counterbalanced samples with 100 calls per
sample. `summary.json` preserves the raw samples and execution order. All four
optimized HLO dumps retain the runtime inputs as parameters and contain one
typed-FFI target, zero constant lines, and zero copy lines.

The phased CUDA payload is a scalar reference Contract/Fold pipeline. It checks
EventTensorPlan readiness, generation-safe reuse, JAX ownership, and device
lowering. It does not execute tensor-core QK/PV and is not an attention
performance result. `StreamingAttentionProgram` currently reaches the same
schedule algebra through a structural adapter; exact circular-buffer slot and
reuse derivation still separates that graph from this physical payload.

The holder was released after the artifact was copied. A status check found no
active holder session. The preceding H100 request timed out before admission
after 3,600 seconds and consumed no GPU time.
