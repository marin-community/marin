# Same-domain RMS backward Fold coalescing on H100

This artifact compares one bounded schedule choice for the generic AxisFold
reverse pipeline: retain the row second-moment Fold and row correlation Fold as
separate kernels, or coalesce their same-domain reductions and input-cotangent
finalization into one kernel. Both paths were generated from the same ordinary
JAX VJP exported to StableHLO. The two schedules have identical semantic
fingerprints and differ only in schedule realization.

This is a component experiment, not a clean-synthesis acceptance claim. No tile,
thread, column-group, or numerical-policy tuning was performed.

## Result

The run used one physical NVIDIA H100 80GB HBM3, driver `595.71.05`, JAX and
JAXlib `0.10.1`, and generated `sm_90a` code. The Iris reservation requested one
H100, one CPU, 16 GB host memory, and 50 GB ephemeral disk at batch priority.
The process sampled a 700 W power limit, 1830 MHz SM clock, and 2619 MHz memory
clock after measurement.

Each primary sample is the average of 100 executions. Thirty samples cover all
six process-order permutations exactly five times.

| Path | Median latency | Ratio to XLA |
| --- | ---: | ---: |
| Matched natural JAX/XLA VJP | 0.061414 ms | 1.000x |
| Generated separate stages | 0.096221 ms | 1.566758x |
| Generated coalesced row stages | 0.089681 ms | 1.460268x |

Coalescing reduced generated median latency by `6.797%` (`0.932032x` the
separate schedule). It helps this fixed component, but the generated result
remains `46.027%` behind matched XLA.

`summary.json` preserves all 30 samples and execution orders. The sample ranges
were `0.096072-0.097305 ms` for separate stages, `0.089514-0.090871 ms` for
coalesced stages, and `0.061025-0.062074 ms` for XLA.

## Kernel evidence

Nsight Systems captured exactly one invocation of each path inside one CUDA
profiler range. The raw report was copied to
`rms-fold-profile-raw-preserved.nsys-rep` before `nsys stats` exported SQLite
or generated `rms-fold-kernel-summary_cuda_gpu_kern_sum.csv`. The retrieved raw
report has SHA-256
`cc497fe0344de8fc632a6496a7c7e66bf86808fc67d43d6998bed97acfb61f35`.
It is held outside Git because Nsight embedded secret-bearing process
environment records and GitHub push protection correctly rejected the binary.

| Path | Kernels | Summed kernel time |
| --- | --- | ---: |
| Separate | `Kernel0`, `Kernel1`, `Kernel2` | 89.856 us |
| Coalesced | `Kernel0And1`, `Kernel2` | 83.616 us |
| XLA | four fusion launches | 48.608 us |

The coalesced schedule removes one launch and one full read of the row input
from the first two stages. Its summed kernel time fell by `6.944%`, consistent
with the primary end-to-end component result. `rms-fold-kernel-trace.csv`
preserves launch order and per-launch duration; the grouped Nsight export is
also retained.

## Correctness and numerical contract

The accepted numerical policy is `allow_rounding_reorder`. Generated reductions
use deterministic trees; source-ordered equivalence to XLA is not claimed
because XLA may select a different deterministic reduction tree.

Both generated schedules produced the same two output hashes and repeated those
hashes exactly. Against both the matched XLA function and an independent natural
JAX VJP:

- input-cotangent maximum and mean absolute error were `0.0078125` and
  `1.9742053e-8`;
- feature-scale-cotangent maximum and mean absolute error were `0.00390625` and
  `9.536743e-7`.

## Boundary

One Python process compiled and registered two distinct generated typed-FFI
targets plus the matched XLA function. All variants received the same BF16
inputs generated from seed `20260809`. Correctness and two-run determinism ran
before timing. Each generated handler then executed 10 warmups plus 3,000 timed
calls; its total handler count was 3,012 after correctness checks.

Timing starts before enqueueing 100 calls and ends after
`jax.block_until_ready` on the last result, then divides by 100. Thus the number
includes the exact JAX typed-FFI/XLA process boundary and launch overhead, but
not compilation. The profiler used a second process and captured one separate,
one coalesced, and one XLA call in that order.

## Files

- `summary.json`: primary counterbalanced measurements and semantic evidence.
- `profile-summary.json`: profiling-process validation output; its one-iteration
  timing samples are not used for the performance conclusion.
- `measurement/*/generated_axis_fold_ffi.cu`: generated CUDA for both schedules.
- `xla/*.txt`: optimized HLO for the matched XLA path and its components.
- `rms-fold-kernel-summary_cuda_gpu_kern_sum.csv`: grouped Nsight kernel summary.
- `rms-fold-kernel-trace.csv`: ordered launch evidence derived from the raw
  report's `CUPTI_ACTIVITY_KIND_KERNEL` table.
- `reproduction.txt`: reservation and benchmark commands.
- `SHA256SUMS`: publishable artifact checksums excluding itself, nonportable
  `.so` files, and the secret-bearing raw report.
