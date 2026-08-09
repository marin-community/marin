# Thirteen-call Grug H100 attribution trace

This artifact profiles, without tuning, the accepted-structure and
unaccepted-performance thirteen-call Grug path at Shuttle revision
`ac34883d030f428b6c3230abca1226ce5e79c23c`. The generated execution remains
`shared_map_fused_reverses`; all thirteen selected targets occur once and every
handler executes three times in the profiling process. The 53-leaf ordered-FP
comparison passes with maximum absolute error `9.760260582e-7`, mean absolute
error `7.977750306e-11`, and 38 bitwise-equal leaves.

The profiler perturbs this sub-millisecond program. Its Python wall timings are
not performance evidence. The performance truth remains the sealed 30-sample
run in
`xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0`: Shuttle
`0.731302 ms`, stock XLA `0.591416 ms`, and a `0.139885 ms` gap. This trace is
used only to attribute that gap.

## Warm GPU timeline

Nsight Systems 2026.1.3 traced one correctness execution and two
counterbalanced measurements of each path. The table below compares the last
measurement of each path, after both executables had already run in the same
process.

| Activity | stock XLA | Shuttle |
| --- | ---: | ---: |
| End-to-end GPU span | 261.151 us | 440.606 us |
| CUDA graph segments | 1 | 9 |
| Direct kernel launches | 0 | 28 |
| Generated GEMM primitive launches | 0 visible outside graph | 7 / 97.280 us |
| Generated named kernels | 0 visible outside graph | 14 / 60.479 us |
| Residual XLA direct kernels | 0 | 7 / 8.576 us |
| Device-to-device copies | 0 | 2 KiB in 2 copies / 2.112 us |
| CUDA graph activity | 261.151 us | 228.863 us |
| Union of recorded GPU activity | 261.151 us | 396.382 us |
| Inter-segment/unattributed gap | 0 | 44.224 us |

The warm GPU-span delta is `179.455 us`. Categorized activity overlaps by
`0.928 us` at trace boundaries, so additive attribution uses the union of
recorded intervals. Relative to stock XLA's single graph, Shuttle removes
`32.288 us` of graph activity but adds `157.759 us` of generated direct kernel
activity, `10.688 us` of residual direct-kernel/copy activity, and `44.224 us`
of inter-segment gap, less the `0.928 us` overlap.

This says the gap is not primarily a copy/layout-conversion problem. The two
copies cost about two microseconds, and both final HLO modules contain nine
`copy` instructions. The main issue is physical work split across generated
kernel bodies, followed by the loss of one continuous XLA command graph. The
trace cannot compare the kernels inside stock XLA's graph individually because
Nsight recorded CUDA graphs at graph granularity.

## Final optimized HLO

| Count | stock XLA | Shuttle |
| --- | ---: | ---: |
| Instructions | 4,377 | 3,724 |
| `dot` | 62 | 49 |
| `fusion` | 252 | 239 |
| `transpose` | 1,509 | 1,269 |
| `copy` | 9 | 9 |
| custom calls | 6 | 16 |
| cuBLAS custom calls | 6 | 3 |
| Shuttle custom calls | 0 | 13 |

Counts include nested computations in the scheduled final HLO. The transformed
module is smaller and contains fewer dots, fusions, and transposes, but the
thirteen generated calls split execution into nine command-graph regions and
direct launches.

## Recommended next experiment

1. Make one adjacent generic Contract/Map/Fold cluster executable as one
   command-buffer-compatible region, preserving the same semantic ASTs. Measure
   whether the nine graph segments and `44.224 us` gap contract. This is the
   smallest experiment that targets an observed whole-step cost rather than an
   isolated kernel.
2. If the boundary experiment leaves most of the gap, target the generated
   GEMM-heavy shared-Map reverse cluster. Seven GEMM primitive launches consume
   `97.280 us`; attach its compatible Maps/Folds to generic Contract
   preparation/finalization instead of adding workload-specific kernels.
3. Do not optimize the D2D copies first. They account for `2.112 us`, and the
   optimized-HLO copy count is unchanged.

`warm-transformed-timeline.csv` preserves every graph, kernel, and copy interval
used in the attribution. `profile-attribution.json` contains the machine-readable
totals and limitations. The two final optimized HLO modules are retained as
gzip files.

The raw `.nsys-rep` and SQLite export are not committed because profiler reports
can retain environment and process metadata. They are stored locally under a
mode-0700 directory and identified by SHA-256 in
`raw-profiler-checksums.txt`.
