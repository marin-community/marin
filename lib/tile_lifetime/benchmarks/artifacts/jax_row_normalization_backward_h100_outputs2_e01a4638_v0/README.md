# H100 row Fold output-schedule comparison

This artifact records the single authorized H100 validation of the generic
coalesced AxisFold schedule with one versus two feature outputs per logical
group. The two-output candidate is a negative result: it is `1.352718x` slower
than the one-output schedule. The one-output schedule remains the selected
candidate.

| Candidate | Median latency | Minimum latency | Ratio to matched XLA |
| --- | ---: | ---: | ---: |
| One output per group | `0.089399 ms` | `0.089195 ms` | `1.466627x` |
| Two outputs per group | `0.120932 ms` | `0.120812 ms` | `1.983933x` |
| Matched XLA | `0.060956 ms` | `0.060643 ms` | `1.000000x` |

The one-output result is `0.314%` faster than the earlier `0.089681 ms`
measurement, which is consistent with reproducing that baseline. Neither
generated candidate approaches the `1.20x` acceptance threshold. No further
tile, block, or output-count tuning followed.

## Workload and measurement

The ordinary JAX function uses 2,048 rows, hidden size 4,096, BF16 inputs and
outputs, and an uncentered second-moment normalization backward. JAX owns AD.
StableHLO recovery erases the named semantics into three generic Fold stages.
Both physical candidates use the `coalesce_compatible_row_stages` pipeline,
256 threads, 32 groups per block, and eight reduction lanes per group.

The benchmark contains 30 counterbalanced samples per candidate, 100 iterations
per sample, ten warmups, and all six permutations of the three candidates. The
raw distributions and execution order are in `summary.json`. Timings are host
enqueue intervals followed by `jax.block_until_ready`.

## Correctness and code-generation gates

Both generated candidates produced the same deterministic output hashes:

```text
8e1957ee64710916f839397d1a2489910223006335999a94edc30cb5cca4051a
8bfb71bac6c3b6516379b66fa399a2ea65500cf94591aedbd2811a2422939a0d
```

Each candidate executed its typed-FFI handler exactly 3,012 times. Against both
the matched and independently evaluated natural JAX VJP, the input cotangent
has maximum/mean absolute error `0.0078125` / `1.9742053e-08`; the feature-scale
cotangent has `0.00390625` / `9.5367432e-07`. The numerical policy is
`allow_rounding_reorder`: each generated Fold uses a deterministic tree, but
source-order equivalence with XLA's reduction tree is not claimed.

The candidates have identical semantic fingerprints and different generated
source hashes:

| Outputs per group | Generated source SHA-256 |
| ---: | --- |
| 1 | `cda342a8445c934f30b8bb829b7aeb708016d587fa203e05358c1c7029cebe0f` |
| 2 | `129a8586de148ce8159cccff4e8ce7d6637f1944249a581cc6f50de13139edc3` |

These are the code generator's hashes of the source strings. The benchmark
writer appends one trailing newline when it stores each `.cu` file, so the raw
file hashes in `SHA256SUMS` intentionally differ.

The two-output schedule preserves the same per-column eight-lane deterministic
FP32 reduction tree. It uses two independent accumulators and doubles the small
feature-gradient shared scratch. The generated source is Torch-free and uses no
reduction atomic RMW. Its only `std::atomic` is the typed-FFI invocation counter.

## Environment and resource cleanup

The run used one NVIDIA H100 80GB HBM3 at 700 W, SM clock 1,830 MHz, and memory
clock 2,619 MHz. Driver version is 595.71.05. JAX, jaxlib, CUDA plugin, and PJRT
are 0.10.1; NVCC is 13.3.73.

Iris job `/dlwh/shuttle-row-fold-output2-validation-h100-20260809` succeeded
once with zero failures, preemptions, and retries. The exact task-label pod was
absent after evidence retrieval. The direct batch job created no `dev_gpu`
holder session.
