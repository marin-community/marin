# Corrected JAX-owned RMS reverse Fold profile on GB200

This artifact profiles Shuttle's generic generated row-normalization reverse
Fold at `rows=2048`, `hidden=4096` on one NVIDIA GB200. Ordinary JAX owns AD.
Shuttle imports the exported StableHLO, erases the RMS name, and generates the
deterministic row and column Fold programs through Torch-free CUDA typed FFI.

All benchmark arrays are runtime arguments. The optimized HLO contains tensor
parameters and executable reductions, rather than constants. This is the
corrected GB200 replay that replaces the withdrawn `0.913018x` comparison in
`../generated_gradient_skeletons_gb200_v2`.

## Results

Two independent captures each contain 30 counterbalanced samples with 100
iterations per sample. The first capture reports:

| Boundary | Generated | Matched XLA | Generated / XLA |
| --- | ---: | ---: | ---: |
| Full reverse | 0.101149 ms | 0.114715 ms | 0.881737x |
| Input-cotangent row Fold | 0.093349 ms | 0.104646 ms | 0.892046x |
| Feature-scale column Fold | 0.097220 ms | 0.101979 ms | 0.953333x |

The confirmation capture reports:

| Boundary | Generated | Matched XLA | Generated / XLA |
| --- | ---: | ---: | ---: |
| Full reverse | 0.112671 ms | 0.125017 ms | 0.901247x |
| Input-cotangent row Fold | 0.104491 ms | 0.116372 ms | 0.897903x |
| Feature-scale column Fold | 0.102890 ms | 0.108881 ms | 0.944978x |

Pooling the 60 raw samples gives full generated/XLA medians of
0.107179/0.121351 ms (`0.883215x`). The pooled row-Fold ratio is `0.882168x`;
the pooled column-Fold ratio is `0.946152x`. Both generated components beat
their matched XLA components, and the unfused two-kernel generated full path
also beats XLA's fused full reverse on this configuration.

The separately generated components are bitwise identical to the corresponding
outputs of the full generated handler inside each capture. Repeated executions
with identical runtime buffers have stable hashes.

Against matched explicit FP32 algebra, maximum/mean errors are:

| Cotangent | Maximum | Mean |
| --- | ---: | ---: |
| Input | `9.536743e-7` | `1.303593e-9` |
| Feature scale | `3.051758e-5` | `4.385423e-6` |

The natural BF16 JAX VJP remains an ordering diagnostic. The accepted policy is
`deterministic_tree`, not source-order equivalence.

## Runtime-input and determinism audit

`hlo-runtime-parameter-audit.txt` and `xla/` show that the corrected XLA
modules consume runtime parameters. `summary.json` and
`confirmation-summary.json` preserve every raw sample, execution order,
correctness metric, semantic fingerprint, handler count, and within-capture
determinism hash.

The input-cotangent hash differs between the two independent processes. A
separate two-process diagnostic in `cross-process-runtime-input-hashes.txt`
isolates the difference to the FP32 `inverse_scale` buffer produced by upstream
XLA. The random BF16 inputs and BF16 standardized activation are identical.
The generated handler is bitwise stable for identical runtime buffers, so the
captured hashes are intentionally described as capture-local rather than as a
cross-process source-order guarantee.

## Interpretation

The current generic row and column Fold kernels meet the GB200 performance gate
without a workload-specific fused RMS backward kernel. A generic multi-output
Map/Fold fusion remains a plausible optimization, but it is not needed for
parity at this shape.
