# Corrected JAX-owned RMS reverse Fold profile on H100

This artifact profiles the generic generated row-normalization reverse Fold at
`rows=2048`, `hidden=4096` on one H100. Ordinary JAX owns AD. Shuttle imports
the exported StableHLO, erases the RMS name, and generates deterministic row
and column Fold programs through Torch-free CUDA typed FFI.

The benchmark functions take all arrays as runtime arguments. The optimized
HLO files contain parameters and executable reductions rather than constants.
This corrects the closed-over-input bug documented in the adjacent invalid
artifact and withdraws the earlier H100 `1.955629x` and GB200 `0.913018x`
performance ratios. Their correctness and determinism evidence remains valid.

## Results

Thirty counterbalanced samples with 100 iterations per sample give:

| Boundary | Generated | Matched XLA | Generated / XLA |
| --- | ---: | ---: | ---: |
| Full reverse | 0.072270 ms | 0.072500 ms | 0.996827x |
| Input-cotangent row Fold | 0.041130 ms | 0.053499 ms | 0.768795x |
| Feature-scale column Fold | 0.031282 ms | 0.033126 ms | 0.944353x |

The separately generated components are bitwise identical to the corresponding
outputs of the full generated handler. Repeated component and full-handler
hashes are stable.

Against the matched explicit FP32 algebra, maximum/mean errors are:

| Cotangent | Maximum | Mean |
| --- | ---: | ---: |
| Input | `9.536743e-7` | `1.311717e-9` |
| Feature scale | `3.051758e-5` | `4.385423e-6` |

The natural BF16 JAX VJP remains a diagnostic because XLA may select a
different reduction tree and cast order. Its maximum differences after the
source-dtype cast are `0.0625` for the input cotangent and `1.0` for the
feature-scale cotangent. The accepted policy is `deterministic_tree`, not
source-order equivalence.

## HLO finding

The corrected full XLA module uses two physical fusions:

1. a Triton row reduction for the input-cotangent correlation;
2. an input fusion that produces the input cotangent and performs the
   feature-scale column reduction together.

The isolated input baseline instead uses the row reduction followed by a loop
fusion, while the isolated feature baseline is one input-reduction fusion.
Thus XLA does combine the output Map and feature Fold in the full program. The
generated full path remains competitive without workload-specific code, but
this is a concrete generic multi-output Fold/Map fusion opportunity.

See `summary.json` for every raw sample, execution order, hash, correctness
metric, semantic fingerprint, and handler count. `xla/` contains optimized HLO
for the full and isolated matched baselines.
