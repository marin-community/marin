# Generic affine chunk pipeline on H100

This artifact records the first clean StatefulScan result that meets the
prototype performance gate without importing or dispatching through a named
recurrent-model kernel.

The source is an ordinary exported JAX `lax.scan`/StableHLO `while`. Shuttle
recovers a diagonal-plus-low-rank affine state update and lowers its generated
diagonal, left/right, additive, residual-scale, and read factors through three
generic physical stages:

```text
AffineIntraChunkPrepare -> AffineStateScan -> AffineReadout
```

The selected H100 candidate uses 64-token chunks. Preparation generates
transformed factors, forms the within-chunk interaction matrix, applies a
generic 4x16 blocked unit-lower inverse, and solves the right/additive factors.
The scan forwards chunk-start state and solved residual coefficients. Readout
reconstructs each token output from those values. The compiler path does not
call Flash Linear Attention or dispatch on a Gated DeltaNet name.

## Selected result

Primary shape: `B=1, T=2048, H=32, K=128, V=128`, BF16 factors, FP32 state,
scalar diagonal, rank-1 update, chunk size 64, value block 32.

| Stage | Median (ms) |
| --- | ---: |
| AffineIntraChunkPrepare | 0.422912 |
| AffineStateScan | 0.089360 |
| AffineReadout | 0.069696 |
| Combined | 0.579776 |

The pinned external oracle is 0.510624 ms. The generated pipeline is 1.1354x
the oracle and passes the 1.2x gate of 0.612749 ms. All 50 raw measurements are
in `shuttle-affine-split-primary.json`.

The selected result forwards 50,331,648 bytes between scan/readout stages and
materializes 76,021,760 bytes during preparation. Output and final state are
bitwise deterministic across repeated executions. Relative error is not useful
near zero; maximum absolute output and state errors against the compiler-owned
source-ordered recurrent skeleton are 4.8828125e-4 and 3.6969408e-4.

The same emitter was exercised with scalar and per-key diagonals and rank-1
and rank-2 updates. All four cases are finite and bitwise deterministic. The
rank-2 cases use the same split preparation and generic 2x16 blocked inverse,
not a named fallback.

## Preserved negative baselines

| Candidate | Prepare | Scan | Readout | Combined | Oracle ratio | Forwarded bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C16, recomputed factors | 0.142640 | 0.393616 | 0.239712 | 0.775824 | 1.5194x | 150,994,944 |
| C64, naive row inverse | 0.499024 | 0.089264 | 0.069920 | 0.656400 | 1.2855x | 50,331,648 |
| C64, monolithic 4x16 block inverse | 0.599456 | 0.089600 | 0.069936 | 0.757904 | 1.4843x | 50,331,648 |
| C64, K-loop monolithic block inverse | 0.524848 | 0.089376 | 0.070064 | 0.682720 | 1.3370x | 50,331,648 |
| C64, split factors/interactions/inverse | 0.422912 | 0.089360 | 0.069696 | 0.579776 | 1.1354x | 50,331,648 |

The C16 result established that chunk-state traffic dominated after the first
preparation optimization. Moving to the oracle-like 64-token decomposition cut
that traffic by three. Profiling then isolated the monolithic triangular
inverse at 90.3% of preparation time. The final split turns preparation into
separate generic factor-transform, interaction, blocked-inverse, and factor-
solve kernels. The preserved profile reports 289.921 us for factor transform,
75.680 us for the 4x16 inverse, and 29.920 us for the remaining interaction and
right/additive solves in one profiled execution.

## Reproduction

```bash
cd lib/tile_lifetime/benchmarks
PYTHONPATH=../src python h100_affine_chunk_pipeline.py \
  --sequence-length 2048 \
  --heads 32 \
  --key-dimension 128 \
  --value-dimension 128 \
  --chunk-size 64 \
  --block-v 32 \
  --warmups 10 \
  --repeats 50 \
  --shuttle-revision 4fba36752bdbfd28ad9a0ea8dee121bb382b21c9 \
  --json-output shuttle-affine-split-primary.json
```

The exact executed source is under `source/`; the exported semantic fixture is
`natural_scan.stablehlo.mlir.bc`. `SHA256SUMS` covers all preserved files.

