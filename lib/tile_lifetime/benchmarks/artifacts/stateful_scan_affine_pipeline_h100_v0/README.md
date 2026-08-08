# Generated affine StatefulScan H100 checkpoint

This checkpoint records a natural JAX `lax.scan` exported as StableHLO,
recovered as a generic diagonal-plus-low-rank affine `Scan`, and lowered to
three generated physical stages:

1. `AffineIntraChunkPrepare`
2. `AffineStateScan`
3. `AffineReadout`

The generated path does not import or invoke FLA, FlashQLA, Gated DeltaNet, or
another complete recurrent kernel. Pinned FLA is loaded only by the benchmark
harness as the matched expert oracle.

## Legacy single-capture matched boundary

Both implementations receive the same BF16 `q`, `k`, and `v`, FP32
`log_decay`, `beta`, and initial state. Both return the sequence output and
final FP32 state. Q/K normalization is disabled and the query scale is exactly
one on both paths. Fifty samples are interleaved with alternating launch order
after ten warmups.

Configuration: H100 80GB HBM3, driver 595.71.05, Torch 2.8.0+cu128, Triton
3.4.0, batch 1, sequence 2048, 32 heads, key/value dimensions 128, rank-one
update, scalar decay, chunk 64, value block 32.

| Implementation | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Generated Shuttle | 0.466752 ms | 0.457216 ms | 0.471424 ms |
| FLA `9c8e42e` oracle | 0.420528 ms | 0.395712 ms | 0.459552 ms |

The generated/oracle ratio is **1.1099x**, within the 1.2x clean-synthesis
target.

## Accepted counterbalanced boundary

Two additional independent captures use the same H100 holder, pinned FLA
revision, natural boundary, and Torch 2.8/Triton 3.4 toolchain. Run one starts
with generated and run two starts with oracle. Within each run, all ten warmup
pairs and all 50 measured pairs alternate launch order.

| Capture | Generated | FLA oracle | Ratio |
| --- | ---: | ---: | ---: |
| Generated-first | 0.466096 ms | 0.425392 ms | 1.095686x |
| Oracle-first | 0.465184 ms | 0.423664 ms | 1.098002x |
| Pooled 100 samples | 0.465824 ms | 0.424304 ms | 1.097854x |

The pooled oracle median freezes completion/stretch thresholds of
0.509165/0.466734 ms. The pooled generated result passes both. Output
maximum/mean errors are
`4.8828125e-4`/`5.270477e-5`; final-state maximum/mean errors are
`3.154259e-4`/`4.448347e-5`. Both captures preserve StableHLO hash
`417852499eed3f1dcc4b270d73c3922b1c0a5e5071951c78879319d41a65730a`.
Raw records, exact launch orders, and the holder environment are under
`raw/counterbalanced/`; `pooled-counterbalanced.json` is the summary.

Maximum/mean absolute differences against FLA are
`4.8828125e-4`/`5.270477e-5` for the output and
`3.154259e-4`/`4.448347e-5` for the final state. The generated output and state
repeat bitwise. The same generator also passes scalar/per-key decay crossed
with rank-one/rank-two update mutations; every mutation repeats bitwise.

## Optimization result

The first four-block triangular inverse reduced combined latency against the
historical generic path but left factor preparation at 0.407 ms. Profiling
showed that the factor transform repeated the 64-token diagonal-prefix work
for four 32-wide K tiles. Increasing the generic K tile to 64 reduced
preparation to 0.281 ms and combined standalone latency to 0.439 ms. No
recurrence-specific dispatch or semantic change was introduced.

The physical program forwards 50,331,648 bytes between generated stages and
uses a `bounded_reassociation` numerical contract. FLA-inspired source lineage
is limited to the physical chunk decomposition and triangular/contraction
schedule; the generated executor has no FLA import.

The shared Shuttle semantic-erasure validator now runs before physical
candidate execution. Its report is preserved in
`semantic_erasure_report.json`: `stablehlo.while` and its tensor-expression
body lower to `Scan`, `Map`, and `Contract`, while all candidate-selection keys
are derived from ordered extent, state rank, generic primitive arity,
transition structure, and numerical policy. Tests reject named or stale
scheduling keys.

Raw distributions, complete correctness and mutation records, StableHLO, source
hashes, hardware telemetry, and oracle revision are under `raw/`.
