# Dense clean-synthesis counterbalanced H100 checkpoint

This artifact closes the statistical acceptance gap for the natural-source
dense proof. The generated path is:

```text
ordinary JAX
  -> StableHLO
  -> 36 generic Map / Contract / Fold / DomainRestriction operations
  -> generic eight-skeleton plan
  -> generated Contract ASTs and generated SM90 streaming attention
```

It does not select named Transformer epilogues or an official FlashAttention
entry point. The hand-composed named QuACK/CODA plus FlashAttention-4 CuTe path
is loaded only as the expert oracle.

## Counterbalanced results

Each implementation has two independent captures with ten warmups and 30
steady-state samples of ten complete-region iterations. Run 1 launches the
generated process before the oracle process; run 2 reverses that order. The
ratios use pooled 60-sample medians.

| Sequence | Generated policy | Pooled median | Oracle median | Ratio |
|---:|---|---:|---:|---:|
| 2,048 | source-ordered prologue | 1.705818 ms | 1.523838 ms | 1.119422x |
| 2,048 | delayed epilogue | 1.650502 ms | 1.523838 ms | 1.083122x |
| 4,096 | source-ordered prologue | 3.478322 ms | 3.253411 ms | 1.069131x |
| 4,096 | delayed epilogue | 3.390837 ms | 3.253411 ms | 1.042240x |

Both policies pass the 1.20-times completion gate at both required sequence
lengths. Delayed scaling passes the 1.10-times stretch target at both shapes;
source-ordered preparation passes stretch at 4,096 and misses it at 2,048.

## Numerical and synthesis evidence

Every generated output repeats bitwise within and across captures. The two RMS
policies differ by at most 0.03125 in `x2` and next-QKV, consistent with their
declared source-ordered versus real-algebra-equivalent contracts. The earlier
component audit compares the direct generated scalar-AST SiLU expression with
the named fast-math oracle and records maximum BF16-rounded error 0.125.

The mutation from `SiLU(left) * right` to `left * right` uses the same semantic
erasure report, planner, Contract skeleton, and scalar-AST generator. Generated
source manifests under `raw/*-generated-source` show that no named QuACK
Transformer callback or official FA3 forward is selected.

## Evidence layout

- `raw/run*-s*-generated.json` contains generated samples, output hashes,
  environment telemetry, and exact generated-source identities.
- `raw/run*-s*-oracle.json` contains the hand-composed expert samples and
  deterministic output hashes.
- `pooled-summary.json` computes the frozen pooled medians and ratios.
- `manifest.json` records the matched boundary, dependency classification, and
  source revisions.
- `SHA256SUMS` covers the complete artifact.

GPU clocks were not locked. End-of-capture SM clocks range from 1,425 to 1,695
MHz, while memory remained at 2,619 MHz. Counterbalanced process ordering and
pooled distributions limit order bias; every raw telemetry value is retained.
