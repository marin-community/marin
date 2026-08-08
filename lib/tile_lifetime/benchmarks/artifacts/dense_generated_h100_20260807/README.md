# Generated dense H100 replay

This checkpoint replays the natural StableHLO dense region with generated
QuACK GEMM programs and the generated SM90 streaming-attention adapter. The
generated path does not select a named Transformer GEMM callback or the
official FA3 oracle.

## Results

| Sequence | RMS placement | Median (ms) | Historical manual oracle (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 2048 | consumer prologue | 1.7253 | 1.4561 | 1.185x |
| 2048 | delayed epilogue | 1.6724 | 1.4561 | 1.149x |
| 4096 | consumer prologue | 3.3922 | 3.0080 | 1.128x |
| 4096 | delayed epilogue | 3.3043 | 3.0080 | 1.098x |

Each JSON file contains 30 raw samples, bitwise repeated-output hashes,
generated source hashes, implementation file hashes, hardware telemetry, and
the selected attention schedule. Component files compare generated tile
programs against hand-authored QuACK callbacks with the same semantics. Every
component comparison is bitwise equal; generated/named median ratios are
within 0.23% at both sequence lengths.

## Provenance

- Shuttle base revision: `4fba36752bdbfd28ad9a0ea8dee121bb382b21c9`
  plus the uncommitted generated dense/attention changes identified by file
  hashes in the JSON.
- QuACK: `84ef91df9bec87c7e4938517234fafb07ef844dd`.
- QuACK FP32 row-scale patch SHA-256:
  `40318b9b390e111c38f4838a50cf8913695c9f94142122b374bf09c220cfd9a1`.
- FlashAttention CuTe helper package: `flash-attn-4==4.0.0b16`, wheel SHA-256
  `857bd84cd5884d41b7096826b31c16c281ddde269760bbd5dfafe19a4639b250`.
- CUTLASS DSL: `4.6.1`.
- Torch: `2.13.0+cu130`.
- CUDA runtime: `13.0`.
- Driver: `595.71.05`.
- GPU: NVIDIA H100 80GB HBM3, 700 W power limit.

## Acceptance caveat

This is a clean physical-lowering result, not yet a clean end-to-end semantic
synthesis result under `clean_synthesis_acceptance.md`. Dense semantic
recovery still uses the named `_find_rms_region` / `_rms_plan` macro rewrite.
RMS must still be erased into generic `Map` / `Fold` / `Contract` structure
before schedule synthesis for the stricter acceptance gate.
