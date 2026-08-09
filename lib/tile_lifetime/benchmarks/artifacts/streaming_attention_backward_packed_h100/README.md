# Packed-query-head streaming backward on H100

This artifact compares the scalar-head and packed-query-head dQ schedules for
the generated streaming normalized-exp reverse program. It is a schedule
ablation, not a complete clean-synthesis acceptance result.

## Configuration

- GPU: NVIDIA H100 80GB HBM3
- Driver: 595.71.05
- Power limit: 700 W
- Recorded SM/memory clocks: 1,830/2,619 MHz
- CUDA reported by Torch: 12.8
- Torch: 2.8.0+cu128
- Triton: 3.4.0
- Shape: batch 1, sequence 2,048, 32 query heads, eight K/V heads, dimension 128
- Semantics: causal BF16 GQA backward
- Physical configuration: 32-by-32 tiles, eight warps, three stages
- Benchmark: five warmups, 30 counterbalanced samples, five iterations per sample

## Results

| Schedule | Generated median | SDPA median | Generated / SDPA |
|---|---:|---:|---:|
| Scalar-head dQ (`281569d86f`) | 1.297498 ms | 0.464624 ms | 2.792576x |
| Packed dQ (`f0f2aa6b73`) | 0.584992 ms | 0.465133 ms | 1.257688x |

Packing reduced generated latency by 54.91% and produced the same deterministic
SHA-256 output hash as the scalar-head control:
`6957ec539e0a1d6270a1a8d1efa0531a5bec5a624d885dbd8bc51120d723f58d`.
Both paths passed the configured numerical thresholds. The packed result is
outside the 1.20 acceptance gate.

The raw JSON files preserve all 30 generated and oracle measurements. The
Triton metadata under `resources/` reports 114,688 bytes of shared memory for
packed dQ and 45,568 bytes for scalar-head dQ. Both used 256 threads. Triton's
metadata did not expose register or spill counts in this environment.

The prior 0.864582 ms generated and 0.148534 ms SDPA measurements were collected
on GB200. They are retained as historical context and are not used as the H100
schedule control.

## Checksums

```text
6d9d18c13e8653aec8f52529e6b3579922ed72848414f937bef84805bbfba69f  raw/s2048-bm32-bn32-w8-s3-packed.json
eb69844bc89bc1e2f1a07c1372ccc339447115b7a6b1e49cb63fbf20485b2951  raw/s2048-bm32-bn32-w8-s3-scalar-head.json
7b16732914882f350203a4ff51257264ef8cf7ff675354a8308ad7c200ee0645  resources/packed-dkdv.json
010a02e7c3aaf4a01886f217a4a0b4fd17798d502fa96e96913c24069c3b18f6  resources/packed-dq.json
27ffb0e8334982ed2e9a7afa7a33a8e68c454484aefea49c2a448fc01bfeb693  resources/scalar-head-dq.json
```
