# Matched FlashMoBA H100 sparse-attention oracle

This artifact replaces the Ampere-oriented Seer timing as the primary
sparse-attention comparison. It measures Shuttle and pinned FlashMoBA on an
identical natural-program boundary:

```text
FP32 metadata Contract
→ causal block restriction
→ sorted top-k Relation/RelationPlan
→ BF16 exact attention over selected blocks
→ BF16 output
```

QKV and output projections are excluded symmetrically. The primary shape is
batch 1, sequence 16,384, logical query/KV block 128, top-8, 32 query heads,
8 KV heads, head dimension 128, BF16, and causal attention.

## Semantic boundary

FlashMoBA's native router is not interchangeable with the natural Shuttle
router. It scores every query token and head against mean-pooled K blocks and
forces the current causal block. Shuttle scores explicit FP32 block metadata,
shares one relation across heads and tokens in a query block, and permits a
selected set that omits the current block. The matched full comparison
therefore uses the identical Shuttle router on both paths, generically
reorients its relation into FlashMoBA's KV-column-major query-row lists, and
then calls FlashMoBA's precomputed-relation attention interface.

The primary fixture contains 95 query blocks whose selected set omits the
current KV block. Correct output on those blocks closes the most important
MoBA-specific semantic risk. Slot order is erased to a selected set before the
exact normalized-exponential Fold.

## Revisions and environment

- Shuttle source: `5fd34c7057` plus the benchmark follow-up captured in
  `raw/h100_natural_routed_streaming_attention.py`
- FlashMoBA: `39d9ac043b271d046a2181a9991e99a26b67bca1`
- FlashMoBA CUTLASS: `a2439551c765c5393aebe557ee75d3a0412d2211`
- GPU: NVIDIA H100 80 GB HBM3
- Driver: 595.71.05
- CUDA compiler: 13.2.86
- PyTorch: 2.13.0+cu130
- Power limit: 700 W

The source patch only specializes the pinned build to inference by disabling
backward, dropout, alibi, softcap, uneven-K, and local-attention template
variants. It does not change the forward algorithm.

## Physical query-group sweep

Logical query blocks remain 128 for every candidate. The query-group parameter
only changes FlashMoBA's physical grouping.

| Physical group | Shuttle full | FlashMoBA full | FlashMoBA payload |
|---:|---:|---:|---:|
| 128 | 0.628592 ms | 6.063392 ms | 5.696448 ms |
| 256 | 0.622128 ms | 5.765760 ms | 5.405952 ms |
| 512 | 0.625072 ms | 5.588032 ms | 5.209648 ms |
| 768 | 0.635376 ms | 5.811184 ms | 5.409008 ms |
| 1024 | 0.637024 ms | 5.282320 ms | 4.905136 ms |

Group 1024 was selected before the two primary captures.

## Frozen result

Two independent counterbalanced captures contain 30 steady-state samples per
implementation. Pooled medians are:

| Measurement | Pooled median | Range |
|---|---:|---:|
| Generated Shuttle full boundary | 0.617200 ms | 0.582528–0.718240 ms |
| Matched FlashMoBA full boundary | 5.264560 ms | 5.213344–5.457760 ms |
| FlashMoBA cached-relation payload | 4.894560 ms | 4.864256–4.948960 ms |
| Common router only | 0.044080 ms | 0.043168–0.052000 ms |
| Relation reorientation only | 0.211664 ms | 0.200832–0.224704 ms |

The generated/full ratio is 0.117237 times; the generated/payload ratio is
0.126099 times. Generated and FlashMoBA outputs differ by at most 0.00390625
with mean absolute difference 0.0000651724. Both outputs repeat bitwise. The
generated path's independent semantic-reference check has maximum/mean error
0.00790286/0.000179032.

Stable hashes across both captures:

- relation: `0a2a06781755f5f577237a2e48c810cb160fd88db3295835458350f47ad61cbb`
- Shuttle output: `e9399766068941b3b60329760c04b576c79cec38fd036c8cbdfe43cdf8da3a83`
- FlashMoBA output: `d4991478a94211c0a48b442259004f660efaa8f56c34c549b3f832f73067ee21`

## Interpretation

This is an exact-semantic expert comparison and closes the stated 1.20-times
gate. It is not evidence that 5.26 ms is the best possible implementation of
the block-shared relation. FlashMoBA retains a more general per-token,
per-head row-list interface, and its active implementation uses SM80-style MMA
plus `cp.async` rather than WGMMA/TMA. Shuttle's generated skeleton is
specialized to the shared block relation and is Hopper-native. The current MIT
Block-Sparse-Attention measurement of 1.423632 ms remains a tighter secondary
local H100 control, but that kernel is also SM80-style.

A future tight oracle would need a hand-optimized block-shared WGMMA/TMA body
or a natural workload that exactly matches FlashMoBA's native token/head
router. Further tuning against the current loose denominator is not useful.

The raw directory contains all distributions, source pins, environment and
hardware captures, build logs, the inference-only source patch, and the exact
benchmark source used for the run. `SHA256SUMS` covers every frozen file.
