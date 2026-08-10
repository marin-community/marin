# Streaming-attention training H100 reviewed replay

This artifact pins the single fixed H100 replay authorized after the physical
and numerical boundary review at Shuttle revision
`11e8dca05ed4e2207cc02312c9d6265ea1c32c58`.

The component boundary takes ordinary JAX Q, K, V, and output-cotangent BSHD
buffers and returns forward output plus Q, K, and V cotangents. The generated
candidate comes from recovered Contract, Map, normalized-exponential Fold, and
DomainRestriction semantics. The expert is Torch SDPA forced to its Flash
backend; Torch remains benchmark-only.

The fixed configuration is BF16 causal GQA with batch 1, sequence 2,048, 32
query heads, 8 K/V heads, head dimension 128, block M/N 32, eight warps, and
three pipeline stages. The timed Torch wrapper includes zero-copy BSHD/BHSD
input and result views. Both implementations include forward and backward.

Before warmup, the benchmark must establish exact dtype, shape, stride, and
minor-to-major equivalence for Q/K/V/dO and O/dQ/dK/dV. It must also establish
repeat determinism, finite outputs/errors, maximum absolute error at most
0.125, and mean absolute error at most 0.01 for generated and expert outputs
against the independent semantic reference.

The reproduction performs the locked dependency and CUDA header preflight,
then invokes the fixed benchmark command exactly once. It permits no tuning,
retry, or GB200 follow-up. Results and raw evidence are added only after the
terminal H100 job is collected.
