# Streaming-attention training H100 fixed run

This artifact records the single fixed H100 replay authorized after
`nvidia-cuda-cccl==13.3.3.4.1` was added to the Linux GPU dependency lock at
Shuttle base revision `ec9cc46259df3ce29e7ebe95675e5f1bb63d854e`.

The component boundary is the existing ordinary JAX function from Q, K, V, and
output cotangent to forward output plus Q, K, and V cotangents. The generated
candidate is recovered from Contract, Map, normalized-exponential Fold, and
DomainRestriction semantics. The matched expert is Torch SDPA forced to its
Flash backend; Torch is benchmark-only.

The fixed configuration is BF16 causal GQA with batch 1, sequence 2,048, 32
query heads, 8 K/V heads, head dimension 128, block M/N 32, eight warps, and
three pipeline stages. One process uses five warmups followed by 30
counterbalanced samples of five iterations. No tuning or automatic retry is
permitted.

The earlier CPU-only dependency gate
`/dlwh/shuttle-attention-cccl-preflight-20260809` compiled CUDA FP16/BF16 and
`<nv/target>` headers with the corrected lock before this H100 experiment was
submitted. The H100 reproduction repeats that header compile before the only
benchmark invocation.

The result, raw logs, generated sources, toolchain records, and release proof
will be added after the terminal job is collected. Until then this is a
reproduction checkpoint, not a performance claim.
