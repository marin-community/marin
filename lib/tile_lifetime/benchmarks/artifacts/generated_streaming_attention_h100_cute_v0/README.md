# Generated SM90 streaming attention checkpoint

This checkpoint records the compiler-instantiated CuTe producer/consumer
streaming skeleton derived from backend-neutral `Contract/Map/Fold` semantics.
The generated path does not call the FlashAttention interface or a precompiled
FA3/FA4 forward kernel.

Configuration: one H100 80GB HBM3, driver 595.71.05, 700-W limit, observed
1,830-MHz SM and 2,619-MHz memory clocks, Torch 2.11.0+cu128, BF16, batch 1,
32 query heads, 8 KV heads, head dimension 128, causal, 128x128 tiles, two
pipeline stages, packed GQA, RS PV, and intra-warpgroup overlap. Every timing
file contains 30 raw CUDA-event samples after 20 warmups.

| Program | Median | Minimum | Pinned FA3 | Ratio |
|---|---:|---:|---:|---:|
| S=2048, scale=1/sqrt(128) | 0.078640 ms | 0.077568 ms | 0.0672 ms | 1.1702x |
| S=4096, scale=1/sqrt(128) | 0.237696 ms | 0.236832 ms | 0.2100 ms | 1.1319x |
| S=2048, scale=0.125 | 0.077760 ms | 0.076960 ms | comparison only | - |
| S=2048, scale=0.125, softcap=16 | 0.092528 ms | 0.091712 ms | comparison only | - |

The S=2048 and S=4096 sampled independent-reference maximum/mean absolute
errors are 0.015625/0.0001144 and 0.015625/0.0001208. All four programs are
bitwise deterministic across three complete-output hashes. The changed scale
and softcap programs flow through the same semantic lowering and physical
stages; they are not separate named attention modes.

The physical stage extraction is based on `flash-attn-4==4.0.0b16` source
files pinned by SHA256 in every JSON record and in
`backends/h100/cute_streaming_README.md`. The official FA3 oracle revision is
`3fa810570e17bb4354155bdb71d826eca6079208`. The artifact records the SHA256
of each Shuttle backend file actually executed.

The earlier compiler-owned Triton implementation is retained as a clean
fallback and negative performance result. It measured 0.100602 ms at S=2048
and 0.328818 ms at S=4096, or approximately 1.50-1.57x pinned FA3. Grouping
multiple GQA heads per Triton program and dynamic TMA descriptors were both
slower. The CuTe extraction is the first generated path to meet the 1.2x
criterion.
