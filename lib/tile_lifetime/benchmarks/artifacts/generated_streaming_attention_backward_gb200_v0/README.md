# Generated streaming-attention backward compile smoke on GB200

This is the requested compile-and-run smoke for Shuttle revision
`2f45798ff4f0adaaf52e3ae0bd003ba52c90e701`. It uses the natural generated
backward stages rather than an opaque attention-backward call: QK and
probability recomputation, DV/DP/DQ/DK Contracts, and the generated score-map
VJP.

Configuration: BF16, sequence 128, 32 query heads, eight KV heads, head
dimension 128, causal score map, 32x32 blocks, eight warps, three stages, two
counterbalanced samples, and one iteration per sample. There was no tuning or
primary performance run.

Compilation and execution succeeded and repeated output was deterministic.
The generated median was 0.131408 ms versus 0.181536 ms for the selected Torch
SDPA backward. However, the numerical errors are not acceptable evidence of a
correct backward: maximum/mean absolute errors were 0.761719/0.026239 for DQ,
0.843750/0.069165 for DK, and 0.765625/0.073522 for DV. The JSON's `passes`
field reflects only its 1.2x performance gate; this checkpoint must be treated
as a successful compile smoke and a correctness failure pending diagnosis.

Subsequent diagnosis found that the harness saved forward state with causal
M=32,N=64 tiles even though the forward diagonal split is legal only when M is
a multiple of N. That schedule double-counted or unmasked keys. Commit
`14f2bbc3f9` uses matched legal 32x32 forward state, rejects illegal causal tile
pairs, and makes correctness part of acceptance. Its replay is separate from
this frozen failure artifact.

Command:

```text
TORCH_CUDA_ARCH_LIST=10.0a python \
  lib/tile_lifetime/benchmarks/h100_generated_streaming_attention_backward.py \
  --sequence 128 --mutation causal --block-m 32 --block-n 32 \
  --repeats 2 --iterations 1 \
  --json-output /tmp/shuttle-streaming-backward-smoke-gb200.json \
  --shuttle-revision 2f45798ff4f0adaaf52e3ae0bd003ba52c90e701
```

Environment: one low-priority NVIDIA GB200, driver 595.71.05, Torch
2.10.0+cu130, Triton 3.6.0, CUDA 13.0. The exact source archive SHA-256 was
`121a5292ca05bd68d7aa37ffd2a6567c67322f545723231e54e1b88a683f5cf6`.
