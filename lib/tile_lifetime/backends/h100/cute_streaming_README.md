# SM90 streaming contraction/fold extraction

This directory contains an experimental, optional CuTe backend for the
backend-neutral `StreamingAttentionProgram`. The compiler input remains a
generic `Contract -> Map -> Fold -> Contract` program. The emitter legalizes
its score expression, GQA index map, and tile schedule before instantiating the
physical producer/consumer stages.

The backend does not call `flash_attn_func`, the FlashAttention interface, or
an official precompiled attention kernel. `cute_streaming_base.py` and
`cute_streaming_sm90.py` are mechanically extracted from the FlashAttention 4
CuTe source and retain the upstream copyright. They are marked as preserved
upstream-style source so repository formatters do not mechanically rewrite the
CuTe DSL.

## Pinned extraction source

- Package: `flash-attn-4==4.0.0b16`
- `flash_attn/cute/flash_fwd.py` SHA256:
  `dec41cc35c28ee122c9808238dd97c482edfe22c2817697c2df44e5dfa46a222`
- `flash_attn/cute/flash_fwd_sm90.py` SHA256:
  `4dcf8ecabc518888aad8a677de2279348b03bc278ccdb78f356f453fedb5f3f4`

## Remaining helper boundary

The extracted stages still import these FlashAttention CuTe helpers:

- tensor alignment and dynamic-layout conversion from `cute_dsl_utils`;
- scale conversion and index fast-divisors from `utils`;
- online max/sum-exp state and score-map application from `softmax`;
- causal/tail predicate construction from `mask`;
- sequence and tile-coordinate records from `seqlen_info` and `block_info`;
- packed-GQA layout/index/TMA helpers from `pack_gqa`;
- named barriers, bounded pipelines, and tile schedulers;
- block-sparse and paged-KV helpers imported by preserved branches but unused
  by the initial dense compilation.

The QuACK dependencies provide TMA copy/layout helpers and SM90 WGMMA
mainloops. CUTLASS/CuTe provides tensor layouts, pipeline barriers, TMA, and
WGMMA instructions. These are physical primitives, not complete workload
calls.

The first dependency-reduction target is to prune the unused SM80, varlen,
paged-KV, and block-sparse branches. The second is to move the small online
state, masking, packed-GQA, and scheduler helpers behind Shuttle-owned physical
interfaces. Until that is done, this is an exposed and compiler-instantiated
stage extraction, not a dependency-independent reimplementation.
