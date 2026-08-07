# Routed Sparse Attention: Background Research

Date: 2026-08-06

Effort: high. The review covered primary papers, pinned official repositories, tests, kernel interfaces, benchmarks, and adversarial issue searches for unsupported shapes, hardware, determinism, memory growth, and missing functionality.

Research brief: [routed_sparse_attention_brief.md](routed_sparse_attention_brief.md).

Implementation plan: [routed_sparse_attention_plan.md](routed_sparse_attention_plan.md).

## Decision

Start on H100 with one prerecorded causal block relation and two complementary physical references:

1. MIT Block-Sparse-Attention for a query-major standard-attention baseline.
2. Flash Sparse Attention's selected-attention path for explicit KV-major inversion, partial-state materialization, and reduction.

Also test FlashMoBA's low-level precomputed-pattern API. It is likely the strongest complete MoBA performance oracle for the requested standard GQA shape, but its forward schedule is hybrid rather than purely KV-major.

Do not change the Shuttle workload to match FlashMLA. FlashMLA is the strongest native SM100 sparse kernel inspected, but its sparse-prefill path is token-sparse MLA/MQA with one KV head and 512/576-dimensional latent vectors. It is an index-plane and GB200 roofline control, not the primary semantic oracle.

## Semantic correction

Exact MoBA routes each `(query token, query head)` to top-k KV blocks. Shuttle's first experiment deliberately uses the simpler relation:

```text
query block → selected KV blocks
```

This is MoBA-like block-shared routing, not exact MoBA semantics. The same relation can still be expanded across rows and heads at an oracle adapter boundary, which keeps the compiler experiment clean and the comparison fair.

## Oracle taxonomy

### FlashMoBA

FlashMoBA exposes a valuable seam: routing and attention can be called separately. Its low-level attention function accepts precomputed KV-block-major metadata—column offsets, nonzero counts, and sorted query-row indices—so Shuttle can feed an identical prerecorded relation without invoking FlashMoBA's router.

Its forward kernel should not be labeled pure KV-major. The metadata is KV-major, but a CTA owns a logical query block, loops over KV blocks, gathers incident query rows, and retains query-owned FP32 state in a full-size temporary output. It is best described as a hybrid schedule and retained as an expert performance oracle.

The published H100 envelope includes BF16/FP16, MHA/MQA/GQA, head dimensions through 256, logical query blocks from 128 to 1024, KV blocks in multiples of 64, top-k through 64, and causal/non-causal forward. The paper reports 49 ms at sequence 64K versus 99 ms for FlashAttention-2 for its stated configuration.

### Flash Sparse Attention

Flash Sparse Attention is the closest inspected implementation of Shuttle's pure KV-major candidate:

```text
top-k relation
→ invert by KV block
→ grouped selected QK/PV work
→ partial states
→ inverse map
→ online-softmax reduction
```

It supports BF16/FP16, standard equal Q/K/V dimensions through 256, GQA ratios from 1 to 16, block sizes 32/64/128/256, and Hopper/Ampere prefill. Its implementation is mechanically rough and materializes large inverse maps and partial-state buffers, so it is a structural oracle and benchmark baseline rather than a schedule to copy.

### Block-Sparse-Attention

MIT Block-Sparse-Attention accepts arbitrary per-batch/per-head binary block relations and is the easiest standard-shape query-major reference. It supports causal BF16/FP16 forward/backward, GQA, and head dimension 128. The code compiles for SM90 and SM100, but it remains FlashAttention-2-style and has no published H100/B200 WGMMA/TMA performance evidence.

### FlashMLA

FlashMLA cleanly separates sparse indices from KV payload. Sparse prefill accepts `indices[s_q, h_kv, topk]`; invalid entries are `-1`. Its SM100 path reports up to 1450 TFLOP/s on B200.

The mismatch is substantial: sparse prefill is query-major token sparsity with `h_kv=1`, `d_qk` 512 or 576, `d_v=512`, no batch dimension, and causal legality already encoded in the indices. It should not determine Shuttle's initial workload.

### Quest, NSA, and SeerAttention

Quest is the clearest design reference for separating query-dependent page-selection metadata from KV payload movement, but its paper kernel is a decode implementation and does not support GQA.

FLA Native Sparse Attention has a useful explicit-index query-major interface, but it is community code rather than the paper authors' implementation and requires a GQA ratio of at least 16 in the inspected path.

SeerAttention shows that a learned router can emit thresholded or top-k block relations, which guards against overfitting semantic recovery to MoBA's parameter-free router. Its current Triton kernel consumes a dense binary block mask and remains query-major.

## Exact state algebra

The proposed partial state matches both FlashMoBA and official FlashAttention split-combine code:

```text
AttentionPartial {
    row_max
    row_sum_exp
    weighted_value_accumulator
}
```

For disjoint states `a` and `b`:

```text
m = max(a.m, b.m)
a_scale = exp(a.m - m)
b_scale = exp(b.m - m)
l = a_scale * a.l + b_scale * b.l
o = a_scale * a.o + b_scale * b.o
```

Empty state is `(-inf, 0, 0)`. Empty scales must be explicitly zero to avoid evaluating `exp(-inf - -inf)`. Finalization divides `o / l` and rejects rows whose selected fold domain is empty.

Duplicate edges remain valid in the generic relation because it may represent a multigraph. Selected-block attention rejects duplicate KV blocks because top-k represents a set and duplicates change probability mass. Causal MoBA always includes the current block, so an empty selected set is also rejected.

## Minimum executable experiment

Use one deterministic block-shared relation:

```text
batch:           1
sequence:        16K, with a 2K correctness scale-down
query heads:     32
KV heads:        8
head dimension:  128
Q block:         128
KV block:        128
selected blocks: 8 initially, then 16 and 32
dtype:           BF16 inputs, FP32 state
causal:          true
```

Each query block selects deterministic historical blocks plus its current block. Expand those block-shared edges across query rows and heads only in external-backend adapters.

Measure separately:

- relation construction;
- query-major execution;
- KV-major grouped computation;
- partial-state routing/materialization;
- state merge;
- end-to-end execution.

Compare:

1. dense FA3;
2. query-major Block-Sparse-Attention;
3. Shuttle query-major;
4. Flash Sparse Attention selected attention;
5. Shuttle KV-major;
6. FlashMoBA attention-only on the same precomputed relation.

Benchmark FlashMoBA's router wrapper separately. Do not compare its router-inclusive latency to prerecorded-route Shuttle execution.

## Falsifiers

- If `RelationPlan` cannot emit both source traversal and right-oriented offsets/sorted source IDs without a new attention-specific index structure, the reuse claim is weakened before kernel work.
- If KV-major partial-state traffic dominates and no relation-degree regime benefits from KV reuse, the orientation remains expressible but is not a useful candidate for this workload.
- If the generic index plane transfers but counted readiness, bounded buffers, and worker allocation require an expert-specific rewrite, report index-plane success and schedule-layer failure separately.

## Revision ledger

| Source | Revision |
|---|---|
| MoBA paper | [arXiv 2502.13189v1](https://arxiv.org/abs/2502.13189v1) |
| MoonshotAI/MoBA | [`b5d58363311d3ca946f1ec444182727c15e338b5`](https://github.com/MoonshotAI/MoBA/tree/b5d58363311d3ca946f1ec444182727c15e338b5) |
| FlashMoBA paper | [arXiv 2511.11571v2](https://arxiv.org/abs/2511.11571v2) |
| FlashMoBA | [`39d9ac043b271d046a2181a9991e99a26b67bca1`](https://github.com/mit-han-lab/flash-moba/tree/39d9ac043b271d046a2181a9991e99a26b67bca1) |
| FlashMoBA CUTLASS | `a2439551c765c5393aebe557ee75d3a0412d2211` |
| FlashAttention | [`69e1bcbe77c359c84b3a4589e92a7c076e33a202`](https://github.com/Dao-AILab/flash-attention/tree/69e1bcbe77c359c84b3a4589e92a7c076e33a202) |
| Flash Sparse Attention | [`7ff144fd7ff485dc4220d439f31cc1708b64fef3`](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3) |
| Block-Sparse-Attention | [`49d6c39e4dc0303442cda3bb758b3925d4399c49`](https://github.com/mit-han-lab/Block-Sparse-Attention/tree/49d6c39e4dc0303442cda3bb758b3925d4399c49) |
| FlashMLA | [`15f13e5030374295491c5ce31b02d7e63a7772c6`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6) |
| DeepSeek-V3.2-Exp | [`87e509a2e5a100d221c97df52c6e8be7835f0057`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) |
| Quest | [`01c1623bf9395009520874e989e29f683203b357`](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357) |
| FLA Native Sparse Attention | [`bd67af59b90afa34b25f61d2922e612d10dba3bd`](https://github.com/fla-org/native-sparse-attention/tree/bd67af59b90afa34b25f61d2922e612d10dba3bd) |
| SeerAttention | [`aba03e3f2caefd0ccd21e576670aa830b748c84e`](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) |

## Generality baseline before physical planning

The index-plane result is promising but narrower than the long-term architecture currently implies.

Reused unchanged from MoE:

- destination ownership coordinates;
- stable destination grouping;
- padded group counts and offsets;
- source dispatch and coalesced dispatch;
- inverse route mapping;
- capacity checks.

Generalized from MoE:

- per-edge validity for a rectangular storage envelope;
- invalid-slot fill values during inverse dispatch;
- weighted merge skipping invalid routes.

New generic Shuttle machinery:

- executable exact attention partial-state algebra.

Not yet demonstrated reusable:

- task derivation;
- counted readiness and event generations;
- bounded buffer reuse;
- worker pools;
- physical transport scheduling.

The existing MoE records for those concepts are typed around expert stages, and readiness events do not yet carry derived arrival counts. Do not claim that the schedule layer transferred until the physical candidate generator proves it.
