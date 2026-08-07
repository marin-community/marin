# Shuttle Next Prototype: Routed Sparse Attention

Status: active research phase on `research/shuttle-routed-sparse-attention`.

Baseline: annotated local checkpoint `shuttle-gb200-moe-v1`.

Implementation plan: [routed_sparse_attention_plan.md](routed_sparse_attention_plan.md).

## Goal

After freezing the dense-Transformer and distributed-MoE checkpoint, test whether Shuttle's runtime-relation machinery generalizes to routed sparse long-context attention:

```text
query block
→ select a sparse set of KV blocks
→ compute exact attention over those selected blocks
→ merge partial attention states
```

This is a generality experiment, not a production sparse-attention project. The key question is whether the same relation/index planning, grouping, placement, buffering, and scheduling machinery used for MoE can generate a good sparse-attention implementation when mainly the grouped computation and merge operation change.

Do not begin a major XLA integration project during this experiment.

## Why this experiment

The current MoE compiler handles:

```text
token route
→ runtime sparse relation
→ group by expert
→ move payload
→ grouped computation
→ return
→ merge by source token
```

Routed sparse attention has the analogous shape:

```text
query block
→ runtime sparse relation
→ group by selected KV block
→ fetch or stage KV payload
→ grouped block-attention computation
→ return partial attention state
→ merge by query block
```

The grouped bodies and merges are deliberately different:

```text
MoE:
    GMM → SwiGLU → GMM
    merge = weighted vector sum

Sparse attention:
    QKᵀ → local softmax contribution → PV
    merge = online-softmax-state merge
```

Clean transfer would show that Shuttle found a useful compiler abstraction rather than reconstructing only MoE.

## Semantic target

Normalize a selected-block attention program approximately to:

```text
selected_blocks =
    router(query_metadata, kv_metadata)
relation =
    build_relation(
        left = query_block,
        right = kv_block,
        edges = selected_blocks
    )
partial_states =
    relation_program(
        relation,
        group_by = kv_block or query_block,
        body = block_attention_partial
    )
output =
    segmented_merge(
        partial_states,
        group_by = query_block,
        operator = online_softmax_merge
    )
```

The first router may use prerecorded routing indices, a simple top-k block scorer, or synthetic sparse patterns. A sophisticated retrieval model is out of scope.

## Required reusable abstraction

Reuse the MoE relation/index-plane representation directly where possible. A relation edge should contain enough information for:

- left identity;
- right identity;
- left-local position;
- right-local position or storage location;
- edge metadata;
- validity.

For this experiment, the left side is a query block and the right side is a KV block. The relation must support traversal and grouping by either side.

Do not add `SparseAttentionRoutePlan` if a small generic extension to `RelationPlan` suffices. Document MoE-specific assumptions that prevent reuse.

## Attention partial state

Represent the exact contribution from one selected KV block explicitly:

```text
AttentionPartial {
    row_max
    row_sum_exp
    weighted_value_accumulator
}
```

Implement an associative merge that rescales two partials to a common maximum before combining their denominator and weighted-value accumulator. Finalization divides the weighted-value accumulator by the row sum.

Reuse the online-softmax algebra already validated by the dense FlashAttention work where possible. The compiler must treat this object as reduction state, not as an output tensor that can be elementwise-added.

## Two execution orientations

### Query-major

```text
for each query block:
    keep query state resident
    for each selected KV block:
        QKᵀ
        online-softmax update
        PV update
    finalize query
```

This is a sparse Fold analogous to ordinary FlashAttention.

### KV-major

```text
group queries by selected KV block
fetch or stage one KV block
process all query blocks that selected it
produce partial states
route partial states back to query owners
merge partial states by query
```

This is closer to MoE and may expose KV reuse. Both must be representable as schedules of the same semantic relation. Do not assume which wins.

## Backend strategy

Reuse expert physical components where useful, including FlashAttention-style QK/PV primitives, CuTe or QuACK mainloops, existing online-softmax code, transport primitives, MoE relation-plan code, and task/event/buffer machinery.

De novo WGMMA instruction generation is not required. The experiment concerns composition and scheduling.

## Initial workload

Start with a fixed long-context BF16 configuration, choosing exact dimensions based on available kernels and hardware:

```text
query heads:       32
KV heads:          8
head dimension:    128
sequence length:   16K or 32K
KV block size:     64 or 128
selected blocks:   8–32 per query block
causal:            true
```

Start single-GPU if it materially accelerates iteration. Attempt a distributed variant only when it can reuse the existing MoE communication machinery without large new infrastructure.

## Measurements

For each candidate record:

- total latency;
- QK/PV compute latency;
- routing/index-plan latency;
- KV movement or staging latency;
- partial-state merge latency;
- HBM traffic;
- communication traffic when distributed;
- buffer sizes;
- worker allocation;
- selected relation orientation.

Compare with a dense-attention correctness baseline, a straightforward sparse implementation, a strong available sparse-attention reference with matching semantics, and the compiler-generated candidate. The important performance comparison is against a strong hand-designed implementation of the same sparse semantics, not necessarily dense FlashAttention.

## Generality accounting

Classify every implementation change as:

- reused unchanged from MoE;
- generalized from MoE;
- new generic Shuttle machinery;
- sparse-attention-specific backend code;
- sparse-attention-specific semantic recovery.

The central audit is whether `RelationPlan`, task derivation, grouping/orientation, buffer logic, event logic, and transport scheduling transfer. A strong outcome needs new code mainly for the block-attention grouped body, online-softmax-state merge, and attention-specific legality/layout rules.

If relation planning or scheduling must be substantially rewritten, document the mismatch instead of hiding it behind a new workload-specific abstraction.

## Success criteria

Strong success requires:

1. Runtime sparsity is represented without MoE-specific semantic concepts.
2. At least query-major and KV-major orientations execute correctly.
3. Online-softmax partial states merge correctly and deterministically.
4. Existing grouping, scheduling, and buffering machinery is substantially reused.
5. The selected implementation is plausibly competitive with a strong hand-written reference for the same sparse workload.
6. Profiling explains the remaining performance gap.

Transfer is more important than a flattering latency number. A result that is 20% behind an oracle because of one weak attention primitive is more useful than an excellent bespoke sparse-attention compiler.

## Research sequence and integration boundary

After this experiment, and only if the relation abstraction survives, test a minimal `StatefulScan` on Gated DeltaNet or Kimi Delta Attention. Reevaluate and simplify the normal form after both experiments. Begin a serious XLA/Shuttle compiler implementation only if runtime sparse relations transfer outside MoE and ordered Scan with chunk algebra also works.

The intended sequence is:

1. Freeze dense plus MoE.
2. Routed sparse attention: test `RelationPlan` reuse.
3. Gated DeltaNet or KDA: test `StatefulScan`.
4. Reevaluate the normal form based on what transferred.
5. If still coherent, begin the real Shuttle/XLA implementation.
6. Later address backward, whole-step scheduling, distributed parallelism, MXFP8, and other architectures.

## Required reading and executable references

The first three are primary:

1. **MoBA — Mixture of Block Attention for Long-Context LLMs**
   Paper: <https://arxiv.org/abs/2502.13189>
   Code: <https://github.com/MoonshotAI/MoBA>
   Use its naive semantic implementation and optimized implementation to define selected-block semantics and study query-to-KV-block relation planning.
2. **FlashMoBA — Optimizing Mixture of Block Attention**
   Paper: <https://arxiv.org/abs/2511.11571>
   Code: <https://github.com/mit-han-lab/flash-moba>
   Prefer it as the performance oracle when supported shapes and hardware match.
3. **DeepSeek Sparse Attention and FlashMLA sparse kernels**
   Model: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
   Kernels: <https://github.com/deepseek-ai/FlashMLA>
   Study the explicit selected-index tensor and SM100 sparse-prefill path as an index-plane reference and possible GB200 physical oracle.
4. **Quest — Query-Aware Sparsity for Efficient Long-Context LLM Inference**
   Paper: <https://arxiv.org/abs/2406.10774>
   Code: <https://github.com/mit-han-lab/Quest>
   Use it to study separation of selection metadata from KV payload movement.
5. **Block-Sparse-Attention**
   Code: <https://github.com/mit-han-lab/Block-Sparse-Attention>
   Use it as a physical reference for arbitrary block masks on Hopper and Blackwell.
6. **Native Sparse Attention**
   Paper: <https://arxiv.org/abs/2502.11089>
   Third-party code: <https://github.com/fla-org/native-sparse-attention>
   Retain as a later multi-branch stress test rather than the first implementation target.
7. **SeerAttention**
   Paper: <https://arxiv.org/abs/2410.13276>
   Code: <https://github.com/microsoft/SeerAttention>
   Use it to ensure the relation representation does not assume MoBA's router.

Keep the original FlashAttention paper and repository nearby for merge algebra:

- <https://arxiv.org/abs/2205.14135>
- <https://github.com/Dao-AILab/flash-attention>

The prototype order is: start from MoBA semantics, inspect FlashMoBA and FlashMLA as implementation oracles, and use Quest to reason about index-plane/payload-plane separation. NSA and SeerAttention are secondary checks against overfitting the abstraction to MoBA.
