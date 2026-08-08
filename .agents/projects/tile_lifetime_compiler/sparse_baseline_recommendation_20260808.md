# Routed Sparse-Attention Baseline Recommendation

## Background Research Brief

Status: retained as the full-NSA follow-up design. The later SM100 oracle audit
in [sparse_attention_oracle_research_20260808.md](sparse_attention_oracle_research_20260808.md)
promotes MiniMax Sparse Attention to the first routed experiment because it
isolates relation synthesis and has a stronger current GB200-native oracle.

- Effort: medium
- Stop rule: stop when additional official implementations no longer change the
  first experiment or its acceptance boundary.
- Date: 2026-08-08

## Recommendation

After the MiniMax Sparse Attention relation experiment, use a full Native
Sparse Attention (NSA) forward core as the next composition test. Use the
pinned Flash Sparse Attention (FSA) repository as the H100 end-to-end oracle and
current cuDNN Frontend NSA components as a B200 component oracle. Keep DeepSeek
Sparse Attention (DSA) as the subsequent token-routed stress test.

NSA is a stronger Shuttle experiment than another precomputed-mask sparse
kernel because its route is produced by the program:

```text
learned strided K/V compression
  -> compressed exact attention
  -> block-score reduction and top-k
  -> Relation(query token x KV head, selected KV block)
  -> selected exact attention
  + sliding-window exact attention
  -> gated branch sum
```

This exercises `Contract`, `Fold`, `DomainRestriction`, `Selection`,
`RelationPlan`, right-resource grouping, and materialization choices in one
natural program. The official NSA paper defines compression, selection, and
sliding branches; FSA provides an H100-capable implementation, and current
cuDNN Frontend exposes source-visible component APIs and CuTe DSL kernels.

## Important contradiction: NSA is not one attention over a union

NSA computes three independently normalized attention outputs and combines them
with learned gates:

```text
output = g_compressed * O_compressed
       + g_selected   * O_selected
       + g_sliding    * O_sliding
```

It is not correct to merge compressed, selected, and sliding states into one
`(max, sum_exp, weighted_value)` state. That would implement softmax over the
union of three domains and change the model.

The exact normalized-exponential state merge belongs inside each branch. In
particular, each selected KV block produces an `AttentionPartial(m, l, o)`, and
those partials merge by the usual common-maximum rescaling. The final
cross-branch operation is an ordinary gated `Map` and sum `Fold`.

This distinction should be an explicit semantic test.

## Minimal primary workload

Run the attention core after projection and RoPE so the new experiment is not
dominated by already-tested dense machinery.

```text
hardware:              one H100 SXM
batch:                 1 packed sequence
sequence length:       65,536 primary; 16,384 debug
query heads:           16
KV heads:              4
head dimension:        128 for Q, K, and V
dtype:                 BF16 inputs/output, FP32 attention state
compression window:    32
compression stride:    16
selected block size:   64
selected blocks:       16
forced initial blocks: 1
forced local blocks:   2
sliding window:        512
causal:                true
```

Inputs are post-RoPE `Q`, post-RoPE original `K`, original `V`, the learned
compression weights/intra-block positional values, and three gate logits per
query. The timed acceptance boundary includes learned compression, compressed
attention, block scoring, top-k, `RelationPlan` construction/legalization,
selected attention, sliding attention, sigmoid gates, and gated combination. It excludes QKV,
RoPE, and output projection. Component-only and cached-relation timings are
diagnostics, not acceptance numbers.

The FSA repository documents the same 64K, compression-32/stride-16,
block-64/top-k-16, and window-512 family. Its exact selected relation is per
query token and KV head, not block-shared:

```text
left  = (sequence, query_token, kv_head)
right = (sequence, kv_head, kv_block)
edge  = selected block, validity, deterministic slot
```

GQA expansion belongs in the grouped computation, not in the index plane.
The reference implementation sorts selected block IDs into ascending physical
order and marks invalid/future entries `-1`. Shuttle should preserve this order
as its deterministic selected-domain fold order while still building the
right-oriented grouping needed for KV reuse.

### Semantic matching hazards

- The FSA-supported variant is not the NSA paper's flagship tensor shape. The
  paper evaluates `D_qk=192`, `D_v=128`; FSA requires equal Q/K/V head
  dimensions and its public example uses 128. Call this the FSA-supported NSA
  variant, not exact reproduction of the paper configuration.
- The routing score is not simply a compressed-attention logit. FSA recomputes
  normalized compressed-attention probability, then applies an overlap-count
  weighted transform from strided compression windows to original KV blocks.
  Initial and local blocks are assigned infinite priority before top-k.
- The public FSA module forces its serial/per-KV-head top-k path. That path casts
  block scores to BF16 before top-k, whereas the parallel path keeps FP32. The
  natural JAX fixture must represent this cast explicitly if route hashes are
  required to match the public H100 oracle. It is a numerical-policy choice,
  not innocuous backend metadata.
- FSA's gate is a sigmoid, not a softmax over branches, and is shared across
  attention heads (`[token, 3]`). FLA's NSA implementation accepts per-head
  gates, so it is not an exact oracle unless constrained to shared gates.
- Causal early rows have fewer than 16 valid blocks. FSA performs top-k first
  and replaces future block IDs with `-1`, so validity is ragged even though
  the index tensor has a fixed final extent.
- Top-k ties need an explicit deterministic rule. Freeze the input/parameter
  fixture and route hash; do not infer equivalence only from close final output.

## What counts as clean synthesis

The accepted Shuttle path may retain or abstract generic physical machinery:

- generic matrix mainloops and layouts;
- TMA/WGMMA/CuTe copy and barrier primitives;
- generic streaming normalized-exponential state code;
- generic deterministic top-k/selection;
- generic `RelationPlan` orientation and bounded right-resource staging.

The accepted path may not call FSA, `parallel_nsa`, cuDNN `NSA.*`, or
FlashAttention sliding-window attention. Those are oracle-only semantic
kernels. The learned compression must lower to generic map/contract/fold
structure, top-k must produce a generic relation, and all three branch bodies
and the gate expression must be generated.

Mutation tests should change, without editing GPU source:

1. sliding-window `DomainRestriction` from 512 to 256;
2. top-k from 16 to 8;
3. one gate expression or selected-score scale;
4. a non-monotone selected relation with the same cardinality.

## Oracle roles and caveats

### FSA on H100: primary whole-core oracle

Pinned source: [`7ff144fd7ff485dc4220d439f31cc1708b64fef3`](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3)

FSA supports Ampere and Hopper, BF16/FP16, equal Q/K/V head dimensions up to
256, and GQA ratios 1-16. Its full module contains learned compression,
compressed attention and routing, optimized selected attention, sliding
FlashAttention, and gated combination.

Contradiction: only the selected branch is the novel FSA physical kernel. The
full wrapper also uses reference NSA compression code and an opaque
FlashAttention window kernel. It is still a valid end-to-end performance
oracle, but it is not a source that Shuttle can import wholesale and call clean.

### cuDNN Frontend on B200: current source-visible component oracle

Pinned source inspected:
[`6d01e3c53f1e27d6e8994a45bfa85a0b4786dfe4`](https://github.com/NVIDIA/cudnn-frontend/tree/6d01e3c53f1e27d6e8994a45bfa85a0b4786dfe4)

Current cuDNN Frontend exposes separate selection, compression, top-k, and
sliding-window APIs. Selection accepts explicit `(T, H_kv, K)` block indices
and returns output, log-sum-exp, and maximum. The source includes the CuTe DSL
kernels, making it particularly useful for physical-skeleton study.

Contradictions:

- There is no central end-to-end NSA API in the checked-in source.
- Its checked-in `sparse_attention.md` still says only selection is implemented,
  although the source tree and public docs expose all four components.
- Selection permits SM90+, but compression and top-k currently require SM100+.

Therefore this is a B200 component and assembled-core oracle, not the primary
H100 whole-program denominator.

### DSA/FlashMLA: strong follow-up, poor first slice

Current FlashMLA source inspected:
[`15f13e5030374295491c5ce31b02d7e63a7772c6`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6)

DSA is an even more direct runtime relation:

```text
lightweight multi-head indexer
  -> causal top-2048 token selection
  -> Relation(query token, selected KV token)
  -> sparse MLA attention
```

It has official SM90 and SM100 sparse-prefill kernels and current cuDNN
Frontend has source-visible indexer/top-k machinery. It is not the best first
experiment because the natural workload adds several confounders at once:
FP8 block-scaled indexer semantics, token-level top-k 2048, MQA/MLA layout,
`D_qk=576`, `D_v=512`, and a very large relation. FlashMLA's sparse prefill API
also accepts precomputed indices, so benchmarking it alone would omit the
router.

Promote DSA only after the NSA program can synthesize its own route and all
three branches. It is then an excellent test that the same relation machinery
handles token-level rather than block-level destinations.

### Rejected as primary

- FlashMoBA remains useful, but the currently matched workload is not native to
  its per-token/per-head router and therefore gives a loose denominator.
- SeerAttention is a useful learned-router semantic reference, but the audited
  physical baseline is not a stronger current H100 denominator than FSA/cuDNN.
- NATTEN and current block-sparse kernels have strong Hopper/Blackwell physical
  implementations but do not generate a runtime routed relation.

## Falsifiable experiment sequence

1. Export the natural NSA core from JAX and recover the three independent Fold
   branches, learned compression, top-k `Selection`, and relation.
2. Prove that the generated `RelationPlan` matches the FSA `topk_idx` set,
   validity, and deterministic ascending source order.
3. Run the existing generated streaming skeleton over the selected relation,
   then add generic right-oriented KV-block grouping/staging.
4. Generate compressed and window branches from the same attention skeleton
   with different fold domains.
5. Generate the gated branch sum and benchmark the complete core against FSA.
6. On B200, assemble the equivalent cuDNN component oracle and compare each
   component before changing Shuttle schedules.

The abstraction is falsified if NSA requires a new workload-specific route plan,
if the selected branch cannot use the same partial state as dense/routed
attention, or if most performance requires calling an opaque NSA component.

## Source ledger

| Source | Revision | Use |
|---|---|---|
| [NSA paper](https://arxiv.org/abs/2502.11089) | arXiv 2502.11089 | Three-branch semantics and published configuration |
| [Flash Sparse Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3) | `7ff144f` | H100 full-core and selected-branch oracle |
| [FLA NSA](https://github.com/fla-org/native-sparse-attention/tree/bd67af59b90afa34b25f61d2922e612d10dba3bd) | `bd67af5` | Transparent semantic/Triton reference |
| [cuDNN Frontend NSA](https://github.com/NVIDIA/cudnn-frontend/tree/6d01e3c53f1e27d6e8994a45bfa85a0b4786dfe4) | `6d01e3c` | B200 component oracle and CuTe skeleton reference |
| [DeepSeek V3.2 Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | `87e509a` | DSA natural indexer semantics |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6) | `15f13e5` | SM90/SM100 DSA payload oracle |

## Stop reason

Additional block-sparse implementations changed the physical-oracle list but
not the first experiment. Full NSA is the smallest current workload that both
generates a runtime relation and stresses composition beyond a single sparse
attention body.
