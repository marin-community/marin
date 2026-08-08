# Routed Sparse-Attention Oracle Research

## Background Research Brief

- Effort: medium
- Stop rule: stop when new primary-source candidates no longer change the first
  experiment or its acceptance boundary.
- Date: 2026-08-08

## Question

Which public routed sparse-attention system gives Shuttle the strongest clean
synthesis target after the helper-boundary cleanup?

The oracle must expose a natural mathematical program, a runtime relation, and
an exact sparse-attention payload. It should have public source that can run on
Hopper or Blackwell and a representative BF16, GQA, head-dimension-128
configuration. The acceptance boundary must include the router and relation
construction when they are part of the source program.

## Recommendation

Use MiniMax Sparse Attention (MSA) as the primary experiment on B200 or GB200.
Pin the repository at
[`80434d7f67877c6570ca19cac444b84bc9855dac`](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac)
and its CUTLASS submodule at
`eb61c911471867a5fd2466bfd8f29306cea6ebf8`.

MSA is a direct test of Shuttle's intended normal form:

```text
index Q/K Contracts
  -> causal token score Contract
  -> block-max Fold
  -> top-k Selection
  -> Relation(query token x GQA group, selected KV block)
  -> exact selected QK Contract
  -> normalized-exponential Fold
  -> PV Contract
```

The [paper](https://arxiv.org/html/2606.13392) defines this program explicitly.
The public CuTe-DSL implementation accepts runtime block indices, supports BF16
GQA with head dimension 128, and implements a KV-outer schedule with hot-KV
staging and a deterministic two-phase combine. The code targets SM100, so this
experiment should use the low-priority B200/GB200 capacity that is already
available.

MSA supersedes Native Sparse Attention (NSA) as the first experiment. NSA is a
good second-stage composition test, but its natural program contains compressed,
selected, and sliding-window attention branches with separate normalizations and
a learned gated sum. MSA isolates the runtime-relation question while retaining
a learned router and a strong physical oracle.

## Current Shuttle Context

The current sparse prototype already recovers a natural JAX program into
`Relation`, `RelationPlan`, `Contract`, `Fold`, and `DomainRestriction`. It has
real query-major and KV-major schedules, including a non-monotone relation and
shared-memory KV staging. Two acceptance gaps remain:

1. The generated streaming body still depends on imported FlashAttention
   softmax/mask helpers. Shuttle must own the normalized-exponential state and
   domain predicate before this new result can count as clean synthesis.
2. The existing performance denominators are loose. The matched FlashMoBA run
   changes the native router semantics, and the local MIT Block-Sparse-Attention
   control still uses an SM80-style physical body.

The preserved local FlashMoBA artifact uses revision
`39d9ac043b271d046a2181a9991e99a26b67bca1`, BF16, `Hq/Hkv=32/8`, `D=128`,
block size 128, and top-8 on H100. Shuttle's common router scores metadata once
per query block and may omit the current block. Native FlashMoBA scores every
query token/head against mean-pooled keys and forces the current block. The
payload comparison is useful, but it is not a matched natural-program oracle.

## Primary Candidate: MiniMax Sparse Attention

### Natural semantics

For hidden states `X`, MSA forms ordinary main Q/K/V projections and a separate
lightweight index branch:

```text
Q_idx = stop_gradient(X) @ W_q_idx   # [N, H_kv, D_idx]
K_idx = stop_gradient(X) @ W_k_idx   # [N, 1, D_idx]
S_idx[i, r, j] = Q_idx[i, r] @ K_idx[j] / sqrt(D_idx)
M_idx[i, r, b] = max(S_idx[i, r, j] for j in block b and j <= i)
selected[i, r] = top_k(M_idx[i, r], k)
```

The selected relation is shared by the query heads in each GQA group. The local
block is always included. Main attention is ordinary scaled softmax attention
restricted to the causally visible tokens in those blocks. The paper's
representative model uses `Hq=64`, `Hkv=4`, main head dimension 128, block size
128, and top-k 16.

This frontend is suitable for the clean-synthesis mutation test. Changing
block aggregation from `max` to another generic Fold, changing the forced-block
policy, or adding a domain predicate should regenerate the relation and physical
program without selecting an `MSA` kernel by name.

### Public physical oracle

The pinned repository provides an SM100 CuTe-DSL sparse-prefill implementation.
Its useful properties are:

- explicit runtime `kv_block_indexes`;
- BF16 main attention with GQA and head dimension 128;
- a KV-outer schedule that derives a reverse sparse index;
- persistent CTA work scheduling and chunked hot-KV staging;
- fixed partial-state slots and a two-phase combine without atomic accumulation;
- an exp-free small-k selector for top-k 16;
- integration tests that compose proxy scoring, top-k, and selected attention.

The [paper's kernel section](https://arxiv.org/html/2606.13392#S4) is the primary
description of the KV-outer and two-phase design. The pinned
[repository README](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/README.md)
and
[integration test](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/tests/integration/test_proxy_kv_e2e.py)
show the executable API composition.

### Oracle caveats

The published 14.2x prefill and 7.6x decode speedups were measured on H800 at
one-million-token context. They do not establish the latency of the current
SM100 repository on B200.

The public
[sparse-attention benchmark](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/benchmarks/bench_sparse_attention_ops.py)
uses synthetic first-k block indices. It measures the payload kernel, not the
natural index projection, block scoring, block maximum, top-k, reverse relation,
and payload path together. The repository's full-path tests establish
composition and correctness but do not publish a full routed latency.

The repository also has an FP4 indexer. Do not use it for the first proof. BF16
index scores keep scale-tensor semantics out of this experiment. FP4 belongs in
a later numerical-contract test.

## Exact First Experiment

### Natural JAX program

Export this complete program to StableHLO:

```text
Q, K, V = main projections of X
Q_idx, K_idx = detached index projections of X
token_scores = Q_idx @ K_idx.T / sqrt(D_idx)
block_scores = causal blockwise max(token_scores)
selected = deterministic top-16(block_scores, force_local_block=True)
output = exact BF16 GQA attention(Q, K, V, selected)
```

Use the MSA model family for the first primary configuration:

```text
hardware:            one B200 or GB200
batch:               1 packed sequence
sequence length:     16K debug; 64K primary
query heads:         64
KV heads:            4
main head dimension: 128
index dimension:     128 initially
KV block size:       128
selected blocks:     16 including the forced local block
causal:              true
dtype:               BF16 tensors, FP32 online state/accumulation where specified
```

If repository constraints require a different index dimension, change both
programs and record it. Do not silently adapt MSA's relation to Shuttle's old
block-level router.

### Semantic matching checklist

- The left relation identity is `(query token, KV head/GQA group)`, not query
  block alone.
- All query heads in one GQA group share the same selected block set.
- Token scores are aggregated by a causal block maximum.
- Top-k reserves one slot for the local block and selects the remaining slots
  under the same tie policy.
- Selected block IDs refer to original physical token positions.
- A partially visible current block still applies token-level causality.
- Invalid or padded route slots use the same validity convention.
- Slot order may be canonicalized only if exact attention semantics are
  unchanged. Deterministic source order must be fixed for reproducibility.
- Main Q/K/V scale, BF16 casts, FP32 online state, and output dtype match.
- Router projections are either included by both implementations or excluded by
  both. The accepted full result should include them.

### Two benchmark boundaries

Record both boundaries; only the second is the end-to-end acceptance number.

1. Payload boundary: start from identical runtime selected indices and time
   reverse relation/index construction, scheduling, selected QK/Fold/PV, partial
   combine, and output.
2. Full routed boundary: time index projections, token score Contract, causal
   block-max Fold, top-k, RelationPlan construction/orientation, selected
   attention, and deterministic combine. Include main QKV projections only if
   both paths can match that boundary exactly.

Do not reuse a route or reverse index across timed iterations unless the natural
program itself makes the relation persistent. Exclude compilation and one-time
allocation from both paths. Preserve raw repeated-run distributions, clocks,
power policy, source pins, and route/output hashes.

### Acceptance hypothesis

Hypothesis: Shuttle's generic `Contract`/`Fold`/`Selection`/`RelationPlan`
pipeline can generate both query-major and KV-major MSA implementations, and
empirical selection will place the best full routed result within 1.20x of the
pinned MSA implementation on B200/GB200.

Falsifier: the natural MSA program cannot be represented without an
MSA-specific semantic node, or the accepted generated path requires calling the
MSA top-k/attention semantic kernel. A result slower than 1.20x is a performance
failure, not an abstraction failure, if the remaining gap is isolated to one
generic primitive.

## External Prior Art and Ranked Roles

### 1. MSA: primary SM100 routed oracle

The natural program, runtime relation, GQA D128 shape, public SM100 source, and
KV-outer deterministic schedule all match Shuttle's target. The missing piece is
a local full-boundary benchmark.

### 2. Native FlashMoBA: H100 fallback

Pin
[`39d9ac043b271d046a2181a9991e99a26b67bca1`](https://github.com/mit-han-lab/flash-moba/tree/39d9ac043b271d046a2181a9991e99a26b67bca1).
FlashMoBA remains the best H100 fallback if B200 access is delayed. The next run
must implement native MoBA semantics exactly: per-query-token/head scores over
mean-pooled key blocks, forced current block, matching top-k, and the same routing
boundary. Reusing Shuttle's old common metadata router would preserve the loose
denominator.

### 3. Full NSA/FSA: second-stage composition test

The [NSA paper](https://arxiv.org/html/2502.11089) defines compressed, selected,
and sliding-window branches with a learned output combination. Pin FSA at
[`7ff144fd7ff485dc4220d439f31cc1708b64fef3`](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3).

NSA is useful after MSA because it tests three attention Fold domains and a
gated merge. The branches have separate normalizations. Merging them as one
online-softmax state would change the model. The existing local FSA adapter ran
at about 12.54 ms and required a one-line Triton load patch, so FSA is not the
tightest first denominator.

### 4. Kascade: cross-layer relation reuse

Pin
[`d4463fcb4e66507ac7f83d072c43c26932ccc769`](https://github.com/microsoft/kascade/tree/d4463fcb4e66507ac7f83d072c43c26932ccc769).
Kascade computes exact top-k at anchor layers and reuses indices between anchor
layers. Its H100 configurations include GQA `32/8` and head dimension 128. It is
a later test of `PersistentState` or cross-layer relation reuse. The public path
is FP16 and configuration-specific, so it is a weaker first BF16 oracle.

### 5. FlashMLA sparse prefill: MLA-specific stress test

Pin
[`15f13e5030374295491c5ce31b02d7e63a7772c6`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6).
FlashMLA has official SM90 and SM100 sparse-prefill kernels with explicit runtime
indices. Its sparse interface uses MLA/MQA shapes such as one KV head, QK
dimensions around 512/576, and value dimension 512. It is not a fair oracle for
ordinary GQA D128. Use it only after adopting identical MLA semantics.

## Negative and Rejected Leads

- [SeerAttention](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e)
  has a learned block gate, but the audited prefill baseline does not provide a
  tighter current Hopper denominator than MSA. Its existing local comparison is
  already classified as provisional.
- [Quest](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357)
  cleanly separates query-aware page selection from KV payload movement, but its
  public performance path is decode-oriented and evaluated on Ada-class GPUs.
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
  supports Hopper and Blackwell but takes a mask rather than generating a
  runtime routed relation. Published plots are still A100 results, and the local
  H100 control retained an SM80-style body.
- [SpargeAttention](https://github.com/thu-ml/SpargeAttn) supports dynamic sparse
  masks on Hopper, but its recommended fast path uses SageAttention low-precision
  semantics. It is not an equivalent BF16 exact-attention oracle.
- [HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention) has a
  hierarchical learned router, but the public repository still lacks the
  promised efficient inference release.
- [SparDA](https://github.com/NVlabs/SparDA) has public H100 code and layer-ahead
  routing, but its current artifact is an offload-oriented deployment system
  with a large integration surface. It is a later placement/transport test.
- [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention/tree/bd67af59b90afa34b25f61d2922e612d10dba3bd)
  is an early Triton NSA implementation. Its published defaults center on head
  dimension 64 and large GQA ratios, not the desired BF16 D128 primary shape.

The adversarial search included 2026 routed and learned sparse-attention systems
with H100/B200 public code. MSA changed the first recommendation. Kascade,
HiLS-Attention, SparDA, and the other systems above did not displace it because
of missing inference code, mismatched precision/shape, or a less direct runtime
relation.

## Evidence Map

### Claim: MSA is the best first routed sparse-attention oracle

- Support:
  - The MSA paper gives a natural two-branch mathematical definition with a
    causal block-max/top-k relation and exact selected attention.
  - The pinned public code provides an SM100 BF16 GQA D128 implementation with
    explicit indices and KV-outer deterministic scheduling.
- Contradictions:
  - The public benchmark is payload-only and synthetic.
  - The paper's headline measurements are H800 results, while the public code
    inspected here targets SM100.
- Directness to Shuttle: high on B200/GB200.
- Confidence: exploratory until the pinned repository runs locally.
- Action: reproduce its integration test and build a symmetric full-boundary
  harness before optimizing Shuttle.

### Claim: NSA should follow MSA

- Support:
  - NSA exercises multiple Fold domains, learned compression, a routed selected
    branch, and a gated output combination.
- Contradictions:
  - The extra branches confound the first RelationPlan transfer test.
  - The current local FSA path is a loose performance oracle.
- Directness to Shuttle: high for the next composition experiment.
- Confidence: exploratory.
- Action: keep the existing NSA brief as the second experiment.

## Recommended Next Experiments

### 1. Reproduce pinned MSA on B200/GB200

- Minimum experiment: build the pinned repository, run the proxy/top-k/sparse
  integration test, and measure payload plus full routed distributions at 16K
  and 64K.
- Baseline/control: official MSA implementation and a direct JAX semantic
  reference with identical route indices.
- Expected signal: deterministic route/output hashes and stable local timings.
- Falsifier: pinned code cannot run on available SM100 hardware or the full
  semantic boundary cannot be matched.
- Cost/risk: moderate build risk; low algorithm risk.

### 2. Generate the MSA relation from natural JAX

- Minimum experiment: recover index projections, causal token Contract,
  block-max Fold, forced-local top-k Selection, and RelationPlan from StableHLO.
- Baseline/control: compare every selected block set against the reference.
- Expected signal: exact route equality away from deliberate top-k ties.
- Falsifier: recovery needs an MSA-specific node after semantic erasure.
- Cost/risk: moderate frontend work.

### 3. Compare query-major and KV-major generated schedules

- Minimum experiment: use the same generated attention-state body with both
  relation orientations; include real shared-memory KV staging in KV-major.
- Baseline/control: official MSA KV-outer payload and a no-reorientation
  query-major control.
- Expected signal: KV-major wins when selected-block popularity provides enough
  reuse; query-major remains competitive for flatter relations.
- Falsifier: changing orientation requires workload-specific attention code.
- Cost/risk: high physical-schedule work after the helper cleanup.

## Hypothesis Queue Update

- Add: MSA full routed BF16 prefill on SM100 as the primary sparse acceptance
  experiment.
- Revise: native FlashMoBA is the H100 fallback, not the primary cross-platform
  oracle.
- Revise: NSA/FSA becomes the second-stage multi-branch composition test.
- Stop: do not tune against Seer or the local SM80-style block-sparse control.
- Promote later: Kascade for cross-layer relation reuse and FlashMLA for
  MLA-specific token-level routing.

## Source Ledger

| Source | Type | Revision/location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| MiniMax Sparse Attention | paper | [arXiv 2606.13392v2](https://arxiv.org/html/2606.13392) | Natural semantics, model shape, KV-outer schedule, H800 measurements | high | Paper result is not a B200 local result |
| MiniMax-AI/MSA | external code | [`80434d7`](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac) | Public SM100 API, tests, benchmark boundary | high | CUTLASS submodule `eb61c91` |
| FlashMoBA | external code | [`39d9ac0`](https://github.com/mit-han-lab/flash-moba/tree/39d9ac043b271d046a2181a9991e99a26b67bca1) | H100 fallback and native router | high | Existing Shuttle match changes router semantics |
| Native Sparse Attention | paper | [arXiv 2502.11089](https://arxiv.org/html/2502.11089) | Three-branch NSA semantics | high | Separate normalization per branch |
| Flash Sparse Attention | external code | [`7ff144f`](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/tree/7ff144fd7ff485dc4220d439f31cc1708b64fef3) | H100 NSA implementation | medium | Existing local adapter is loose |
| FlashMLA | external code | [`15f13e5`](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6) | SM90/SM100 explicit-index sparse prefill | high | MLA/MQA dimensions mismatch GQA D128 |
| Kascade | external code | [`d4463fc`](https://github.com/microsoft/kascade/tree/d4463fcb4e66507ac7f83d072c43c26932ccc769) | Cross-layer relation reuse | medium | FP16/configuration-specific |
| SeerAttention | external code | [`aba03e3`](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) | Learned-router alternative | medium | Weak current prefill denominator |
| Quest | external code | [`01c1623`](https://github.com/mit-han-lab/Quest/tree/01c1623bf9395009520874e989e29f683203b357) | Index/payload-plane design | high | Decode/Ada focus |
| Current Shuttle sparse artifacts | report/code | `lib/tile_lifetime/benchmarks/artifacts/` | Existing semantic match and baseline caveats | high | Local preserved evidence |

## Handoff

Suggested issue prior-work block:

> MiniMax Sparse Attention is the primary routed sparse-attention oracle. Its
> natural program is index Q/K projection, causal token scoring, block-max,
> forced-local top-k, and exact selected GQA attention. The pinned public SM100
> implementation exposes runtime indices and uses a KV-outer deterministic
> two-phase schedule. Its published benchmark is payload-only, so acceptance
> requires a new matched full routed benchmark on B200/GB200. NSA/FSA follows as
> a multi-branch composition test; native FlashMoBA remains the H100 fallback.

Open questions:

- Does the pinned top-k implementation's tie and forced-local policy exactly
  match the paper's reserved-slot definition for every tail case?
- Which MSA index dimension is best supported by the public BF16 proxy path?
- Does the official code expose full-route timing without host synchronization,
  or should the harness assemble the stages directly under one CUDA graph?
- At what relation skew does Shuttle's KV-major schedule overtake query-major?

Stop reason: after MSA was added, later primary-source candidates did not change
the first experiment or its acceptance boundary.
