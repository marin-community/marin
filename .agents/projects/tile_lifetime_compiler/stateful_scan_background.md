# StatefulScan Background Research

Date: 2026-08-07

## Decision

Use scalar-decay Gated DeltaNet as the first executable `StatefulScan` and Kimi
Delta Attention as the immediate structured-transition stress test.

The two programs share the same matrix-state update shape. GDN has one scalar
decay per value head; KDA has one decay per key channel. Both are affine in the
incoming state and admit exact full-affine chunk summaries. Their efficient
chunk implementations retain diagonal-plus-low-rank/WY factors and scan chunks
in order.

This supports one generic semantic contract:

```text
summarize(chunk) -> affine or factored-affine summary
apply(summary, incoming_state) -> outgoing_state
emit(summary, incoming_state) -> chunk outputs
optional compose(earlier, later)
```

Unrestricted exact composition is closed for a full affine `(P, H)` summary,
but not for a bounded-rank factorization: rank generally grows across tree
levels. `compose` is therefore optional in the physical chunk algebra.

## Gated DeltaNet semantics

Using Shuttle's `[B,H,K,V]` state orientation, the source-ordered update is:

```text
alpha_t       = exp(g_t)
state_decayed = alpha_t * state_(t-1)
prediction    = state_decayed^T @ key_t
delta         = beta_t * (value_t - prediction)
state_t       = state_decayed + outer(key_t, delta)
output_t      = state_t^T @ query_t
```

The decay-before-prediction ordering is essential. Marin's existing JAX
implementation executes this order. Its module-level introductory equation
omits the decay inside the prediction term, which is a documentation mismatch,
not an implementation mismatch.

The per-token affine form is:

```text
P_t = alpha_t * (I - beta_t * outer(key_t, key_t))
H_t = beta_t * outer(key_t, value_t)
S_t = P_t @ S_(t-1) + H_t
```

For two summaries:

```text
(P2, H2) after (P1, H1)
    = (P2 @ P1, P2 @ H1 + H2)
```

The fast chunk algorithm uses cumulative scalar decay and an extended WY/UT
factorization, parallelizes triangular and contraction work inside each chunk,
then carries the matrix state through an ordered chunk scan.

Primary reference: [Gated Delta Networks](https://arxiv.org/abs/2412.06464).

## Kimi Delta Attention fit

KDA changes scalar decay into a per-key-channel vector:

```text
state_decayed[key, value]
    = alpha_t[key] * state_(t-1)[key, value]
state_t
    = state_decayed
      + outer(key_t, beta_t * (value_t - state_decayed^T @ key_t))
```

Its per-token affine transition is:

```text
P_t = (I - beta_t * outer(key_t, key_t)) @ diag(alpha_t)
H_t = beta_t * outer(key_t, value_t)
```

The diagonal decay no longer commutes with the rank-one correction. Efficient
KDA chunks therefore need per-channel cumulative products and a
diagonal-plus-low-rank factorization. A useful physical summary is conceptually:

```text
FactoredAffineTransition {
    diagonal
    low_rank_left
    low_rank_right
    additive_state
    state_layout
    numerical_contract
}
```

This is a lowering representation, not a new KDA semantic operation. It is the
smallest generic concept KDA adds beyond GDN.

Primary references:

- [Kimi Linear paper and model repository](https://github.com/MoonshotAI/Kimi-Linear/tree/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3)
- [FLA KDA recurrent reference](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/kda/naive.py)

## Production GDN shape

Use the pinned Qwen3-Next-80B configuration:

```text
hidden size:        2048
query/key heads:    16
value/state heads:  32
key dimension:      128
value dimension:    128
convolution width:  4
input/output dtype: BF16
state dtype:        FP32
chunk size:         64
```

Each Q/K head serves two value heads. The semantic matrix state is
`[B,32,128,128]`.

Configuration:
[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/config.json).

## Backend and oracle ledger

| Role | Revision | Use |
|---|---|---|
| NVLabs GatedDeltaNet | [`b53d6d3a161267432a79c1c04af69fa52bddc921`](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921) | Original semantic/model reference |
| FLA 0.5.2 | [`9c8e42e762fce087c27b673af4922795d9edb85e`](https://github.com/fla-org/flash-linear-attention/tree/9c8e42e762fce087c27b673af4922795d9edb85e) | FP32 reference, fused recurrent decode, Triton chunk oracle |
| FlashQLA 0.1.2 | [`050c6bbee9e03efbbfe41063fe4e33742c4a87cb`](https://github.com/QwenLM/FlashQLA/tree/050c6bbee9e03efbbfe41063fe4e33742c4a87cb) | SM90/SM100 fused chunk oracle |
| Kimi Linear | [`8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`](https://github.com/MoonshotAI/Kimi-Linear/tree/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3) | KDA equations/model |
| FLA KDA | [`3c4c54ae7397d37130d7101edd0f4eb596af896d`](https://github.com/fla-org/flash-linear-attention/tree/3c4c54ae7397d37130d7101edd0f4eb596af896d) | KDA recurrent/chunk operational reference |
| FlashKDA | [`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`](https://github.com/MoonshotAI/FlashKDA/tree/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b) | SM90+ KDA prefill oracle |

FLA fused recurrent supports grouped value heads, in-kernel Q/K normalization,
FP32 state, and one-token decode. FLA chunk supports chunk sizes 16, 32, and 64,
initial/final state, grouped value heads, and variable lengths.

FlashQLA supports SM90 and SM100, CUDA 12.8+, PyTorch 2.8+, BF16/FP16, grouped
value heads, and fixed `K=V=128`. Its public API aligns with FLA's chunk API.
The pinned repository reports 0.283 ms on H200 and 0.285 ms on GB200 for
`B=1,T=8192,Hq=16,Hv=32,K=V=128`; these are vendor measurements, not Shuttle
results.

FlashKDA is narrower: SM90+, CUDA 12.9+, BF16 `K=V=128`, forward inference
only. Its running state is stored in BF16 between FP32 updates in the published
schedule, so it requires a different numerical contract from the FP32-state
reference.

## Initial measurements

Correctness lock:

```text
B=1, T<=256, Hq=16, Hv=32, K=V=128
BF16 Q/K/V
FP32 decay, beta, and nonzero initial state
```

Compare independent Shuttle recurrence, FLA recurrent, FLA chunks 16/32/64,
and FlashQLA. Preserve maximum, mean, p99, RMS, finite counts, bitwise repeat,
and output/final-state hashes.

Performance:

```text
decode:  T=1, B in {1,4,16}, FLA fused recurrent
prefill: B=1, T in {2048,8192}, chunk 64, FLA and FlashQLA
```

Only add 32768 after the smaller runs are stable.

## StableHLO frontend probe

An ordinary JAX `lax.scan` of the GDN core lowers to:

```text
transpose time-major inputs
stablehlo.while
    dynamic_slice each current item
    func.call the recurrence body
    dynamic_update_slice the output
transpose outputs back
```

The recurrence body itself remains recognizable: exponential decay,
broadcast/multiply of the FP32 state, two `dot_general` contractions, the
delta map, and one outer-product `dot_general`.

The current Shuttle importer intentionally handles only a single entry block
and rejects `stablehlo.while` and private `func.call` bodies. Recovering an
ordinary exported scan therefore requires one narrow but real frontend feature:
import structured while regions and called private functions with stable value
identities. Static Python-loop unrolling would avoid that work for fixtures but
would not be an honest scalable recovery path.

This frontend gap does not affect the semantic/physical abstraction test, but
it remains required before claiming ordinary JAX-to-StatefulScan recovery.

## Caveats

- Calling FLA or FlashQLA validates backend integration and supplies an oracle;
  it is not evidence that Shuttle synthesized their complete WY schedule.
- The current Marin tests use mostly small FP32 inputs and do not establish
  production-shape BF16 H100 behavior.
- Strongly negative gates can hide cross-chunk state errors. Correctness tests
  must include zero, mild, and strong decay regimes.
- FlashQLA has reported token-level divergence despite small average error, so
  distributional errors and final state must be saved.
- A full GDN/KDA layer has projections, short convolution caches, gate
  formation, output normalization, and output projection around the matrix
  scan. The first slice validates only the matrix-state core.
- KDA's full layer has additional persistent convolution state. That tests
  multiple coupled persistent states later; it does not invalidate the core
  matrix `StatefulScan`.
