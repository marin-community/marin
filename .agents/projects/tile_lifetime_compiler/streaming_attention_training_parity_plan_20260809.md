# Streaming-attention training parity plan

## Acceptance boundary

The source is one ordinary JAX function. JAX owns differentiation through
`jax.vjp`; Shuttle does not author a model VJP. The exported function accepts:

- BF16 `Q[B,S,Hq,D]`;
- BF16 `K[B,S,Hkv,D]` and `V[B,S,Hkv,D]`;
- BF16 output cotangent `dO[B,S,Hq,D]`.

It returns BF16 `(O, dQ, dK, dV)`. The primary fixed configuration is
`B=1, S=2048, Hq=32, Hkv=8, D=128`, scale `D**-0.5`, and the independent
domain predicate `key_position <= query_position`. The GQA relation maps four
query heads to each K/V head. There is no dropout, bias, softcap, variable
length, or padded tail in the primary comparison.

StableHLO recovery assigns roles using shapes, index relations, operation
structure, and input dependencies. In particular, it distinguishes the
forward output from `dQ` because only `dQ` depends on `dO`. No model or
attention implementation name participates in physical dispatch.

## Generated physical family

The existing compiler-owned streaming family is instantiated from recovered
Contract, Map, normalized-exponential Fold, and DomainRestriction semantics:

1. a generated forward streams K/V over resident query rows and emits BF16
   output plus internal FP32 log-sum-exp state;
2. a generated query-major reverse emits `dQ` and the internal output-dot Fold;
3. a generated K/V-major reverse emits deterministic `dK` and `dV`.

The forward output buffer is now a direct FFI result rather than discarded
scratch. Log-sum-exp and output-dot remain internal state. The runtime handler
contains emitted CUBIN launchers and CUDA/JAX FFI primitives only; it imports
neither Torch nor Triton and invokes no opaque FlashAttention semantic call.
Triton remains a build-time compiler for the generic physical skeleton.

The training result policy is distinct from the reverse-only and externally
saved-state policies. Forward-plus-gradients is legal only when forward state
is produced inside the same call. This prevents accidentally comparing a
generated recompute boundary against an oracle that receives free saved state.

## Matched expert comparison

The expert is PyTorch Flash-SDPA forced to the flash backend, with cuDNN,
memory-efficient, and math SDPA disabled. One timed call executes forward and
`torch.autograd.grad`, returning output and all three gradients. Torch owns its
internal saved state exactly as Shuttle owns the log-sum-exp produced by its
forward stage. Torch is benchmark/oracle-only.

Both boundaries consume the same JAX-owned BF16 buffers. Torch makes zero-copy
`B,H,S,D` views. Generated outputs request logical `B,S,H,D` with physical
minor-to-major layout `(D,S,H,B)`, matching contiguous Torch `B,H,S,D` results
without a timed repack. The artifact records both requested XLA layouts and
Torch strides. Correctness converts only after synchronization and outside the
timed region.

## Numerical and mutation audit

The recovered program records JAX's equal-split max derivative. The generated
kernel applies the normalized-exponential maximum-VJP cancellation under
`allow_rounding_reorder`; this is not bitwise/source-ordered equivalence.
Artifacts report maximum and mean absolute error for all four results against
the natural JAX function, plus the expert's errors and a repeated generated
output hash. The existing scale mutation (`0.5` to `0.375`) changes generated
specializations and fingerprints while preserving the same physical family.
The causal restriction remains an independent generated predicate.

## Static gate and measurements

Before GPU allocation:

- the natural four-result StableHLO must recover all six live Contracts and
  all generic Fold/DomainRestriction roles;
- the reverse-only path must remain unchanged;
- direct-result output layout must reach the generated forward specialization;
- result/save-state policy mismatches must fail before FFI dispatch;
- focused recovery/generation tests and repository pre-commit must pass.

After review, run one fixed expert-derived configuration on H100, then the same
configuration on GB200 if the generated AOT family is portable. Use
counterbalanced repeated samples and preserve every sample, correctness result,
deterministic hash, source/compiler revisions, generated handler/CUBIN hashes,
and environment metadata. Do not tune workload-specific semantics. If parity
is not within 1.20x, attribute the gap using the already separate forward, dQ,
and dK/dV generated stages and an expert kernel timeline; do not change the
semantic body to chase the oracle.
