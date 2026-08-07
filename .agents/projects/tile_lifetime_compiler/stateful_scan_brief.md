# Shuttle StatefulScan Prototype Brief

Date: 2026-08-07

## Goal

Test whether Shuttle can represent one stateful linear-attention program as a
generic ordered scan and recover both of its useful execution forms:

```text
recurrent decode
chunkwise parallel prefill / training
```

Begin with Gated DeltaNet because Marin already has independent JAX recurrent
and chunkwise implementations. Use Kimi Delta Attention as the next stress test
only after the GDN representation is executable and its generic boundaries are
clear.

The compiler experiment is successful when the semantic node is `StatefulScan`,
not `GatedDeltaNetKernel`, and the workload-specific content is confined to the
state update, readout, and chunk-summary algebra.

## Research question

Can one dependency-free Shuttle representation describe:

- an ordered token recurrence with persistent state;
- a chunk summary that transforms an incoming state;
- composition or sequential application of chunk summaries;
- output emission that combines incoming state with within-chunk work; and
- candidate selection between recurrent and chunkwise physical backends?

## Semantic target

The initial Gated DeltaNet state is a rectangular matrix per batch and head:

```text
S_t = alpha_t S_(t-1)
      + k_t [beta_t (v_t - S_(t-1)^T k_t)]^T
o_t = S_t^T q_t
```

The source program must retain explicit FP32 state/update semantics and the
normalization/scaling applied to Q and K. The chunkwise form is a lowering of
this ordered program, not a distinct semantic operator.

Conceptually:

```text
StatefulScan {
    ordered_axis
    state
    initialize
    update(state, item)
    read(state, item)
    numerical_contract
    optional chunk_algebra
}

ChunkAlgebra {
    summarize(chunk)
    apply(summary, incoming_state)
    emit_outputs(summary, incoming_state)
    optional compose(earlier, later)
}
```

Do not require `compose` to be closed over a compact summary until the GDN
experiment proves that such a representation is useful. A sequential scan over
chunk transforms is still a valid chunkwise lowering because parallel work
inside each chunk removes the token-by-token critical path.

## Required candidates

1. Recurrent candidate: source-ordered token scan, suitable for decode.
2. Chunkwise candidate: parallel in-chunk transform plus ordered scan over
   chunk summaries, suitable for prefill/training.

The selected candidate must expose its state layout, accumulator dtype,
materializations, chunk size, backend, and finite-precision relationship to the
source recurrence.

## Initial implementation boundaries

- Keep the semantic and plan records in `lib/tile_lifetime` free of Levanter,
  Haliax, CUDA, and Triton types.
- Use a small NumPy executor as an independent semantic oracle.
- Adapt the existing Levanter JAX GDN functions only in a benchmark/backend
  boundary.
- Compare with the official or de facto expert implementation after pinning its
  exact revision.
- Do not begin an XLA integration or introduce MLIR dialects.
- Do not add a KDA-specific semantic node.

## Correctness requirements

- Recurrent execution matches an independent step-by-step reference.
- Chunkwise execution matches the recurrent program under an explicit
  numerical tolerance and reassociation policy.
- Prefix/suffix continuation through persistent state matches one-pass
  execution.
- Non-divisible sequence lengths and nonzero initial state work.
- Extreme decay and update gates remain finite for supported inputs.
- Plan dumps explain every numerical reordering.

## Performance questions

- Decode: how close is the recurrent candidate to the expert recurrent kernel?
- Prefill: how close is the chunkwise candidate to the expert chunkwise kernel?
- Which costs come from the physical primitive versus Shuttle's chosen
  materializations or boundaries?
- Does the planner select recurrent execution for one-token decode and
  chunkwise execution for long prefill without a model-name special case?

## Generality accounting

Classify changes as:

- reused unchanged from existing Shuttle machinery;
- generalized Shuttle state/materialization machinery;
- new generic `StatefulScan` or `ChunkAlgebra` machinery;
- GDN-specific semantic recovery;
- GDN-specific physical backend code.

The experiment fails its abstraction test if the executable plan requires a
semantic `GatedDeltaNet` node or if most scheduling logic is keyed on the model
name rather than the ordered-state properties.

## Definition of done

1. A generic `StatefulScan` describes the GDN recurrence.
2. Recurrent and chunkwise candidates lower from the same semantic record.
3. Both candidates execute and match an independent reference.
4. At least one realistic H100 workload is benchmarked against a pinned expert
   implementation for decode and prefill.
5. The selected plan and complete candidate set are preserved.
6. The report states whether KDA fits the same representation and identifies
   the smallest missing generic concept if it does not.
