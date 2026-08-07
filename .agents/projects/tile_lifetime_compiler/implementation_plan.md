# Tile-Lifetime Compiler: First Vertical Slice

The project mission, Flow model, and current first-principles MoE definition of done are defined in [Shuttle: Prototype Brief and Big-Picture Specification](shuttle_brief.md). When this historical implementation plan is narrower or uses older MoK-oriented wording, the Shuttle brief is authoritative.

Status: implemented on 2026-08-05. This first vertical slice led to the connected dense-region compiler and executable H100 oracle documented in `lib/tile_lifetime/docs/progress.md`; recognizer variation remains future work.

## Outcome

Compile a small normalized tensor graph for:

```python
h = linear(x, weight_0)
u = h + residual
n = rms_norm(u, gamma, epsilon)
y = linear(n, weight_1)
```

into an inspectable region plan containing:

```text
GEMM 0 epilogue: residual add, gamma multiply, RMS sum-of-squares partial
Auxiliary reduction: combine partials and compute inverse RMS
GEMM 1 epilogue: delayed row scale
```

The plan must state that moving the BF16 row scale changes floating-point operation ordering.

## Package boundary

The prototype lives in the isolated `lib/tile_lifetime/` workspace package. It does not add a dependency from Levanter onto the compiler, and GPU backends remain optional.

## Public data model

```python
graph = TensorGraph()
x = graph.input("x", shape=(tokens, hidden), dtype=DType.BF16)
h = graph.linear(x, weight_0, accumulation_dtype=DType.FP32)
u = graph.residual_add(h, residual)
n = graph.rms_norm(u, gamma, epsilon=1e-6, reduction_dtype=DType.FP32)
y = graph.linear(n, weight_1, accumulation_dtype=DType.FP32)

result = compile_region(graph)
print(result.explain())
```

`compile_region` returns a selected plan and structured rewrite explanations. Values record their disposition as materialized, aliased, recomputed, epilogue-only, partial-reduction-only, or internal attention state.

## Legality checks

The delayed-RMS rule requires:

- the normalization input to be the residual sum produced after the first GEMM;
- the RMS reduction to cover the complete hidden dimension;
- gamma and inverse RMS to broadcast as row-independent and row-scalar factors respectively;
- exactly one consumer of the canonical normalized activation;
- a following right-multiplication GEMM;
- declared accumulation and reduction dtypes;
- an explicit numerical policy permitting reordered BF16 rounding.

The first slice rejects unsupported graphs with structured reasons and preserves a materialized fallback plan.

## Dense-to-expert-parallel continuation

The dense H100 path is the executable substrate for first-principles expert-parallel synthesis. Mixture-of-Kittens is a pinned GB200 correctness and performance oracle; it is not the compiler backend and its megakernel task graph is not a frontend primitive.

The MoE compiler starts from an ordinary global routed-MoE graph and applies five generic lowering layers:

1. **Route relation.** Convert top-k results into a relation with `(source_token, route_slot, global_expert, weight)` fields. Derive expert ownership from the declared expert-axis partition instead of assuming rank-local expert indices in the semantic graph.
2. **Segmented contraction.** Group the relation by `(owner_rank, local_expert)`, pad legal expert segments, and express gate, up, and down projections as contractions over segmented rows. Shared experts remain ordinary dense contractions.
3. **Tile flow.** Tile dispatch, segmented contractions, SwiGLU, reverse exchange, and weighted scatter-reduction. Represent tile values, layouts, consumer counts, and readiness granularity independently of CUDA barriers or MoK event names.
4. **Buffering.** Derive symmetric send/receive buffers, expert-grouped activation buffers, double/ring buffering, aliases, lifetimes, and capacity bounds from the tile-flow graph.
5. **Scheduling.** Assign generic communication, producer, consumer, and epilogue worker roles; choose multi-kernel or persistent execution; and tune communication workers, tile sizes, minibatches, macrobatches, and pipeline depths from a bounded legal family.

The first executable baseline may use existing grouped-GEMM, all-to-all, and tile primitives, but plan construction must not select a pre-fused MoK region operation. A later backend may generate a persistent kernel from the generic tile-flow schedule. The performance target is within 20–30% of the pinned MoK BF16-forward oracle on the same four-GB200 configuration.

The initial structural slice must therefore expose:

- a global semantic expert axis and an explicit rank-local partition relation;
- route, partition, segment, pad, exchange, segmented-GEMM, SwiGLU, reverse-exchange, and weighted-scatter stages;
- tile-flow edges with layouts, readiness granularity, and fan-out/fan-in;
- derived buffers and reuse constraints;
- schedule choices and legality arguments; and
- a lowering trace showing which generic transform created every stage.

The already recovered MoK task/event graph is retained only as an oracle description used to compare boundaries, resource choices, and measured schedules. H100 dense and GB200 expert-parallel results remain separate hardware baselines.

## Tests

- A legal graph produces two GEMMs, one auxiliary reduction, and no materialized normalized activation.
- An extra consumer of the normalized value disables delayed scaling and produces a materialized fallback.
- A wrong normalization axis disables the rewrite.
- Strict bitwise policy disables the rewrite.
- The algebraic reference is equal in FP64, while a BF16 emulation records nonzero deviation and stays under a declared test tolerance.
