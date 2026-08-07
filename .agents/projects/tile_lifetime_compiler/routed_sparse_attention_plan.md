# Routed Sparse Attention: RelationPlan Reuse Experiment

Status: next prototype after the `shuttle-gb200-moe-v1` checkpoint.

## Question

Does Shuttle's existing runtime-relation machinery describe routed sparse computation in general, or did it accidentally encode MoE?

The experiment should compile a MoBA-like block-selection program into both query-major and KV-major exact sparse-attention plans. The important result is the amount of compiler machinery reused unchanged. Performance matters after both orientations are structurally and numerically correct.

## Constraint

Do not add a `MoBA` semantic operation or call a complete FlashMoBA kernel as the generated result. An expert sparse-attention implementation may be an oracle and may contribute low-level QK/PV or data-movement primitives.

The first slice should use the existing Python IR, bounded candidate generation, and standalone tests. It should not add a dialect, XLA patch, or general sparse framework.

## Existing pieces to reuse

- `RelationPlan` stable grouping, inverse mapping, padding, ownership, and capacity checks.
- `StreamingAttentionSkeleton` and its FP32 online maximum, denominator, and weighted-value state.
- Expert-parallel tile-flow edges, buffer lifetime derivation, readiness granularity, and worker-pool vocabulary.
- Materialization dispositions and numerical policies.
- Snapshot-style structural and differential tests.

One expected pressure point is that `RelationPlan` is mechanically generic but still uses MoE names such as `route_slot`, `weight`, and `weighted_merge`. Do not rename these preemptively. Build a sparse-attention adapter first, record which fields transfer unchanged, and generalize only the concepts that the experiment actually exercises.

## Slice 1: CPU semantic and algebra test

Construct a deterministic debug workload with a small number of query blocks and KV blocks. Each query block selects a fixed number of KV blocks. The reference computes dense attention with all unselected score blocks masked to negative infinity.

Implement a structured partial state:

```python
@dataclass(frozen=True)
class AttentionPartial:
    row_max: np.ndarray
    row_sum_exp: np.ndarray
    weighted_value_sum: np.ndarray
```

Implement only two algebraic operations:

```python
def summarize_kv_block(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionPartial: ...

def merge_attention_partials(left: AttentionPartial, right: AttentionPartial) -> AttentionPartial: ...
```

The merge rescales both partials to a shared maximum, then combines denominator and value state. Finalization divides the weighted-value sum by the denominator.

Required tests:

- query-major incremental updates equal the masked dense reference;
- merging KV-block partials in stable selection order equals query-major updates;
- a tree merge is allclose under a declared reassociation policy;
- empty, tail, duplicate-selection, causal, and uneven-degree cases are explicit;
- no score or probability value has shape proportional to total query length times total KV length.

## Slice 2: RelationPlan adapter

Map sparse attention onto the existing binary relation:

```text
source_item      = query block
route_slot       = selected-block slot
destination_item = KV block
edge attributes  = causal/bias metadata
```

Begin with one logical placement rank so the test isolates relation orientation. Then assign KV blocks to multiple logical ranks and validate coalesced transport and inverse mapping without requiring a GPU collective.

Query-major plan:

```text
query block
→ visit selected KV blocks in stable slot order
→ extend one AttentionPartial
→ finalize
```

KV-major plan:

```text
RelationPlan groups query blocks by KV block
→ stage one K/V block
→ compute one partial per incident query block
→ inverse-dispatch partials
→ merge by query block in stable slot order
→ finalize
```

Do not use `weighted_merge`; attention needs a structured state merge. The smallest likely generalization is a stable source-grouped merge hook over inverse-dispatched edge values. Add it only after the adapter test demonstrates the exact required API.

Exit condition: both orientations produce the same selected-block semantics and the plan dump exposes relation degree, orientation, grouping, padding, inverse mapping, and merge order.

## Slice 3: Physical plan candidates

Generate exactly two initial candidates:

1. Query-major sparse Fold: one resident query block streams selected K/V blocks.
2. KV-major RelationProgram: one staged K/V block feeds grouped query work and returns partial states.

Each candidate should declare:

- Q/K/V and partial-state layouts;
- task families for staging, QK, online update, PV, inverse routing, merge, and finalize;
- buffer capacities and lifetimes;
- readiness conditions derived from incident-edge counts;
- materialized values and their byte counts;
- kernel boundaries;
- numerical/reassociation policy.

The initial search space may be small:

```text
orientation: query-major, KV-major
Q block:     64, 128
KV block:    64, 128
group bucket: exact degree, padded degree bucket
buffer depth: 2, 4
```

Prune candidates whose partial-state traffic or padding is obviously worse than query-major execution. Do not build a broad cost model yet.

## Slice 4: Executable baseline

Start with a single GPU and BF16 Q/K/V with FP32 attention state. Reuse an existing QK/PV tile primitive where convenient, but retain Shuttle's relation orientation, task decomposition, partial-state merge, materializations, and kernel boundaries as visible plan decisions.

Benchmark against:

- a dense masked reference for correctness only;
- a straightforward query-major block-sparse implementation;
- FlashMoBA or another pinned expert implementation if its supported shape and environment are reproducible.

Only after the single-GPU plans work should KV-block placement be split across devices and `Reshard`/transport candidates introduced.

## Generality accounting

Every change should be classified in the experiment report:

| Category | Expected examples |
|---|---|
| Reused unchanged | stable grouping, inverse map, capacity checks, online state fields |
| Generalized existing machinery | source-grouped structured merge, relation edge attributes |
| New generic machinery | orientation-aware relation program, if actually required |
| Workload-specific recovery | selected KV-block semantics, causal block legality |
| Workload-specific backend | grouped QK/PV tile body and attention layouts |

The experiment fails its architecture test if most compiler changes land in the last two categories or if the cleanest implementation requires a `MoBA`-specific plan node.

## Definition of done

The first routed-sparse-attention phase is complete when:

1. An ordinary selected-block attention program normalizes into Relation plus Fold/Contract structure.
2. Query-major and KV-major plans are emitted from the same relation.
3. Both plans match a masked dense reference under an explicit numerical policy.
4. The KV-major plan stages each selected K/V block once per group and merges exact online-softmax partial states by query.
5. Plan dumps expose layouts, materializations, buffers, readiness counts, and kernel boundaries.
6. At least one executable GPU candidate is benchmarked against a pinned expert or straightforward sparse baseline.
7. The report quantifies how much MoE relation/schedule code was reused unchanged.

The first implementation step is the CPU `AttentionPartial` merge plus a `RelationPlan` adapter test. That is the cheapest experiment capable of falsifying the proposed reuse before any backend work.
