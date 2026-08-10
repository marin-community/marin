# Dense and MoE frontend provenance audit

## Background research brief

- Effort: medium, internal code and frozen-artifact audit.
- Audited revision: `0d9bc70082d969bdfb8f517530b738fefb8a7730`.
- Stop rule: stop once every public Dense/MoE entrypoint, current benchmark
  caller, generic post-SPMD importer, and physical-selection boundary had an
  unambiguous source-to-schedule classification.
- Scope exclusion: routed-attention and relation-transport implementation.

## Question

Do the Dense and MoE prototype claims begin at ordinary JAX-derived IR and
reach physical generation through one generic, workload-name-free recovery
boundary, or do exact workload recognizers still participate in accepted
scheduling?

## Ranked findings

### 1. MoE's old public pipeline is an exact named reference planner

`pipeline.compile_stablehlo_expert_parallel_region` imports StableHLO through
`moe_recovery.recover_moe_region`. That recognizer requires an exact eight-input,
three-output shared-plus-routed program and reconstructs these named IR classes:

```text
LinearOp
TopKRouterOp
SharedExpertMLPOp
RoutedExpertMLPOp
WeightedExpertCombineOp
```

`expert_parallel._recover_region` then requires exactly those classes and uses
their roles to build the schedule. The generated Map and Fold bodies are generic
scalar ASTs, and the selected GMM and transport implementations are generic
physical mechanisms, but there is no intervening generic semantic program or
`SemanticErasureReport`. This is erasure during physical-plan construction, not
machine-checked erasure before candidate selection.

Verdict: the path is useful reference/fixture planning, but it is not eligible
for a current clean-synthesis claim under the shared-importer criterion.

### 2. Dense erases names before scheduling, but its recovery is still one exact whole-region recognizer

`semantic_recovery.recover_dense_transformer_region` requires a ten-input,
four-output bounded Llama-shaped region and walks the expected QKV, RoPE,
attention, residual, RMS, SwiGLU, and following-QKV structure. It reconstructs
named frontend operations.

Unlike MoE, `dense_region.compile_dense_transformer_region` immediately calls
`dense_flow.erase_dense_transformer_semantics`. That pass lowers every named
operation into `FlowContract`, `FlowMap`, `FlowFold`, and
`FlowDomainRestriction`, derives workload-name-free scheduling keys, and
validates the erasure before `dense_flow_planner` selects generic GEMM and
generated-streaming skeletons. No official FA, CODA, or other named semantic
kernel is selected by this path.

That is a clean scheduling boundary in isolation, but it still does not meet
the stronger shared-plugin requirement: region membership was established by
an exact whole-workload recognizer rather than by the common HLO dataflow
importer. This checkpoint therefore carries the artifact hash and source
operation IDs but labels the result `experimental_exact_recognizer`; the
current-acceptance guard rejects it until its regions come from generic HLO
dataflow.

### 3. The generic Grug MoE path already exists at post-SPMD HLO

`xla_hlo_recovery` parses entry computations, inlines elementwise fusions, and
recovers Contract/Map regions from opcodes, shapes, dimension numbers, and data
dependencies. It explicitly ignores frontend metadata and model names.

`xla_relation_program_recovery` builds on the same graph to recover:

```text
Selection -> RelationPlan
SegmentedContract -> generated scalar Map -> SegmentedContract
source-keyed Fold
input-adjoint Contract/Map/Contract/Fold
group-batched weight-gradient Contract
external collective boundaries
```

The frozen natural Grug HLO tests form, generate, numerically interpret, and
replace forward, input-adjoint, and weight-gradient regions. Mutation tests
change HLO scalar algebra and regenerate different CUDA source without a model
key. JAX owns AD; collectives remain explicit HLO boundaries.

Verdict: this is the correct basis for the shared Shuttle plugin. It remains an
experimental HLO-text attachment bridge, but its provenance and structural
recovery are substantially cleaner than the exact StableHLO MoE planner.

### 4. The preserved GB200 MoE natural-boundary benchmark overstates frontend ownership

The current benchmark source compiles the exact StableHLO plan, then separately
implements runtime router execution as Torch `mm -> topk -> softmax` in
`NaturalRouterRuntime`. The recovered plan supplies dimensions and relation
configuration; it does not instantiate that runtime router program. The timing
comparison can still be a valid matched performance measurement, but it is not
evidence that one compiler-owned JAX/HLO program generated the complete runtime
frontend.

The frozen artifacts remain immutable historical evidence. New acceptance
claims should use the post-SPMD natural Grug module and record the transformed
module, recovered generic regions, generated source, and runtime boundary from
the same compilation.

## Exact recognizer versus generic recovery matrix

| Surface | Source boundary | Region formation | Schedule input | Status |
| --- | --- | --- | --- | --- |
| Dense StableHLO pipeline | Frozen/natural JAX StableHLO | Exact bounded dense recognizer | Erased Flow Contract/Map/Fold/DomainRestriction | Experimental; generic scheduling, exact frontend |
| MoE StableHLO pipeline | Frozen/natural JAX StableHLO | Exact shared-plus-routed recognizer | Named MoE operation classes | Reference only |
| Pair/contract-map HLO recovery | Post-SPMD JAX/XLA HLO | Opcode, shape, dimensions, dataflow | Generic Contract/Map regions | Current shared-importer candidate |
| Relation-program HLO recovery | Post-SPMD JAX/XLA HLO | Opcode, shape, relation ancestry, dataflow | RelationPlan/SegmentedContract/Map/Fold | Current shared-importer candidate |
| GB200 natural-boundary runtime | StableHLO fixture plus Torch runtime | Exact plan plus handwritten router | ExpertParallelPlan and benchmark-specific runtime | Performance reference, not frontend acceptance |

## Bounded cleanup in this checkpoint

The Dense pipeline now returns a structured compilation containing:

- the recovered `RegionPlan`;
- StableHLO artifact SHA-256;
- all matched source-operation IDs; and
- `FrontendCompilationStatus.EXPERIMENTAL_EXACT_RECOGNIZER`.

Validation fails closed for hand-authored provenance, missing source-operation
evidence, named schedule keys, and any attempt to treat the exact recognizer as
the current generic-HLO path. Existing runtime and benchmark callers explicitly
unwrap `.plan`, so provenance is no longer silently discarded at compilation.

The overlapping MoE public-surface migration was deliberately not changed here.
The modernization owner is editing `pipeline.py`, `reference_pipeline.py`, and
the package exports concurrently.

## Requested MoE quarantine patch for the modernization owner

Apply this as one API migration, with no compatibility aliases:

1. Move `recover_stablehlo_moe_region` and
   `compile_stablehlo_expert_parallel_region` out of `pipeline.py`.
2. Expose them only from `reference_pipeline.py` as
   `recover_reference_stablehlo_moe_region` and
   `compile_reference_stablehlo_expert_parallel_region`.
3. Remove `compile_expert_parallel_region`, `recover_moe_region`, and the two
   StableHLO wrappers from the package root. Reference tests should import their
   defining modules explicitly.
4. Mark the current GB200 exact-fixture caller
   `frontend_compiler_status=exact_named_reference_only`; do not use it for the
   frontend acceptance column.
5. Route the current Grug MoE claim through
   `xla_relation_program_recovery` and its generated typed-FFI region plans.
   Keep JAX-produced collectives outside the owned region until generic
   Transport integration is selected.

## `lib/shuttle` migration inventory

Move or extract into the current compiler package:

- the generic HLO graph/parser/inliner currently in `xla_hlo_recovery`;
- generic HLO scalar Map import;
- generic relation-program recovery and region boundary records;
- generic semantic-erasure validation and the shared TensorProgram algebra;
- typed module-transform/plugin entrypoints that consume the recovered generic
  regions; and
- one provenance/acceptance envelope shared by Dense, MoE, attention, and Scan.

Leave in `tile_lifetime` as reference, oracle, or research compatibility
surfaces:

- `semantic_recovery.recover_dense_transformer_region`;
- `moe_recovery`;
- the named `expert_parallel` planner;
- `dense_flow` until its useful pieces are subsumed by the shared TensorProgram
  importer rather than copied as a second Dense IR;
- `reference_pipeline`, `reference.py`, `moe_reference.py`;
- named oracle adapters and frozen benchmark artifacts; and
- benchmark-only Torch execution scaffolds.

Do not scaffold this migration from this audit branch. The modernization work
owns the package inventory and should move each module only after its public
entrypoint is covered by the shared acceptance harness.

## Shared acceptance harness requirements

For each of Dense, MoE, attention, and Scan, require one structured record with:

1. source provenance: JAX export or post-SPMD HLO digest and source region;
2. generic recovery: exact generic operations, boundaries, and numerical policy;
3. module transformation: inserted targets plus proof that only the recovered
   region was replaced; and
4. execution evidence: generated-source digests, independent numerical result,
   matched runtime boundary, and explicit external collectives.

An exact named recognizer or separate benchmark runtime may remain a diagnostic
oracle, but cannot populate any of these four current-path fields.

## Negative and rejected leads

- Adding a semantic-erasure report after `expert_parallel._recover_region`
  would not fix MoE provenance: the named classes already selected the schedule
  structure.
- Treating Dense's Flow erasure as sufficient for the new goal would preserve a
  workload-specific whole-region finder. It is retained as an experimental
  baseline, not promoted.
- Wrapping the existing post-SPMD functions in another `pipeline.py` helper
  would add an alias, not a shared importer or compiler plugin.
- Rewriting frozen acceptance artifacts would destroy their value as historical
  evidence; only future artifact claims should change.

## Recommended next experiments

### 1. Promote one natural Grug MoE module through the shared plugin

- Minimum experiment: recover forward, input adjoint, and two weight-gradient
  regions from one post-SPMD module; insert generated targets; preserve
  collectives; execute one CPU or GPU correctness case.
- Falsifier: any region requires a model name, frontend metadata, or an exact
  MoE operation class.
- Confidence: high; the component tests already cover each region family.

### 2. Replace Dense whole-region discovery with generic region composition

- Minimum experiment: on a natural Grug HLO module, compose generic Contract/Map
  regions, normalized-exp Contract/Fold regions, row Folds, and domain
  restrictions into the same generated skeleton family used by the current
  dense prototype.
- Falsifier: candidate selection requires a Dense/Transformer/Llama key rather
  than only generic operation and boundary structure.
- Confidence: medium; generic component recovery exists, but whole-region
  composition and one module transform remain to be demonstrated together.

## Source ledger

| Source | Type | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- |
| `pipeline.py` at `0d9bc70082` | Marin code | Public Dense/MoE entrypoint trace | High | Direct call graph |
| `semantic_recovery.py` at `0d9bc70082` | Marin code | Dense and MoE-adjacent exact recognizer behavior | High | Exact signatures and expected subgraphs |
| `dense_flow.py` and `dense_flow_planner.py` at `0d9bc70082` | Marin code | Dense name erasure precedes scheduling | High | Structural erasure report and generic schedule keys |
| `moe_recovery.py` and `expert_parallel.py` at `0d9bc70082` | Marin code | Named MoE classes drive planning | High | No intervening generic program/report |
| `xla_hlo_recovery.py` at `0d9bc70082` | Marin code | Generic opcode/shape/dependency recovery | High | Explicitly ignores metadata/model names |
| `xla_relation_program_recovery.py` at `0d9bc70082` | Marin code | Generic Grug routed forward/backward recovery | High | Relation, Contract, Map, Fold, external collectives |
| `test_xla_relation_program_recovery.py` at `0d9bc70082` | Marin test | Natural Grug HLO mutation, replacement, and numerical evidence | High | Uses frozen pre-scheduler HLO |
| `gb200_deepep_mok_distributed.py` at `0d9bc70082` | Benchmark code | Runtime router is handwritten Torch | High | Plan is used for validation/configuration |
| `CURRENT_ARCHITECTURE.md` at `0d9bc70082` | Design ledger | Existing quarantine and remaining cleanup | High | Already identified named MoE path as debt |

## Handoff

The immediate safe merge is the Dense experimental provenance envelope and its
fail-closed tests. The MoE quarantine should land with the concurrent
modernization API move. The next accepted proof should come from one shared
post-SPMD importer/plugin and one four-workload acceptance harness, not from
adding more workload-specific wrappers to `pipeline.py`.
