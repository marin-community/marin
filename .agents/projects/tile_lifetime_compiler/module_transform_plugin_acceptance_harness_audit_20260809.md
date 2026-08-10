# Shuttle module-transform plugin and acceptance-harness audit

> Historical audit: this document predates the MLIR-extension decision. Its
> inventory remains useful, but its serialized-HLO callback and typed-HLO
> importer proposals are not the production architecture. The current design
> inserts native Shuttle MLIR passes inside XLA before StableHLO-to-HLO
> conversion; see `shuttle_mlir_extension_design_20260809.md`.

## TL;DR

- Audit revision: `f3ce4aac3c`
- Effort: medium internal audit
- Stop rule: stop after every current `xla_*` source module and module-transform
  harness is classified, and further source reads no longer change the proposed
  target-1 API or first implementation slice.
- Result: the repository already has most of the mechanisms needed for target 1,
  but they are joined inside workload-specific benchmark scripts. There is no
  workload-free compiler plugin, module/region identity, common legality and
  coverage ledger, common handler manifest, or four-way acceptance record.
- Recommendation: build one `shuttle` module-transform session around a lossless
  typed module, generic region-provider plugins, an explicit staged compilation
  record, and one shared acceptance harness. The first vertical slice should
  reuse the generic Contract/Map forward-and-reverse path. Do not promote the
  2,198-line Grug harness or any `xla_*` regex rewriter into the public API.

This is a design and inventory checkpoint. It does not add `lib/shuttle`, change
the current rewriters, or run a GPU benchmark.

## Question

What is the smallest workload-free JAX/XLA module-transform API and acceptance
harness that can become target 1 for Shuttle, while retaining the useful parts
of the current prototypes and leaving workload-specific bridges in
`tile_lifetime`?

The target is one plugin path:

```text
JAX-owned primal and VJP modules
  -> lossless typed HLO import
  -> generic algebra recovery
  -> algebraic rewrites
  -> task decomposition
  -> exact dependencies
  -> EventTensor candidates where useful
  -> physical candidates
  -> selected generated handlers
  -> audited module rewrite
  -> JAX compilation
```

Attention, MoE, Dense, and Scan are acceptance fixtures for this path. They are
not plugin names or public compiler entry points.

## Current state

### Runtime attachment boundary

`jax_hlo_rewrite_runtime.py` is a useful compatibility boundary, but not yet a
plugin:

- `HloRewriteRuntime` round-trips serialized HLO through JAX's compiler-IR type
  and parses rewritten text (`jax_hlo_rewrite_runtime.py:43-58`).
- `audit_hlo_rewrite_runtime` checks the compiler-IR round trip, private text
  parser, and `jax.extend.xla` transformation API before device allocation
  (`jax_hlo_rewrite_runtime.py:105-138`).
- `require_hlo_rewrite_runtime` fails closed if any capability is absent
  (`jax_hlo_rewrite_runtime.py:141-156`).

It does not own:

- registration lifetime or cleanup;
- concurrent callback safety;
- module identity;
- structural no-match versus compilation failure;
- handler-library lifetime;
- candidate enumeration and selection;
- transform/coverage records; or
- an idempotence rule for callbacks that see the same module more than once.

Every harness currently defines its own callback and directly calls
`register_hlo_module_transformation` and
`clear_hlo_module_transformation`. The duplication is visible in the pair-Map
smoke (`xla_pair_map_custom_call_smoke.py:1517-1547`), the Grug backward smoke
(`xla_grug_backward_multi_output_gpu_custom_call_smoke.py:330-360`), the
streaming reverse smoke (`xla_streaming_attention_backward_gpu_custom_call.py:224-262`),
and the combined Grug harness (`xla_grug_routed_combined_gpu_custom_call.py:793-1160`).

Several callbacks also compile and register handler libraries inside the XLA
callback. That is acceptable as a prototype mechanism, but it needs a
session-owned cache and lifetime contract before becoming a compiler API.

### HLO import and generic recovery

`xla_hlo_recovery.py` supplies the strongest generic structural base:

- `HloModuleGraph`, `HloComputation`, and `HloInstruction` retain opcode,
  operands, shapes, raw attributes, and source order
  (`xla_hlo_recovery.py:55-104`).
- `InlinedHloGraph` exposes ordinary fusion bodies while retaining source
  computation and source instruction provenance (`xla_hlo_recovery.py:106-184`).
- `RecoveredEntryRegionBoundary` records internal instructions, typed entry
  inputs and outputs, external users, sharding, and effects
  (`xla_hlo_recovery.py:337-353`).
- the parser and inliner are model-name independent
  (`xla_hlo_recovery.py:751-884`);
- pair-Map and multi-output recovery are generic Contract/Map analyses
  (`xla_hlo_recovery.py:356-445`, `xla_hlo_recovery.py:886-979`).

`xla_scalar_map_import.py:26-148` imports a pointwise HLO slice into a cast-aware
scalar AST. `xla_relation_program_recovery.py:667-704` similarly recovers
RelationPlan, SegmentedContract, scalar Map, Fold, weight Contract, and explicit
external-collective structure without model names.

These are valuable inputs to the new importer, not yet the importer itself.
`xla_hlo_recovery.py` describes itself as an inspection path and its parsed
graph cannot serialize itself. A lossless target-1 import must retain the
original serialized proto and untouched HLO text alongside its typed structural
view. Unsupported opcodes and unknown attributes must survive a no-op transform
byte-for-byte or through a verified semantic round trip.

### Region formation, rewriting, and auditing

The current `xla_*` family repeatedly implements a sound local pattern:

```text
recover candidate
  -> prove a typed boundary
  -> construct a generated ABI
  -> replace text with typed FFI
  -> parse round trip
  -> audit liveness/layout/collective preservation
```

The local audits are often strong. Examples include:

- axis-Fold recovery and replacement audits validate scalar semantics,
  layouts, effects, and no extra boundary operations
  (`xla_axis_fold_ffi.py:109-370`);
- demand-sliced Contracts report covered live Contract count and FLOPs, then
  separate physical calls from attachment opportunities
  (`xla_hlo_recovery.py:284-334`, `xla_demand_sliced_contract.py:213-346`);
- low-rank Contract/Map training proves disjoint boundaries, exact target
  multiplicity, removed dot instructions/FLOPs, live outputs, and unchanged
  collectives (`xla_low_rank_gated_product_ffi.py:187-431`);
- routed training verifies exact custom calls, auxiliary liveness, no added
  copies/transposes, and placement collectives
  (`xla_routed_training_ffi.py:295-379`);
- streaming attention training verifies the early-forward/later-reverse state
  link without spanning intervening JAX dataflow
  (`xla_streaming_attention_training_regions.py:146-301`).

The weakness is composition. Each audit has its own types and metrics. There is
no common statement of:

- which original operations are owned by a region;
- which live edges cross its boundary;
- which requirements make the region legal;
- which original work was removed;
- which external work intentionally remains;
- how two candidate regions overlap; or
- what percentage of the requested module is covered.

The combined Grug harness solves this by knowing every component. Its
`RoutedTrainingCompositionMode` selects a sequence of named composition levels
and hard-codes expected call counts (`xla_grug_routed_combined_gpu_custom_call.py:215-258`).
Its callback explicitly invokes routed, weighted-relation, normalized-exp,
low-rank gated-product, attention, and axis-Fold planners and generators
(`xla_grug_routed_combined_gpu_custom_call.py:793-1135`). This is good
experimental evidence, but it is the wrong abstraction boundary for the public
compiler.

### Final-HLO and handler evidence

`command_buffer_capture.py` contains reusable final-HLO and measurement
machinery:

- `derive_capture_site_manifest` counts exact target occurrences in final
  optimized HLO and binds targets to instrumented handlers
  (`command_buffer_capture.py:71-116`);
- `stabilize_counterbalanced_variants` requires bounded capture stabilization
  before timing (`command_buffer_capture.py:357-413`);
- `measure_counterbalanced_variants` records counterbalanced raw samples and
  callback checkpoints (`command_buffer_capture.py:416-468`);
- `assess_command_buffer_capture` rejects missing, unattributed, fallback, or
  recaptured handlers (`command_buffer_capture.py:554-626`).

The combined Grug harness is the only current `xla_*` harness that preserves
`compiled.as_text()` as final optimized HLO and derives a final-HLO capture
manifest (`xla_grug_routed_combined_gpu_custom_call.py:1168-1176`). The other
`xla_*` harnesses preserve only original and transformed PRE_SCHEDULER HLO.

Even the combined harness uses the reusable capture code only to count selected
sites. It does not use the bounded stabilization and timed callback-assessment
protocol. Its correctness, output hashes, and two-way timings are implemented
locally (`xla_grug_routed_combined_gpu_custom_call.py:1179-1217`,
`xla_grug_routed_combined_gpu_custom_call.py:1658-1664`).

Generated handler evidence is also fragmented. Most generators expose a
semantic digest and source digest, but target, handler symbol, API version,
buffer ABI, dependency lineage, and command-buffer eligibility do not share one
manifest type. The combined harness reconstructs that information into a large
workload-specific JSON result (`xla_grug_routed_combined_gpu_custom_call.py:1840-2134`).

### Numerical, determinism, layout, and timing evidence

The generic parts already exist:

- `DenseBufferContract` verifies logical shape, dtype, strides, and
  minor-to-major layout (`benchmark_boundary.py:16-49`);
- `NumericalAcceptanceContract` ties error thresholds to an explicit numerical
  policy (`benchmark_boundary.py:63-77`);
- `benchmark_repeatability_report` records output hashes, semantic-reference
  errors, and every pairwise repeat drift (`benchmark_boundary.py:247-287`);
- `verify_benchmark_repeatability` enforces bitwise or bounded-drift policy
  before timing (`benchmark_boundary.py:289-359`);
- `benchmark_metadata.py` records command, selected environment, GPU topology,
  clocks, power, and toolchain.

The attention component harness uses these checks for generated and expert
oracles before a two-way benchmark
(`jax_streaming_attention_backward_ffi_gpu.py:221-268`,
`jax_streaming_attention_backward_ffi_gpu.py:369-477`). The module-transform
harnesses do not use one common record, and no harness measures all four
required variants: natural JAX, matched expert oracle, generated
source-ordered, and generated fast-math.

## Inventory and migration decision

The table classifies every current `xla_*` source module at `f3ce4aac3c`.
“Extract” means move generic logic only after its inputs use the new typed
module and shared algebra. “Bridge” means leave the current module in
`tile_lifetime` as experimental evidence until replaced; do not make it a
`shuttle` compatibility alias.

| Current module | Structural role | Target-1 decision |
| --- | --- | --- |
| `jax_hlo_rewrite_runtime.py` | JAX/JAXLIB capability and parser boundary | Extract runtime capability audit and registration adapter |
| `xla_hlo_recovery.py` | Generic HLO parser, graph, fusion inliner, Contract/Map regions | Extract concepts into lossless typed import; keep text bridge until replaced |
| `xla_scalar_map_import.py` | Generic cast-aware scalar AST import | Move after it consumes typed imported values |
| `xla_axis_fold_ffi.py` | Generic Fold/final-Map discovery, replacement, audit | Extract region provider and audit obligations |
| `xla_axis_fold_pipeline_ffi.py` | Exact whole-entry Fold pipeline bridge | Leave as bridge; whole-entry assumption is not public API |
| `xla_demand_sliced_contract.py` | Generic Contract demand partition and FFI plan | Extract candidate provider and coverage metrics |
| `xla_partitioned_contract_map.py` | Contract partition plus scalar finalization | Extract algebra/physical candidate logic |
| `xla_partitioned_contract_fold.py` | Contract partition plus auxiliary Fold | Extract algebra/physical candidate logic |
| `xla_fold_consumer_preparation.py` | Generic Fold-to-Contract scalar preparation | Extract rewrite analysis |
| `xla_low_rank_gated_product.py` | Generic diagonal/low-rank Contract structure recovery | Extract structural provider; rename away from model-shaped terminology if needed |
| `xla_low_rank_gated_product_ffi.py` | Typed Contract/Map forward/reverse boundaries and audit | First target-1 bridge; extract normalized boundary and coverage logic |
| `xla_normalized_exp_contract_forward.py` | Generic Contract plus normalized-exp Fold forward | Extract provider after normalized-exp is generic Fold state |
| `xla_normalized_exp_contract_reverse.py` | Generic normalized-exp reverse Contract/Fold recovery | Extract provider after normalized-exp is generic Fold state |
| `xla_rank_two_contract_ffi.py` | Generic rank-two BF16 Contract generator/rewrite | Extract generator adapter; replace backend-specific type names later |
| `xla_contract_relation_fold_ffi.py` | Bounded Contract plus nested relation Folds | Extract physical candidate after generic Relation/Fold input |
| `xla_weighted_relation_reverse_ffi.py` | Generic relation-edge Contract/Fold reverse | Extract provider/generator after generic Relation input |
| `xla_relation_edge_reverse_ffi.py` | Generic relation-edge Map/Fold handler | Extract generator adapter |
| `xla_relation_program_recovery.py` | Generic RelationPlan/SegmentedContract/Map/Fold recovery | Extract structural analyses incrementally; split the 3,042-line mixed module |
| `xla_routed_forward_ffi.py` | Generated routed Contract/Map/Contract/Fold handler | Leave current file as bridge; extract generic program/generator pieces |
| `xla_routed_input_adjoint_ffi.py` | Generated routed input-adjoint handler | Leave current file as bridge; extract generic SegmentedContract pieces |
| `xla_routed_weight_gradient_ffi.py` | Group-batched weight Contract handler | Extract as generic segmented/group-batched Contract generator |
| `xla_segmented_input_adjoint_ffi.py` | Fixed-capacity segmented reverse Contracts and Map | Extract generic physical candidate and buffer-elision legality |
| `xla_shared_contract_multimap.py` | Shared Contract/multi-Map reference and rewrite audit | Extract audit obligations; leave text rewrite as bridge |
| `xla_shared_contract_multimap_ffi.py` | Generated shared Contract/multi-Map handler | Extract generic generator adapter |
| `xla_source_indexed_fold_ffi.py` | Deterministic source-indexed scalar Fold | Extract generic Fold generator/provider |
| `xla_routed_shared_map_training_ffi.py` | Named composition of routed training subregions | Leave in `tile_lifetime`; replace with generic conflict/coverage selection |
| `xla_routed_training_ffi.py` | Named composition of routed plus attention plus axis Fold | Leave in `tile_lifetime`; do not create a Shuttle routed API |
| `xla_streaming_attention_backward_ffi.py` | Generic-dataflow attention-shaped reverse bridge | Extract Contract/Fold/DomainRestriction recovery; leave attention-named text bridge |
| `xla_streaming_attention_training_regions.py` | Attention-shaped early-forward/later-reverse state split | Leave as bridge until generic saved-state/recompute region composition exists |

### Harness inventory

| Harness | What it proves | What remains local/workload-specific |
| --- | --- | --- |
| `xla_pre_scheduler_probe.py` | Natural JAX reaches PRE_SCHEDULER with no semantic custom call | No rewrite or execution ownership |
| `xla_pair_map_custom_call_smoke.py` | CPU structural pair-Map callback and execution | Own parser/emitter/registration; synthetic whole program |
| `xla_grug_pair_map_custom_call_smoke.py` | Natural Grug region replacement on CPU | Chooses Grug forward region and legacy CPU handler |
| `xla_grug_backward_multi_output_custom_call_smoke.py` | Natural Grug multi-output reverse replacement on CPU | Grug region selection and custom record |
| `xla_grug_backward_multi_output_gpu_custom_call_smoke.py` | GPU typed-FFI Contract/Map reverse and counterbalanced JAX comparison | Two-way timing, ad hoc hashes, no final HLO |
| `xla_partitioned_contract_map_gpu_custom_call.py` | Generated partitioned Contract against ordered CPU reference | Frozen HLO input rather than shared plugin |
| `xla_grug_routed_forward_gpu_custom_call.py` | Natural Grug routed forward execution | One named component, local registration/timing |
| `xla_grug_routed_input_adjoint_gpu_custom_call.py` | Natural Grug routed input adjoint | One named component, local registration/timing |
| `xla_grug_routed_weight_gradient_gpu_custom_call.py` | Natural Grug routed weight Contracts | One named component, local registration/timing |
| `xla_streaming_attention_backward_gpu_custom_call.py` | Natural JAX attention VJP region replacement and layout-native outputs | Attention-specific entrypoint; no shared final-HLO/acceptance record |
| `xla_grug_routed_combined_gpu_custom_call.py` | Many generated regions execute in one natural Grug train step | 2,198-line named composition, two-way timing, bespoke evidence schema |

The component `jax_generated_*` and `jax_streaming_attention_backward_ffi_gpu.py`
harnesses are not module transforms, but their numerical, layout, oracle, and
repeatability checks are better factored than most `xla_*` harnesses. Reuse
those checks in the shared acceptance layer.

## Proposed workload-free `shuttle` API

### 1. Lossless typed module

```python
@dataclass(frozen=True)
class ModuleId:
    serialized_proto_sha256: str
    pipeline_stage: PipelineStage
    platform: str

@dataclass(frozen=True)
class ComputationId:
    module: ModuleId
    name: str

@dataclass(frozen=True)
class ValueId:
    computation: ComputationId
    instruction: str

@dataclass(frozen=True)
class RegionId:
    module: ModuleId
    computation: ComputationId
    boundary_sha256: str
    semantic_sha256: str

@dataclass(frozen=True)
class ImportedHloModule:
    id: ModuleId
    serialized_proto: bytes
    original_text: str
    graph: TypedHloModule
    unknown_operations: tuple[ValueId, ...]

@dataclass(frozen=True)
class ImportedAlgebra:
    module: ModuleId
    program: TensorProgram
    source_values_by_operation: tuple[tuple[str, tuple[ValueId, ...]], ...]
    opaque_values: tuple[ValueId, ...]
    external_effects: tuple[ExternalEffect, ...]
```

The serialized callback proto is the source identity. Module name is diagnostic
only and must never select a compiler path. `RegionId` includes both exact
boundary identity and generic semantic identity: two repeated equal regions in
one module have distinct boundary hashes but may share a semantic hash and
physical handler.

The importer must preserve:

- every computation and instruction in source order;
- full shape and layout;
- sharding, alias, side-effect, control, and collective attributes;
- unknown operations and attributes;
- exact entry inputs and outputs; and
- an immutable reference to original serialized bytes.

A typed view may normalize known opcodes for analysis, but cannot drop unknown
data. A no-op compilation returns `None` from the JAX callback or the original
serialized bytes; it never reserializes a partial graph.

`ImportedAlgebra` is a lossless structural view, not a claim that every HLO
instruction became a Shuttle primitive. Generic Map/Contract/Fold/Scan/Relation
operations carry exact source mappings. Unsupported operations remain opaque
typed values and constrain region boundaries.

### 2. Generic region providers

```python
class RegionProvider(Protocol):
    provider_id: str

    def discover(
        self,
        module: ImportedHloModule,
        algebra: ImportedAlgebra,
    ) -> tuple[RegionCandidate, ...]: ...

@dataclass(frozen=True)
class RegionCandidate:
    id: RegionId
    program: TensorProgram
    boundary: RegionBoundary
    owned_values: frozenset[ValueId]
    preserved_values: frozenset[ValueId]
    external_effects: tuple[ExternalEffect, ...]
    obligations: tuple[LegalityObligation, ...]
    coverage: CoverageClaim
```

Providers are registered by generic structure: Contract/Map chain, axis Fold,
normalized-exp Fold, Relation/SegmentedContract, Scan, or another algebraic
family. They do not receive workload labels. They may decline a region with a
structured reason. Discovery is separate from candidate selection.

The importer first creates generic algebra. Providers may then propose
algebraic rewrites, task decompositions, dependency/event plans, and physical
candidates. The public API retains these stages even when the first vertical
slice uses a direct Contract/Map path:

```python
@dataclass(frozen=True)
class CandidateTrace:
    algebra_rewrite: AlgebraRewriteRecord
    task_graph: TaskGraphRecord
    dependency_graph: DependencyRecord
    event_plan: EventTensorPlan | None
    physical_plan: PhysicalPlanRecord
```

This prevents the plugin from becoming one large region recognizer that jumps
from HLO directly to a named handler.

### 3. Legality and coverage ledger

```python
class ObligationKind(StrEnum):
    CONVEX_REGION = "convex_region"
    COMPLETE_BOUNDARY = "complete_boundary"
    NO_EFFECT_CROSSING = "no_effect_crossing"
    LAYOUT_COMPATIBLE = "layout_compatible"
    NUMERICAL_POLICY = "numerical_policy"
    COLLECTIVE_PRESERVED = "collective_preserved"
    ALIASING_SAFE = "aliasing_safe"
    TOPOLOGICALLY_INSERTABLE = "topologically_insertable"

@dataclass(frozen=True)
class LegalityEvidence:
    obligation: ObligationKind
    satisfied: bool
    source_values: tuple[ValueId, ...]
    detail: str

@dataclass(frozen=True)
class CoverageClaim:
    owned_values: frozenset[ValueId]
    work_estimates: tuple[CoverageMetric, ...]
    boundary_inputs: tuple[ValueId, ...]
    boundary_outputs: tuple[ValueId, ...]
    external_users: tuple[tuple[ValueId, tuple[ValueId, ...]], ...]

@dataclass(frozen=True)
class CoverageLedger:
    module: ModuleId
    live_values: frozenset[ValueId]
    selected_regions: tuple[RegionId, ...]
    claims: tuple[CoverageClaim, ...]
    owner_by_value: tuple[tuple[ValueId, RegionId], ...]
    intentionally_external: tuple[ValueId, ...]
    uncovered_requested_values: tuple[ValueId, ...]
```

Selection fails before code generation when owned regions overlap, an external
user is not represented on the boundary, an effect or control edge crosses a
region, or a requested value is silently uncovered. Coverage maps each original
value to exactly one selected region or to an explicit external/unowned reason.
Inserted-call counts alone do not establish coverage.

After text/proto round trip, `TransformAudit` must prove:

- every selected internal instruction is dead or is a declared surviving
  boundary value;
- every boundary output has the same external users;
- every inserted target has its declared ABI and multiplicity;
- no unselected live instruction disappeared;
- preserved collectives/effects remain in source order;
- copy/transpose/layout adapters match the selected physical plan; and
- the transformed module parses and verifies.

Current rewriter audits can implement adapters to these obligations during
migration. The common ledger should not erase their richer local evidence.

### 4. Generic candidate search

```python
@dataclass(frozen=True)
class CompilationOptions:
    numerical_policy: NumericalPolicy
    target: HardwareTarget
    search_policy: SearchPolicy

class PhysicalCandidateProvider(Protocol):
    def enumerate(
        self,
        region: RegionCandidate,
        options: CompilationOptions,
    ) -> tuple[PhysicalCandidate, ...]: ...

class CandidateSelector(Protocol):
    def select(
        self,
        candidates: tuple[PhysicalCandidate, ...],
        conflicts: RegionConflictGraph,
    ) -> SelectionResult: ...
```

Candidate IDs are canonical digests of generic program, numerical policy,
hardware target, task/dependency/event plan, layout, and physical parameters.
Finite hard-coded candidate sets are acceptable. Workload name, module name,
and frontend function name are forbidden candidate keys.

The selector first maximizes a declared coverage/benefit objective subject to
legality and overlap constraints, then applies heuristic or measured cost.
Source-ordered and fast-math candidates are separate compilations because they
carry different numerical contracts; one cannot silently fall back to the
other.

### 5. Generated handler manifest

```python
@dataclass(frozen=True)
class HandlerBuffer:
    role: str
    dtype: str
    shape: tuple[int, ...]
    minor_to_major: tuple[int, ...]
    alias_of_input: int | None

@dataclass(frozen=True)
class GeneratedHandlerManifest:
    candidate_id: str
    region_ids: tuple[RegionId, ...]
    target: str
    handler_symbol: str
    platform: str
    ffi_api_version: int
    inputs: tuple[HandlerBuffer, ...]
    outputs: tuple[HandlerBuffer, ...]
    semantic_sha256: str
    source_sha256: str
    binary_sha256: str
    external_dependencies: tuple[ExternalDependency, ...]
    command_buffer_eligibility: CommandBufferEligibility
    expected_pre_scheduler_occurrences: int
```

Targets are derived from the candidate digest and generic emitter family, not
from a workload. A session-owned `HandlerRepository` compiles once, registers
once, keeps the dynamic library alive through JAX compilation/execution, and
returns the same manifest for repeated equal candidates.

After JAX compilation, final optimized HLO augments the manifest with observed
target occurrences and final-HLO digest. The existing `CaptureSiteManifest`
remains a runtime capture view derived from this handler manifest; it is not the
compiler's only handler record.

### 6. Transform registration

```python
@contextmanager
def register_module_transform(
    compiler: ShuttleCompiler,
    *,
    name: str,
    platforms: tuple[str, ...],
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
) -> Iterator[TransformSession]: ...

class ShuttleCompiler:
    def compile_module(
        self,
        serialized_module: bytes,
        context: TransformContext,
    ) -> ModuleTransformResult | NoTransform: ...
```

The callback algorithm is:

1. import the serialized proto losslessly;
2. recover generic algebra without module-name selection;
3. discover generic regions;
4. enumerate algebra/task/dependency/event/physical candidates;
5. select nonoverlapping legal candidates;
6. generate or load handler manifests;
7. rewrite only selected boundaries;
8. parse/verify/audit the transformed module;
9. record the complete `ModuleCompilationRecord`; and
10. return the serialized transformed module.

Finding no candidate is a normal result. A provider exception, failed
obligation, rewrite verification error, or incomplete requested coverage is an
error. A module that already contains this session's exact generated targets is
either returned unchanged with an idempotence record or rejected; it is never
rewritten a second time.

The context manager owns pass cleanup and handler-library lifetime. The session
is concurrency-safe and stores records keyed by `ModuleId`. Tests must not rely
on one callback invocation or a particular module name.

### 7. Module compilation record

```python
@dataclass(frozen=True)
class ModuleCompilationRecord:
    source_module: ModuleId
    source_hlo_sha256: str
    imported_unknown_operation_count: int
    algebra_sha256: str
    discovered_regions: tuple[RegionDiscoveryRecord, ...]
    selected_candidates: tuple[CandidateTrace, ...]
    legality: tuple[LegalityEvidence, ...]
    coverage: CoverageLedger
    handlers: tuple[GeneratedHandlerManifest, ...]
    transformed_proto_sha256: str
    transformed_pre_scheduler_hlo_sha256: str
    transform_audit: TransformAudit
```

This record is the compiler output. Benchmark evidence is a separate layer that
references it.

## Shared four-way acceptance harness

Every Dense, MoE, attention, and Scan target uses the same harness schema. A
workload bridge supplies only natural inputs, a semantic reference, and an
optional matched expert oracle.

### Variants

1. `jax`: unmodified natural JAX program, including JAX-owned AD.
2. `oracle`: matched expert implementation for the same boundary. This can be
   absent for a compiler correctness smoke but is required for a performance
   acceptance claim.
3. `shuttle_source_ordered`: the plugin with the strict source-ordered numerical
   contract.
4. `shuttle_fast_math`: the same plugin and region providers with explicitly
   permitted reassociation/fast-math.

Torch may implement the optional expert oracle. It is not imported by the
default compiler or the natural JAX/source-ordered/fast-math variants.

### Record

```python
@dataclass(frozen=True)
class AcceptanceVariantRecord:
    role: AcceptanceVariantRole
    executable_sha256: str | None
    final_hlo_sha256: str | None
    compilation: ModuleCompilationRecord | None
    handler_manifest: tuple[GeneratedHandlerManifest, ...]
    final_hlo_capture_sites: CaptureSiteManifest | None
    buffers: tuple[DenseBufferContract, ...]
    numerical_contract: NumericalAcceptanceContract
    repeatability: BenchmarkRepeatabilityReport
    warmup_executions: int
    samples_ms: tuple[float, ...]
    median_ms: float
    minimum_ms: float

@dataclass(frozen=True)
class FourWayAcceptanceRecord:
    schema_version: int
    source_artifact_sha256: str
    fixture_sha256: str
    hardware: HardwareSnapshot
    toolchain: ToolchainSnapshot
    timing_boundary: TimingBoundary
    variants: tuple[AcceptanceVariantRecord, ...]
    execution_orders: tuple[tuple[AcceptanceVariantRole, ...], ...]
    pairwise_ratios: tuple[PairwiseLatencyRatio, ...]
    external_collectives: tuple[ExternalBoundaryRecord, ...]
```

### Protocol

- Freeze one input fixture and materialize equivalent per-runtime views.
- Verify logical shape, dtype, strides, and layout before any timing.
- Run semantic-reference and repeatability checks for every variant before
  timing. Source-ordered must be bitwise repeatable when its contract requires
  it. Fast-math and expert oracles may use predeclared bounded drift only.
- Preserve pre-timing failures before raising.
- Compile all JAX variants before warmup. Compilation time is recorded but
  excluded from latency.
- Capture final optimized HLO for `jax`, `shuttle_source_ordered`, and
  `shuttle_fast_math`. Derive exact generated target sites for both Shuttle
  variants. An external oracle records binary/source/revision lineage instead.
- Warm all four variants with identical iteration bursts and synchronization.
  When command-buffer candidates are present, use the existing bounded
  stabilization protocol before timing.
- Use all balanced permutations or a predeclared balanced Latin-square schedule
  when 24 permutations are too expensive. Record the exact order. Do not use a
  fixed generated-first order.
- Time the same boundary, synchronization, save/recompute policy, collective
  scope, and output materialization.
- Record raw samples and hashes. Summary medians are derived, not supplied.
- Report at least generated/JAX, generated/oracle, source-ordered/fast-math, and
  oracle/JAX ratios. Only the matched oracle ratio gates the 1.20 target.

This schema separates four different questions that current two-way harnesses
conflate:

- Did Shuttle preserve the natural JAX program?
- Is the expert oracle itself admissible and semantically matched?
- What is the cost of retaining source order?
- How close is the best permitted Shuttle candidate to the expert?

## Smallest implementation slice

Wait for the migration agent's `lib/shuttle` base, then implement one vertical
slice. Do not copy modules or add `tile_lifetime` compatibility re-exports.

### Slice: generic two-Contract Map forward and reverse

Use a natural JAX program with two rank-two Contracts surrounding a scalar Map,
plus JAX-produced reverse Contracts. Reuse the structural ideas currently in:

- `xla_hlo_recovery.py` for typed Contract/Map discovery;
- `xla_scalar_map_import.py` for the scalar AST;
- `xla_low_rank_gated_product.py` for forward/reverse structural pairing;
- `contract_map_chain.py` and `cuda_contract_map_chain_codegen.py` for the
  generic program and generated body; and
- `xla_low_rank_gated_product_ffi.py` only as the old text-bridge and audit
  oracle.

The slice should add, in `lib/shuttle`:

1. `hlo_module.py`: lossless module wrapper and typed IDs.
2. `module_transform.py`: provider protocols, session, registration lifetime,
   no-match/error behavior, and compilation record.
3. `coverage.py`: generic legality evidence, conflict graph, coverage ledger,
   and transform audit.
4. `handler_manifest.py`: common generated ABI/source/binary manifest and
   repository.
5. `providers/contract_map.py`: one model-free region provider and finite
   physical candidate enumeration.
6. `acceptance.py`: four-way record and shared timing/correctness protocol.

The provider must work for a scalar-Map mutation such as tanh instead of SiLU
without a new provider or handwritten handler. JAX remains the owner of AD.

### Test matrix

#### Import and identity, CPU

- no-op callback returns the original serialized module and retains an unknown
  operation/attribute;
- equal serialized modules have equal `ModuleId`; an operand, shape, layout, or
  attribute mutation changes it;
- repeated semantically equal regions have distinct `RegionId` boundaries but
  can share one semantic/handler digest;
- module and function names do not affect provider eligibility;
- module-name-only selection is rejected by an API test.

#### Legality and coverage, CPU

- exact Contract/Map boundary produces complete input/output/external-user
  evidence;
- overlapping regions fail before generation;
- an omitted external user, effect, control edge, or alias obligation fails;
- preserved all-reduce remains outside the region and in the same order;
- selected internal instructions are removed and unrelated live instructions
  survive;
- partial requested coverage is reported, never silently accepted.

#### Search and handler manifest, CPU

- candidate order and selection are deterministic under provider registration
  reordering;
- source-ordered and fast-math candidates have distinct IDs and numerical
  contracts;
- mutation changes semantic/source digests while retaining the emitter family;
- identical physical families share a handler manifest;
- ABI shape/layout, target multiplicity, handler symbol, API version, source
  digest, and binary digest are complete and fail closed on mismatch.

#### JAX transform, CPU smoke

- context manager registers and clears exactly one pass even on exception;
- unrelated JAX compilations return `NoTransform` structurally, not by module
  name;
- repeated callback invocation is idempotent and library lifetime extends
  through execution;
- natural JAX forward and VJP execute through the generated CPU test handler;
- a tanh/SiLU mutation follows the same provider and registration path.

#### GPU acceptance, after the CPU slice

- compile four variants through one workload bridge;
- validate exact buffer/layout boundaries and pre-timing repeatability;
- preserve original, transformed PRE_SCHEDULER, and final optimized HLO;
- audit coverage and exact final-HLO target occurrences;
- run identical warmups, command-buffer stabilization when applicable, and
  balanced four-way timing;
- record raw samples, hashes, numerical errors, determinism/drift, handler
  manifests, environment, toolchain, and ratios.

#### Cross-workload extension

The API gate is complete only after Dense, MoE, attention, and Scan each use the
same importer, session, compilation record, and acceptance record. New generic
providers are allowed. New workload-specific module-transform APIs are not.

## Reusable code versus bridges

### Reuse or move after migration base lands

- runtime capability audit from `jax_hlo_rewrite_runtime.py`;
- typed graph and fusion-inlining concepts from `xla_hlo_recovery.py`;
- scalar AST import from `xla_scalar_map_import.py`;
- generic region-boundary and numerical-contract concepts from the structural
  `xla_*` providers;
- semantic/source digest conventions from the generic CUDA emitters;
- `DenseBufferContract`, numerical acceptance, and repeatability from
  `benchmark_boundary.py`;
- final-HLO capture-site derivation and bounded capture stabilization from
  `command_buffer_capture.py`;
- reproducibility metadata from `benchmark_metadata.py`; and
- frozen HLO/artifact fixtures as migration oracles.

### Leave in `tile_lifetime`

- exact text/regex replacement bridges until the typed module rewriter replaces
  them;
- `xla_routed_training_ffi.py` and
  `xla_routed_shared_map_training_ffi.py` named composition;
- attention-named training-region bridge;
- every `xla_*` benchmark script and its natural Grug/debug setup;
- Torch expert-oracle adapters;
- frozen artifacts and snapshot tests; and
- workload-specific JSON evidence schemas.

No compatibility aliases should make these old entry points appear to be the
new compiler.

## Negative and rejected leads

### Wrap the combined Grug callback

Rejected. It hard-codes composition modes, selected target families, counts,
and Grug-specific region expectations. A wrapper would preserve one giant
workload recognizer rather than create a generic plugin.

### Treat target occurrence as coverage

Rejected. A generated target can be present while old arithmetic remains live,
an external user is lost, or an unselected region changes. Final-HLO target
counts are handler/capture evidence, not a semantic ownership ledger.

### Promote `xla_hlo_recovery.py` unchanged as a lossless importer

Rejected. It is a useful typed inspection view, but it cannot serialize itself
and recognizes only a bounded HLO text grammar. The new import must retain the
original proto and unknown data.

### Add per-workload plugin entry points

Rejected. `compile_dense_module`, `compile_moe_module`, and
`compile_attention_module` would reproduce the current split. Workloads belong
in acceptance fixtures and optional structural provider tests, not the public
compiler API.

### Let Shuttle own AD in the plugin

Rejected for target 1. Existing Grug and attention HLO already demonstrate
JAX-owned differentiation. Shuttle imports and optimizes the primal and reverse
regions JAX supplies. Compiler-owned AD can remain a research comparison, not a
default dependency.

### Require Torch in the harness package

Rejected. Torch is an optional external-oracle adapter. The default compiler,
JAX variant, semantic reference, and generated variants remain Torch-free.

## Recommended implementation sequence

1. Land the migration agent's `lib/shuttle` algebra/import base.
2. Add typed module/region IDs and a no-op registration session.
3. Add the generic coverage ledger and use it to adapt one existing
   Contract/Map audit.
4. Add one Contract/Map provider and deterministic finite candidate selection.
5. Normalize its generated handler into the common manifest and repository.
6. Execute the natural JAX forward/VJP CPU smoke through the session.
7. Add the shared four-way record and port one GPU component benchmark.
8. Use the same session and record for Dense, MoE, attention, and Scan by adding
   generic providers, not public workload APIs.

Stop after step 6 for an architecture review. The CPU slice is enough to reveal
whether identity, losslessness, conflicts, callback lifetime, and coverage are
correct before GPU integration makes failures expensive.

## Questions resolved and open

### Resolved

- The public unit is a module-transform session, not a workload compiler.
- The exact original dependency graph and boundary are the coverage source of
  truth; custom calls and final-HLO sites are physical evidence.
- JAX owns AD by default.
- Source-ordered and fast-math are separate explicit candidate families.
- EventTensor is a derived optional stage after exact task dependencies, not an
  HLO semantic primitive.
- Torch is optional oracle-only infrastructure.

### Open

- Whether target 1 should patch HLO text initially or wait for a typed proto/C++
  mutation API. A text bridge is acceptable only behind lossless source
  retention and a complete transform audit.
- Whether handler compilation is allowed synchronously inside the JAX callback
  or must be split into discovery and a second compilation. The first slice can
  cache synchronously, but the session API should not promise that behavior.
- Which coverage objective should select among disjoint legal regions before a
  calibrated cost model exists. A deterministic priority of removed FLOPs,
  removed materialization bytes, and fewer calls is a reasonable prototype.
- Whether final optimized HLO is available through a stable JAX API for all
  target backends. The harness should fail closed when it cannot capture it.

## Evidence map

### Claim: generic structural recovery is sufficient for the first slice

- Support: `xla_hlo_recovery.py`, `xla_scalar_map_import.py`,
  `xla_low_rank_gated_product.py`, and `xla_low_rank_gated_product_ffi.py` recover,
  generate, replace, and audit Contract/Map forward/reverse regions without a
  model-name key.
- Contradiction: the current path still uses HLO text/regex bridges and no
  common lossless imported-module object.
- Confidence: high for the structural family; medium for the new plugin
  lifecycle.
- Action: use this family as the first vertical slice, not as the final importer.

### Claim: current acceptance evidence is reusable but fragmented

- Support: `benchmark_boundary.py`, `command_buffer_capture.py`, and
  `benchmark_metadata.py` cover layout, numerical policy, repeatability,
  final-HLO capture sites, stabilization, timing, and environment metadata.
- Contradiction: no current module-transform harness composes all of them, and
  no four-way record exists.
- Confidence: high.
- Action: define one record and port code rather than adding another benchmark
  JSON schema.

### Claim: the combined Grug harness should not become the plugin

- Support: its named `RoutedTrainingCompositionMode`, explicit generator calls,
  target lists, and expected counts span 2,198 lines.
- Contradiction: it is the best current end-to-end execution and final-HLO
  evidence, so it remains a valuable acceptance oracle during migration.
- Confidence: high.
- Action: compare the shared plugin against its frozen artifacts; do not wrap it.

## Source ledger

| Source | Type | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- |
| `jax_hlo_rewrite_runtime.py` at `f3ce4aac3c` | Marin code | Runtime capability boundary, missing session lifecycle | High | Direct source audit |
| `xla_hlo_recovery.py` at `f3ce4aac3c` | Marin code | Generic structural graph and region boundaries | High | Inspection view, not lossless serializer |
| `xla_scalar_map_import.py` at `f3ce4aac3c` | Marin code | Generic scalar AST import | High | No workload keys |
| all `xla_*.py` source modules at `f3ce4aac3c` | Marin code | Rewriter/provider inventory | High | 28 modules inspected |
| all `benchmarks/xla_*.py` at `f3ce4aac3c` | Benchmark code | Registration, artifact, and timing inventory | High | 11 harnesses inspected |
| `benchmark_boundary.py` at `f3ce4aac3c` | Marin code | Layout, correctness, determinism contract | High | Behavior tests exist |
| `command_buffer_capture.py` at `f3ce4aac3c` | Marin code | Final-HLO handler sites and stabilized timing | High | Generic implementation and tests |
| `benchmarks/benchmark_metadata.py` at `f3ce4aac3c` | Benchmark code | Reproducibility metadata | High | GPU/toolchain/command capture |
| `xla_grug_routed_combined_gpu_custom_call.py` at `f3ce4aac3c` | Benchmark code | Strong end-to-end evidence and composition anti-pattern | High | Only `xla_*` final-HLO capture |
| `jax_streaming_attention_backward_ffi_gpu.py` at `f3ce4aac3c` | Benchmark code | Pre-timing oracle/repeatability checks | High | Two-way component benchmark |
| `CURRENT_ARCHITECTURE.md` at `f3ce4aac3c` | Design ledger | `lib/shuttle` migration boundary | High | Calls `xla_*` bridges experimental |
| `conceptual_shuttle.md` from coordinating workspace | Design input | Staged algebra/task/dependency/event/physical pipeline | Medium | Not present in audit revision; treated as a north star, not implementation evidence |

## Stop reason

Every current `xla_*` module and harness has been classified. Additional reads
reinforced the same boundary: generic structural recovery and evidence helpers
exist, while workload scripts own composition and lifecycle. The proposed first
slice has a bounded implementation and test matrix and does not depend on GPU
work or on creating a second IR.
