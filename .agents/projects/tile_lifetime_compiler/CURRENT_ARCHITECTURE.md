# Shuttle current architecture and legacy ledger

## Accepted frontend boundary

Production Shuttle compilation starts from an ordinary JAX program. JAX-owned
AD produces StableHLO inside the MLIR pipeline, where Shuttle converts selected
operations into `Map`, `Contract`, `Fold`, `Scan`, `Relation`,
`DomainRestriction`, and `Transport` without reconstructing a workload-named
operation. Frozen StableHLO and Python importers remain reproduction or
experimental boundaries. Named frontend canonicalizers remain reference or
experimental code even when they erase into generic algebra.

The intended production path is:

```text
JAX/JAXPR
  -> exported StableHLO
  -> in-pipeline Shuttle MLIR conversion
  -> generic Shuttle algebra
  -> task, lifetime, schedule, and physical lowering
```

Target 1 exposes this path through one MLIR compiler extension and one JAX
compiler-options API. The extension converts selected StableHLO regions inside
the compilation pipeline, builds generic Shuttle algebra, and lowers chosen
schedules. Per-workload `compile_stablehlo_*` functions are prototype fixtures,
not the target API.

Frozen StableHLO is an accepted test and reproduction boundary. The Python
parser under `shuttle.experimental` and the `xla_*` text rewriters are not the
production compiler path. A hand-built `TensorGraph`, `StatefulScan`, task list,
or callable workload kernel is not an accepted frontend input. PyTorch and
Triton may appear in standalone benchmark or emitter code; neither is part of
the default semantic frontend.

## Current and reference surfaces

| Surface | Status | Boundary |
| --- | --- | --- |
| `pipeline.compile_stablehlo_dense_transformer_region` | Experimental reference | The exact recognizer targets one bounded Llama-shaped region. It cannot satisfy the current generic-dataflow frontend gate. |
| routed and projected routed attention recovery | Experimental prototype | StableHLO selection becomes generic Relation/RelationPlan and streaming Fold structure, but the bounded selectors are not accepted plugin frontends. |
| `stablehlo_scan_recovery` | Experimental structural importer | Structured `stablehlo.while` becomes generic affine `Scan`; it remains an executable specification until the MLIR conversion owns the same transform. |
| StableHLO row-normalization recovery | Experimental prototype | Natural JAX forward/VJP HLO becomes generic Map/Fold/Contract programs before code generation. It remains an acceptance fixture until it runs through the shared module plugin. |
| `pipeline.compile_experimental_whole_pattern_stablehlo_streaming_attention_program` and `stablehlo_streaming_attention_backward` | Experimental reference | Match attention-shaped graphs and rebuild canonical programs. These paths reproduce diagnostics but do not count as an accepted plugin frontend. |
| `reference_pipeline` | Reference only | Retains named RMS, MoE, and attention planning, including the opaque official-FA3 comparison path. It is not re-exported from `tile_lifetime`. |
| `attention.compile_reference_attention_region` and `qkv_rope.compile_reference_qkv_rope_attention_region` | Reference only | Hand-built named `TensorGraph` planners for preserved comparisons and unit tests. |
| `moe.compile_mok_oracle_region` | Oracle only | Builds the opaque MoK comparison skeleton. Import it explicitly from `tile_lifetime.moe`. |
| `*_reference.py`, `*_frontend.py` debug exporters, and `delta_rule_reference.py` | Test/benchmark inputs | Natural JAX examples and independent numerical references. They are not package-root compiler APIs. |
| `xla_*` HLO text/regex rewriters | Experimental attachment bridge | Finds bounded HLO regions and inserts custom calls. It is not the semantic recovery layer or the intended final frontend. |

An AST audit of all 450 package-root exports found six direct imports outside
tests and benchmarks. Four came from the historical RMS example. The other two
API families were generic streaming-attention types used by SM100 smoke tools.
After this cleanup, 418 root exports remain and only `DType`,
`StreamingTileSchedule`, and `build_attention_tensor_program` have direct
non-test/non-benchmark callers. The large remaining root is a prototype
convenience barrel. It must not become the `shuttle` API.

The package root no longer exports `TensorGraph`, direct named dense/MoE/SwiGLU
planners, RegionPlan runtime helpers, named MoE recovery, reference scan
functions, or numerical reference executors covered by these cleanup slices.
Tests and benchmarks import those objects from their owning modules.

## `lib/shuttle` migration

`lib/shuttle` becomes the compiler package. Move code; do not copy it and do not
leave a `tile_lifetime` compatibility re-export facade.

### Phase 1: algebra and verification

The first package slice owns the shared `DType` vocabulary. The supported-subset
Python StableHLO parser is quarantined under `shuttle.experimental` for old
executable specifications. `tile_lifetime` depends on `shuttle` while the
compiler migrates; `shuttle` has no reverse dependency or compatibility
re-export through `tile_lifetime`.

Phase 1 adds the in-pipeline MLIR spine:

- generic Shuttle MLIR operations for Map, Fold, Contract, Scan, Relation,
  DomainRestriction, and Transport;
- StableHLO conversion patterns selected by structure and dataflow;
- source-ordered and fast numerical-policy attributes in compilation identity;
- policy-gated canonicalization and legality/coverage diagnostics; and
- JAX compiler-options plumbing into the pass pipeline.

JAX owns AD. Shuttle consumes the primal and transpose/VJP StableHLO that JAX
places in the compilation pipeline. Unsupported operations remain StableHLO or
make a selected region fail closed; no production pass reparses textual HLO or
reconstructs a named workload.

### Phase 2: dataflow and scheduling

Port the generic parts of these schedule modules after Phase 1:

- `relation.py`
- `relation_transport.py`
- `event_dataflow.py`
- `event_dataflow_adapters.py`
- `event_buffering.py`
- `collective_transport.py`
- `fold_placement.py`
- `tiled_fold_finalize.py`
- `grouped_contract_event_schedule.py`
- `segmented_grouped_contract_event_schedule.py`
- `right_resource_event_schedule.py`
- `streaming_event_schedule.py`

Their inputs must be the Phase 1 MLIR algebra and generic runtime contracts.
Relation, Transport, and EventTensor stay independent of MoE and attention
names. Existing Python objects remain transitional specifications until the
MLIR path produces and consumes equivalent records.

### Phase 3: physical generators and JAX attachment

Move reusable generators only after their semantic inputs use the Phase 1 and
2 APIs:

- `gemm_program.py`
- `partitioned_gemm_program.py`
- `cuda_prepared_contract_codegen.py`
- `cuda_map_fold_codegen.py`
- `cuda_axis_fold_codegen.py`
- `cuda_partitioned_gemm_codegen.py`
- `cuda_event_dataflow_codegen.py`
- `cuda_dynamic_event_dataflow_codegen.py`
- `cuda_normalized_exp_contract_forward_codegen.py`
- `cuda_normalized_exp_contract_reverse_codegen.py`
- `ffi_command_buffer.py`
- `command_buffer_capture.py`
- the `jax_*_ffi.py` registration modules whose interfaces are generic

The first `shuttle` public vertical slice should be:

```text
ordinary JAX primal and JAX-owned VJP
  -> jax.jit(..., compiler_options=shuttle.compiler_options(...))
  -> StableHLO-to-Shuttle MLIR conversion for one Contract/Map region
  -> policy-gated source-ordered and fast lowerings
  -> structured compilation and coverage record
```

The same pass pipeline must later accept row normalization, attention, routed
relations, and scans. Adding a second public workload compiler fails this gate.

### Leave in `tile_lifetime`

Do not migrate these named, reference, oracle, or attachment-bridge modules as
compiler APIs:

- `reference_pipeline.py`, `reference_semantic_recovery.py`, `reference.py`
- `attention.py`, `qkv_rope.py`, `swiglu.py`, and
  `compiler.py::compile_reference_region`
- `moe.py`, `moe_recovery.py`, `moe_reference.py`, `moe_training_reference.py`
- `stateful_scan_reference.py`, `delta_rule_reference.py`, and all other
  `*_reference.py` modules
- `msa_frontend.py`, `routed_attention_frontend.py`, and debug exporters
- `pipeline.py`, `dense_region.py`, `expert_parallel.py`, and other bounded
  per-workload prototype compilers
- current `xla_*` HLO-text/regex attachment bridges
- `benchmarks/` and preserved `benchmarks/artifacts/`

Generic algebra embedded in a mixed module must be split and moved before that
module is retired. The old module remains explicit reference code until its
generic portion has a real `shuttle` caller.

## Remaining cleanup, ranked

1. Add the MLIR dialect/conversion spine and JAX compiler-options integration.
   Prove one Contract/Map forward-and-reverse vertical slice without a Python
   import or text-rewrite production path.
2. Preserve Python importers and bounded whole-workload recovery only under
   explicit experimental/reference namespaces.
3. Route public GDN/KDA convenience compilation through the structured
   StableHLO scan importer and quarantine `delta_rule_update_expression` as a
   recovery-unit fixture.
4. Rename `MSADebugConfig`, `RoutedAttentionDebugConfig`, and the streaming
   backward debug exporter around their actual role as natural JAX examples.
5. Replace the `xla_*` HLO text/regex bridges with the MLIR conversion pipeline.
   Until then, keep them labeled experimental and outside frontend acceptance
   claims.
