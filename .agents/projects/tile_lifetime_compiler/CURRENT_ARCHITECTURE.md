# Shuttle current architecture and legacy ledger

## Accepted frontend boundary

Current Shuttle compilation starts from an ordinary JAX program or a frozen
StableHLO artifact exported from one. StableHLO import reconstructs structural
tensor expressions and index relations. Named frontend canonicalization may
identify RMS normalization, attention, routing, or a scan, but those names must
erase into `Map`, `Contract`, `Fold`, `Scan`, `Relation`,
`DomainRestriction`, and `Transport` before physical candidate selection.

The intended path is:

```text
JAX/JAXPR
  -> exported StableHLO
  -> structural recovery
  -> semantic-erasure validation
  -> generic Shuttle IR
  -> schedule and physical generation
```

Frozen StableHLO is an accepted test and reproduction boundary. A hand-built
`TensorGraph`, `StatefulScan`, task list, or callable workload kernel is not an
accepted frontend input. PyTorch and Triton may appear in standalone benchmark
or emitter code; neither is part of the default semantic frontend.

## Current and reference surfaces

| Surface | Status | Boundary |
| --- | --- | --- |
| `pipeline.compile_stablehlo_dense_transformer_region` | Current prototype | StableHLO named canonicalization erases through `dense_flow` before scheduling; the recognizer still targets one bounded Llama-shaped region. |
| `pipeline.compile_stablehlo_streaming_attention_program` | Current prototype | StableHLO attention erases into `TensorProgram` Contract/Map/Fold/DomainRestriction and returns source provenance plus a validated erasure report. |
| routed and projected routed attention recovery | Current prototype | StableHLO selection becomes generic Relation/RelationPlan and streaming Fold structure. |
| `stablehlo_scan_recovery` | Current structural importer | Structured `stablehlo.while` becomes generic affine `Scan`; named GDN/KDA convenience wrappers remain cleanup work on this checkpoint. |
| StableHLO row-normalization and streaming-attention backward recovery | Current prototype | Natural JAX forward/VJP HLO becomes generic Map/Fold/Contract programs before code generation. |
| `reference_pipeline` | Reference only | Retains named RMS and attention planning, including the opaque official-FA3 comparison path. It is not re-exported from `tile_lifetime`. |
| `attention.compile_reference_attention_region` and `qkv_rope.compile_reference_qkv_rope_attention_region` | Reference only | Hand-built named `TensorGraph` planners for preserved comparisons and unit tests. |
| `moe.compile_mok_oracle_region` | Oracle only | Builds the opaque MoK comparison skeleton. Import it explicitly from `tile_lifetime.moe`. |
| `*_reference.py`, `*_frontend.py` debug exporters, and `delta_rule_reference.py` | Test/benchmark inputs | Natural JAX examples and independent numerical references. They are not package-root compiler APIs. |
| `xla_*` HLO text/regex rewriters | Experimental attachment bridge | Finds bounded HLO regions and inserts custom calls. It is not the semantic recovery layer or the intended final frontend. |

The package root no longer exports the named attention planners, MoK oracle,
debug JAX exporters, or numerical reference functions covered by this first
cleanup slice. Tests and benchmarks import those objects from their explicit
reference modules.

## Remaining cleanup, ranked

1. `tile_lifetime.__init__` still exports `TensorGraph`, `compile_region`, and
   other hand-assembled semantic-IR planners. Move their unit tests to explicit
   reference imports, then remove them from the package root.
2. `pipeline.compile_stablehlo_expert_parallel_region` still recovers named MoE
   operations before planning. Connect the natural JAX module to generic
   Relation/SegmentedContract/Map/Fold recovery and carry an erasure report.
3. Split `semantic_recovery.py`: dense and streaming-attention canonicalization
   feed current erased paths, while standalone RMS and exact named-attention
   recovery now serve only `reference_pipeline`.
4. Route public GDN/KDA convenience compilation through the structured
   StableHLO scan importer and quarantine `delta_rule_update_expression` as a
   recovery-unit fixture.
5. Rename `MSADebugConfig`, `RoutedAttentionDebugConfig`, and the streaming
   backward debug exporter around their actual role as natural JAX examples.
6. Replace the `xla_*` HLO text/regex bridges with typed post-SPMD region
   recovery once the semantic normal form stabilizes. Until then, keep them
   labeled experimental and outside frontend acceptance claims.
