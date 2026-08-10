# Shuttle

Shuttle synthesizes progressively more physical interpretations of ordinary
JAX programs. JAX and XLA own tracing, differentiation, and initial lowering.
Shuttle receives StableHLO, imports its typed dataflow without retaining MLIR
objects, and will lower selected regions through this sequence:

```text
typed StableHLO
  -> Map / Contract / Fold / Scan / Relation / DomainRestriction / Transport
  -> algebraic rewrites and materialization choices
  -> tiled tasks, buffers, and exact dependencies
  -> EventTensor readiness plans where synchronization is useful
  -> physical layouts, pipelines, and generated kernels
```

The current package owns the shared `DType` vocabulary and a typed importer for
an explicit supported StableHLO subset. Lossless module import remains a
target-1 requirement. The package does not expose workload-named compiler
entrypoints, kernel selectors, benchmark references, or performance oracles.
Historical and bounded workload prototypes remain in `tile_lifetime` while
their generic parts are moved into this package.

The next migration seam is the generic attention importer once its independent
cleanup lands. That importer must consume the structures in this package; this
package must not depend on `tile_lifetime`.
