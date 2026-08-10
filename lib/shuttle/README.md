# Shuttle

Shuttle is an MLIR compiler extension for ordinary JAX programs. JAX owns
tracing and differentiation. Shuttle converts selected StableHLO regions inside
the compilation pipeline and lowers them through this sequence:

```text
StableHLO MLIR
  -> Map / Contract / Fold / Scan / Relation / DomainRestriction / Transport
  -> algebraic rewrites and materialization choices
  -> tiled tasks, buffers, and exact dependencies
  -> EventTensor readiness plans where synchronization is useful
  -> physical layouts, pipelines, and generated kernels
```

The current package owns the shared `DType` vocabulary and the closed,
canonical compiler-options schema. Stock jaxlib rejects these options; a
Shuttle-enabled build must recognize them and run the native MLIR pass. The
package does not expose workload-named compiler entrypoints, kernel selectors,
benchmark references, or performance oracles.

The Python StableHLO parser is retained under `shuttle.experimental` for
historical executable specifications. It is not the production compiler path.
Other bounded prototypes remain in `tile_lifetime` while their generic MLIR,
planning, and lowering components move here. This package must not depend on
`tile_lifetime`.
