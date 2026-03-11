# Session Directive: Whole-Layer Boundary Prototype

Goal:
- prototype a materially different optimization boundary: the entire GDN-bearing decoder layer.

Prototype shape:
- keep the exact `3/4` GDN math and benchmark setup,
- optimize the whole decoder-layer boundary instead of the chunk-kernel boundary,
- prefer XLA-visible shell + Pallas leaf kernels initially,
- target the costs now dominating the unexplained gap:
  - sharding shell,
  - AD shell,
  - layout/reshape shell,
  - residual/add shell.

Preferred scope for the first prototype:
- projections / gates,
- conv / norm / gating path,
- chunked GDN primitive,
- output projection,
- residual/add,
- backward boundary / custom VJP at the whole-layer level.

What not to do:
- do not spend the whole iteration on another chunk-local tape or solver tweak,
- do not mix in unrelated CE experiments,
- do not treat the first prototype as a promotion candidate unless it materially improves `step_duration_ms`.
