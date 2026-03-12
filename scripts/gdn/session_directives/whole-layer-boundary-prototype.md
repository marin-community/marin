# Session Directive: Fixed-4-Layer Block Prototype With Manual Backward

Goal:
- prototype a materially different optimization boundary: the fixed `3 GDN + 1 attention` block.

Why this boundary:
- Iteration 90 proved that a coarser forward wrapper alone is not enough.
- If the block still relies on generic JAX AD and generic sharding around the boundary,
  the cost simply re-emits as `HackableDecoderBlock/*` shell.

Prototype requirements:
- keep the exact `3/4` GDN math and benchmark setup,
- optimize the fixed 4-layer block as a unit,
- own all three of:
  - the forward block boundary,
  - the backward / AD strategy,
  - the sharding/layout contract.

Preferred first prototype:
- XLA-visible shell + existing leaf kernels,
- manual/custom VJP at the block boundary,
- explicit sharding contract instead of letting generic wrappers rebuild the shell.

What not to do:
- do not spend the whole iteration on another chunk-local tape or solver tweak,
- do not leave backward to generic JAX AD,
- do not treat renamed bucket movement as progress unless the full step and hybrid-specific shell delta both improve.
