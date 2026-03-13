# Session Directive: Fixed-4-Layer Block Prototype With Manual Backward And Sharding

Goal:
- prototype a materially different optimization boundary: the fixed `3 GDN + 1 attention` block.
- the prototype must directly target `dispatch_shard_shell_delta_ms` first and `ad_wrapper_shell_delta_ms` second.

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
- Success is defined by:
  - lower `dispatch_shard_shell_delta_ms`,
  - no regression in `ad_wrapper_shell_delta_ms`,
  - no growth in `interaction_remainder_ms`,
  - and a real `step_duration_ms` improvement.

What not to do:
- do not spend the whole iteration on another chunk-local tape or solver tweak,
- do not leave backward to generic JAX AD,
- do not treat renamed bucket movement as progress unless the full step, canonical shell delta, and xprof-IDLE all improve or stay neutral.
