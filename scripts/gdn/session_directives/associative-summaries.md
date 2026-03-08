# Session Directive: Associative Chunk Summaries

Goal:
- Re-express the training path so chunk state propagation is less serial and less dependent on a large device-side scan shell.

Core idea:
- Model each chunk as an affine summary on recurrent state, e.g. `S_out = A_chunk(S_in) + B_chunk`.
- Compose chunk summaries associatively.
- Use prefix-style state propagation where possible instead of a large serial segment scan.

What to look for:
- Can the train-path forward compute chunk-start states from chunk summaries rather than a `scan` over chunk kernels?
- Can backward reuse chunk summaries or adjoint summaries rather than a reverse scan over large per-segment tape tuples?
- Can chunk-local Pallas kernels become leaf primitives that emit summaries rather than participating in the whole train shell?

Constraints:
- Correctness still matters; if you introduce a probe/ablation, label it clearly.
- Prefer designs that reduce compiler-visible control flow, not just arithmetic inside one kernel.
