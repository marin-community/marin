# Session Directive: XLA-First Outer Train Path

Goal:
- Shift the outer train-path orchestration toward XLA/JAX and keep Pallas only for the leaf chunk kernels where it is clearly beneficial.

What this means:
- Do not make Pallas/custom-VJP responsible for the whole training shell.
- Consider pure-JAX/XLA outer composition, state propagation, remat/checkpoint boundaries, and backward structure.
- Use Pallas only for dense chunk-local transforms, solves, or recurrent microkernels if they still win.

Why this direction matters:
- Current evidence suggests the train-shell lowering is the real bottleneck.
- A reduced-Pallas or XLA-first branch is the fastest way to test whether the current abstraction boundary is boxing the search in.

Success criterion:
- The branch is only interesting if it improves end-to-end training throughput or clearly reduces `while` / `conditional` overhead, even if some leaf kernels become slower in isolation.
