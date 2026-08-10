# Event Tensor right-resource GB200 linkage

This artifact preserves the bounded Torch-free JAX/CuTe linkage smoke from
Iris job `/dlwh/shuttle-event-right-resource-gb200-linkage-2` at Shuttle commit
`ac475ef28b`. The job succeeded on one NVIDIA GB200 and released its tray after
34.59 seconds.

The reduced smoke uses query length 128, key length 1024, 16 left heads, two
right heads, head dimension 128, and four selected right resources per left
item. Its baseline RelationPlan is nonmonotone and leaves right resource 7
empty. A relation permutation changes runtime tables and output while reusing
the same compiled Event Tensor program and work capacity.

The physical path is Torch-free:

1. generic RelationPlan tables form right-major grouped work;
2. the compiler-derived EventTensorPlan describes resource readiness, partial
   Fold readiness, and single-slot resource reuse;
3. `cutlass.jax.cutlass_call` invokes the generic SM100 grouped Contract/Fold
   physical class;
4. the generated typed-FFI Fold finalizer runs on the same JAX stream;
5. the result is checked against the semantic JAX program.

Both relation cases are correct within the declared tolerance and bitwise
deterministic across ten retained executions. Raw latencies are included only
as diagnostics. This is linkage evidence, not an overlap or performance claim.

The outer Event Tensor realization is compiler-owned. Internal SM100 pipeline
`mbarrier` sites remain owned by the audited low-level physical primitive.
There are no opaque external semantic kernels in the accepted path.

`result.json` is a compact lossless record of the measured values and the
non-repetitive task/event/buffer audit. The full event arrays are recoverable
from the terminal Iris log; their exact repeated values and extents are stated
in the compact representation.

