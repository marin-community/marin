# Event Tensor right-resource JAX/CuTe GB200 compile checkpoint

This artifact records the first bounded device attempt for the Torch-free JAX
right-resource path. The reservation supplied one 4xGB200 bare-metal node; the
smoke exposed only GPU 0 and requested two CPUs. The full reservation was
released immediately after the second, conclusive compile result.

The attempt did not launch a device kernel. It therefore provides no routed
attention correctness, determinism, latency, overlap, or performance evidence.
It does establish that the dependency-preflight path reaches CUTLASS CuTe
compilation on a real GB200 without Torch or an opaque semantic kernel.

The first compile at Shuttle revision `7a0891feef` exposed an obvious wrapper
ABI error: the JAX wrapper passed two generic relation-layout operands through
to an extracted physical method that does not accept them. Revision
`728d0dfcd4` removes those two physical-call arguments while retaining the
operands in the generic JAX/EventTensor boundary. Focused EventTensor tests and
the changed-file checks pass.

The single rerun advanced into the extracted SM100 method body, then stopped
while constructing the launch configuration. The extracted physical template
uses `grid=(work_capacity,)`; under `cutlass.jax.cutlass_call`, that value was
represented as an MLIR `BlockArgument`, while the CUTLASS DSL launcher requires
a compile-time integer grid dimension.

The smallest next experiment is host-side specialization of the bounded work
capacity in the generic physical launch ABI. It should preserve the runtime
`work_count` operand for tail work while making only the maximum grid extent a
compile-time schedule parameter. Before another GB200 allocation, a Linux
preflight should lower far enough to prove that the launch grid is a constant.

The outer EventTensor schedule remains Shuttle-owned. Internal TMA/tcgen05
`mbarrier` sites and phase advancement remain primitive-owned. `result.json`
contains the source pins, exact failure stages, fingerprints inherited from the
identical green host preflight, and the explicit absence of device results.

