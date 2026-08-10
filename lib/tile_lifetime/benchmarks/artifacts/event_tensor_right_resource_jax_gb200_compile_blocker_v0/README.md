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
ABI error: the JAX wrapper supplied one extra optional-metadata placeholder.
Revision `728d0dfcd4` removed the positional mismatch, but did so incorrectly by
removing the two generic sequence-offset operands required by the extracted
physical ABI.

The single rerun advanced into the extracted SM100 method body, then stopped
while constructing the launch configuration. Because the interim fix shifted
the remaining arguments, the JAX stream block argument occupied the physical
`work_capacity` position and appeared in `grid=(work_capacity,)`. CUTLASS
correctly rejected that MLIR `BlockArgument` as a grid dimension. This was a
second ABI symptom, not evidence that a correctly positioned Python capacity
would itself become dynamic.

The smallest next experiment is a corrected positional ABI plus explicit
host-side specialization of the bounded work capacity. It should preserve the
runtime `work_count` operand for tail work while making only the maximum grid
extent a compile-time schedule parameter. Before another GB200 allocation, a
Linux preflight should prove the constant-grid source and ABI contract.

The outer EventTensor schedule remains Shuttle-owned. Internal TMA/tcgen05
`mbarrier` sites and phase advancement remain primitive-owned. `result.json`
contains the source pins, exact failure stages, fingerprints inherited from the
identical green host preflight, and the explicit absence of device results.
