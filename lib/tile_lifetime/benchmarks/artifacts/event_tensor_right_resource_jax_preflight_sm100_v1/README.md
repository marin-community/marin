# Event Tensor static-grid JAX/CuTe preflight

The CPU-only Linux preflight passed on Iris job
`/dlwh/shuttle-event-jax-static-grid-preflight` from Shuttle commit
`a21e0d0ecd`. No accelerator was requested or allocated.

This rerun corrects the physical positional ABI exposed by the first GB200
compile attempt. It also specializes the bounded maximum work capacity as a
CuTe compile-time parameter while retaining `work_count` as a JAX device
operand. A source/ABI audit proves that the extracted physical method annotates
capacity as `cutlass.Constexpr[int]`, aliases it to `num_ctas`, and launches on
that static grid. Runtime counts greater than the specialized capacity are
rejected before physical launch.

The preflight compiled and registered the generated typed-FFI Fold finalizer,
imported the pinned generic SM100 extraction, and instantiated
`cutlass.jax.cutlass_call` as a JAX `PjitFunction`. A relation-only mutation
retained the Event Tensor program fingerprint and launch capacity while
changing the runtime fingerprint. Torch was neither installed nor loaded.

This is a host-side dependency, source, and ABI gate. It does not establish
device compilation, execution, correctness, determinism, or latency. A future
GB200 retry remains separately gated on review of this artifact.

