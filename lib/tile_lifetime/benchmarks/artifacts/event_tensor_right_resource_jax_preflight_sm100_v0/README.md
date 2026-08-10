# Event Tensor right-resource JAX/CuTe preflight

The CPU-only Linux preflight passed on Iris job
`/dlwh/shuttle-event-jax-preflight-7` from Shuttle commit `a407da4e4d`.
No accelerator was requested or allocated.

The preflight compiled and registered the generated typed-FFI Fold finalizer,
imported the pinned generic SM100 extraction, and instantiated its
`cutlass.jax.cutlass_call` as a JAX `PjitFunction`. A relation-only mutation
retained the Event Tensor program fingerprint and changed the runtime
fingerprint. The environment did not install or load Torch.

This result establishes dependency, source, handler, and host-side ABI linkage.
It does not establish device execution, correctness, determinism, or latency.
Those require the bounded GB200 experiment in
`docs/event_tensor_routed_attention_gpu_plan.md`.
