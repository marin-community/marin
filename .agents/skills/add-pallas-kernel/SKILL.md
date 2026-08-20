---
name: add-pallas-kernel
description: Add, modify, or autotune a TPU/GPU Pallas kernel.
---

# Add or Update a Pallas Kernel

Build the reference first, prove values and gradients, then implement and
measure the accelerator path. For long-running kernel research, load
`run-research` first. Read only the routed detail needed:

- `docs/kernel-sources.md` when selecting an implementation to imitate;
- `docs/performance-workflow.md` before benchmarking, profiling, roofline work,
  or autotuning;
- `docs/api-patterns.md` before changing wrappers, fallback order, or block
  configuration;
- `docs/tpu-tips.md` / `docs/gpu-tips.md` for backend-specific work;
- `docs/reference/` only when those detail files route there.

## Required shape

For kernel `K`, provide a readable vanilla JAX reference with the public API,
value and small-shape gradient checks, a Pallas implementation with the same
wrapper API, representative steady-state benchmarks, and a roofline estimate.
When tuning is requested, check in the tuned table and a bounded
autotune-on-miss candidate sweep with cached winners for the requested hardware
and shape regimes. Keep the reference
usable when accelerator constraints are not met.

## Correctness and implementation

Use an obvious stable reference (JAX, in-repo, pseudocode, or PyTorch); use a
streaming/blockwise reference if naive intermediates are huge. Check a
shape/dtype grid, max/mean absolute deviations as well as `allclose`, gradients
on small shapes, and CPU plus relevant accelerator backends. Add durable tests
under `lib/levanter/tests/kernels/` after reading root `TESTING.md` and the
nearest module guidance.

Wrap accelerator kernel boundaries (`pl.pallas_call`, Mosaic GPU, FFI) in an
explicit `jax.shard_map` by default. Reshard to the intended local
`PartitionSpec`, leave nonlocal dimensions unsharded unless the kernel supports
them, and test lowered JAXPR/HLO for the expected boundary. A wrapper may omit
it only when inputs are documented and tested fully local/replicated. Validate
against the reference before tuning; inspect compiler dumps/pressure/tile
choices when performance misses the roofline.

Read `docs/api-patterns.md` for public wrapper, validation, backend/fallback,
and block-size conventions. Keep backend validation in backend modules.

Every `pl.pallas_call` must include a reviewed `cost_estimate=` based on a
body-equivalent JAX function, including I/O bytes:

```python
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

body_cost = pl.estimate_cost(reference_impl, q, k, v)
cost = with_io_bytes_accessed(
    body_cost,
    kernel_inputs_specs=kernel_inputs_specs,
    kernel_outputs_specs=kernel_outputs_specs,
)
```

Do not call `pl.estimate_cost` on a body using `pl.program_id`.

## Exit criteria

Values/gradients meet reference tolerances; CPU/reference and accelerator paths
are tested; wrapper/fallback/block config and tuned-table behavior follow the
API guide; every opaque kernel has the required sharding boundary or a tested
locality exception; cost estimates include I/O; benchmark/tuning artifacts use
the performance-workflow schema; and realistic target performance improves or
the limitation is documented. Long-running work also has the logbook, issue,
artifact, and snapshot records required by `run-research`.
