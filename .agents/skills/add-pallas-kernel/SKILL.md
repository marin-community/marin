---
name: add-pallas-kernel
description: Add or change a named Pallas/Mosaic kernel, including its reference implementation, correctness tests, wrapper, or requested tuning.
---

# Add or update a Pallas kernel

## How to apply this skill

Load only the detail files needed for the requested work:
   - [Kernel sources](docs/kernel-sources.md): read when choosing an in-repo or
     external kernel to imitate.
   - [Performance workflow](docs/performance-workflow.md): read before
     benchmarking, profiling, roofline analysis, or autotuning.
   - [API patterns](docs/api-patterns.md): read before adding or changing a
     public kernel wrapper, fallback order, or block-size config.
   - [TPU tips](docs/tpu-tips.md): read for TPU Pallas/Mosaic kernels,
     TPU-specific lowering failures, scoped VMEM, or TPU compiler dumps.
   - [GPU tips](docs/gpu-tips.md): read for GPU Pallas/Mosaic work.
   - Deep references live under `docs/reference/`; read them only when the
     routed detail files point there.

Use `run-research` only when the user explicitly requests its multi-session
research workflow.

## Kernel Deliverables

For a kernel `K`, produce:

- Vanilla JAX reference and Pallas wrapper with the same public API.
- Value, gradient, CPU, and applicable accelerator parity harness.
- Explicit backend and shape validation, with tests for ordered implementation
  selection and each fallback path.
- Roofline estimate and steady-state benchmark on representative shapes/dtypes.
- When tuning is requested, bounded autotuning, a checked-in tuned table,
  explicit fallback, and cached autotune-on-miss results.

## Correctness Workflow

### 1. Start from a reference

Use an existing in-repo implementation, pseudocode, a PyTorch reference, or a
JAX baseline. The baseline must be obvious and stable, not clever. If the naive
baseline would materialize huge intermediates, use a streaming/blockwise
baseline with identical math.

### 2. Write a value and gradient harness

Minimum checks:

- Value parity over a shape/dtype grid.
- Gradient parity on small shapes.
- Backend numerics on CPU and accelerator backends as applicable.
- Pointwise deviation metrics such as max/mean absolute diff, not only
  `allclose`.

Use explicit shape/dtype annotations for public APIs and references, such as
`jaxtyping`, where available.

### 3. Promote long-lived checks to pytest

For in-tree kernels, add or extend tests under `lib/levanter/tests/kernels/`.
Compare the default implementation against the reference on small CPU shapes and
accelerator-aligned shapes for fast paths. Read `TESTING.md` and the nearest
module `AGENTS.md` before writing or changing tests.

## Pallas Kernel Workflow

Once the reference is correct, design the Pallas implementation. Use the
reference as both a correctness oracle and a performance baseline.

Use existing kernels for structure and API inspiration. Read
[Kernel sources](docs/kernel-sources.md) unless the user already named the
specific kernel to follow. Unless there is a stronger local pattern, start by
reimplementing the reference in Pallas.

Wrap accelerator kernel boundaries in an explicit `jax.shard_map` by default.
This applies to `pl.pallas_call`, Mosaic GPU kernels, and custom FFI calls.
Reshard inputs to the intended local `PartitionSpec` before the `shard_map`,
keep the sequence or other nonlocal dimensions unsharded unless the kernel is
explicitly written for them, and add a regression check that the lowered JAXPR
or HLO contains the expected `shard_map`. Do not rely on XLA to infer a good
sharding for an opaque kernel call boundary. Exceptions are limited to wrappers
whose inputs are explicitly documented and tested as fully local or replicated.

Check correctness against the harness and reference implementation before
tuning. Once the kernel is correct, run a performance harness on representative
shapes/dtypes and compare against the roofline. If performance is not near the
expected roofline, read [Performance workflow](docs/performance-workflow.md) and
investigate compiler dumps, pressure signals, and tile choices before broad
rewrites.

## API Conventions

Read [API patterns](docs/api-patterns.md) before adding or changing the public
wrapper, backend selection, block-size config, or input normalization contract.
Keep the reference/XLA path usable even when accelerator-specific constraints
are not met. Keep backend-specific validation in backend-specific modules.

## Cost Estimate Requirement

Add `cost_estimate=` to each `pl.pallas_call`:

- Use `pl.estimate_cost` on a body-equivalent JAX function, not a kernel body
  with `pl.program_id`.
- Include IO bytes from call inputs/outputs.

```python
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


def _cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(reference_impl, q, k, v)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )
```

## Definition of Done

- Values match reference within tolerance on the tested grid.
- Gradients match reference on small shapes.
- CPU/reference and accelerator fast paths are covered by tests where
  applicable.
- Public API, fallback semantics, block-size config, and tuned table behavior
  match [API patterns](docs/api-patterns.md).
- Every Pallas, Mosaic, or FFI kernel call is inside an explicit `shard_map`, or
  its wrapper documents and tests why the inputs are fully local or replicated.
  Tests or profile evidence show it did not lower through unintended
  all-gathers.
- Each `pl.pallas_call` has a reviewed `cost_estimate=`.
- Benchmark/tuning artifacts include the required schema from
  [Performance workflow](docs/performance-workflow.md).
- Roofline performance is within expected bounds, or limitations are explicitly
  documented.
- Performance improves on at least one realistic target shape, or limitations
  are explicitly documented.
- Tuned table is checked in for requested hardware/shape regimes.
- Research artifacts, issue summaries, and snapshot links follow the
  `run-research` workflow when the task is long-running.
