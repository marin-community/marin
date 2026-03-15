---
name: add-pallas-kernel
description: Add or update a TPU kernel using jax.experimental.pallas. Use when asked to implement, modify, benchmark, or autotune a Pallas TPU/GPU kernel.
---

# Skill: Add or Update a Pallas Kernel

This is a specialization of `.agents/skills/agent-research/SKILL.md`.

Use `agent-research` for the generic research lifecycle (branching, issue/logbook cadence,
snapshot/tag discipline, and reporting). This skill adds kernel-specific standards for:

- numerics and gradient safety,
- backend/fallback API design,
- TPU/GPU performance diagnosis,
- block-size autotuning and tuned-table outputs.

## How to apply this skill

1. Load and follow `.agents/skills/agent-research/SKILL.md` first.
2. Apply the additional kernel rules in this document.
3. Keep shared process details in `agent-research`; keep this file focused on kernel-specific constraints.

## Kernel Deliverables

For a kernel `K`, produce:

- A readable **vanilla JAX reference** implementation with the target public API.
- A **Pallas kernel implementation** plus wrapper with the same API.
- A **correctness harness** that validates:
  - value parity vs reference,
  - gradient parity on small shapes,
  - CPU and accelerator numerics where applicable.
- A **performance harness** with steady-state timing on representative shape/dtype grids.
- **Autotuned block/tile sizes** for requested hardware/shape regimes.
- A checked-in **tuned table module** for runtime selection (with explicit fallback behavior).
- An **autotune-on-miss fallback path** that can sweep a bounded candidate set and cache winning configs for later reuse.

Use the research logbook and issue workflow from `agent-research` for experiment history and milestone updates.

## Recommended Module Layout

Tokamax-style decomposition is preferred for maintainability:

- `reference.py`: readable vanilla JAX oracle.
- `xla.py`: default implementation (often same math as reference).
- `pallas_tpu.py`: TPU Pallas implementation.
- `pallas_gpu.py`: optional GPU Pallas implementation.
- `api.py`: stable user-facing entrypoint with `implementation=` override and fallback order.

Reference template:
- `lib/levanter/src/levanter/kernels/pallas/template_kernel.py`

## API and Safety Rules

### Batching convention

Prefer one true batched kernel:

- Implement the core kernel for batched inputs.
- Normalize single-example inputs by temporarily adding a leading batch dimension.
- Reshape leading axes into one batch axis when needed, then restore on output.
- Preserve explicit parallel-dimension semantics on at least one axis (usually batch) for TPU kernels.

### Block size config

Expose tile choices via a dataclass and keep defaults explicit:

```python
@dataclass(frozen=True, slots=True)
class BlockSizes:
    b_block_size: int = 1024
    h_block_size: int = 512
    v_block_size: int = 2048

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()
```

Rules:

- Validate TPU-specific alignment constraints (for example multiples of 128) in the TPU backend.
- Keep reference/XLA paths usable even when TPU constraints are not met.
- If Mosaic reports a layout mismatch for a batched integer operand (for example labels), align the batch block size to
  the XLA tile size used on that TPU generation or raise a clear pre-lowering error.
- If a legacy `block_size` arg exists, map it clearly to the new config and raise on conflicting inputs.

### Fallback semantics

- If a single implementation is explicitly requested (for example `implementation="pallas_tpu"`), fail fast on
  unsupported backend/shape.
- If a sequence of implementations is requested, try each in order, warn on each fallback, and raise if none work.
- If using a default implementation order, treat it the same as a sequence and keep behavior explicit.
- Keep backend selection behavior explicit and predictable in `api.py`.

### Input normalization rule

Prefer a canonical kernel input shape and make callers normalize to it:

- Define one canonical shape contract for the kernel (for example rank-2/1/2 forms).
- Expect callers/users to flatten or reshape batch axes before kernel invocation.
- If you provide wrapper reshaping helpers, keep them thin and explicit at API boundaries.

## Correctness Workflow

### 1) Start from a reference

Use one of:

- existing implementation in repo,
- pseudocode,
- PyTorch reference,
- Optax/JAX baseline.

Baseline must be obvious and stable, not clever.

If the naive baseline would materialize huge intermediates, use a streaming/blockwise baseline with identical math so
correctness checks stay feasible.

### 2) Write value + grad harness

Minimum checks:

- value parity over a shape/dtype grid,
- gradient parity on small shapes,
- backend numerics on CPU and accelerator backends as applicable.

Report pointwise deviation metrics (max/mean absolute diff), not only `allclose`.

Use explicit shape/dtype annotations for public APIs and references (for example via `jaxtyping`) where available.

### 3) Promote long-lived checks to pytest

For kernels that remain in-tree, add/extend tests under:

- `lib/levanter/tests/kernels/`

Compare default implementation against the reference on small CPU shapes and accelerator-aligned shapes for fast paths.

## Cost Estimate Requirement

Add `cost_estimate=` to each `pl.pallas_call`:

- Use `pl.estimate_cost` on a body-equivalent JAX function (not a kernel body with `pl.program_id`).
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

## Performance and Profiling Workflow

Use the execution environment guidance and cadence from `agent-research`; this section only adds kernel-specific constraints.
For kernel-specific profiling capture/compare guidance, see `docs/reference/profiling.md`.

Key iteration loop:
- `profile -> hypothesis -> change -> tests -> microbench -> profile`

Always report:

- compile-including timing (`time-to-first-step`),
- steady-state timing,
- exact hardware type and shape/dtype grid.

## Autotuning Workflow

Keep tuning explicit and reviewable.

1. Define a bounded config space (block/tile candidates).
2. Define target shape/hardware buckets.
3. Benchmark every `(bucket, config)` pair and capture timing + failures.
4. Store raw results as artifacts (CSV/JSON; W&B artifact preferred).
5. Derive best-config table keyed by `(tpu_type, dtype, shape_bucket[, invariants])`.
6. Check in a Python tuned-table module with:
   - bucket definitions,
   - best configs,
   - `infer_block_sizes(...)` helper,
   - default fallback to `BlockSizes.get_default()`.

Do not key tuned tables by every exact shape; keep buckets stable and reviewable.

### Fallback autotuning requirement

Support two levels of fallback tuning, similar to the fused softmax cross-entropy kernel:

1. Static lookup fallback:
   - infer block sizes from a checked-in tuned table by `(device, dtype, shape bucket)`,
   - validate/sanitize for backend constraints,
   - fall back to default/safe entries when no exact tuned match exists.
2. Autotune-on-miss fallback:
   - when tuned lookup misses (and autotune is enabled), sweep a bounded candidate list,
   - benchmark candidates on the real implementation, select the best viable config,
   - cache and persist the winner under a kernel-specific key (include implementation + shape/device/dtype context).
3. Runtime failure fallback:
   - if a candidate or implementation is unsupported (for example compile/runtime constraints), warn and try the next
     candidate/implementation in order when a sequence is available.

## Compiler and Runtime Hints (TPU Pallas)

### Matmul precision

If Mosaic reports errors like `Expected matmul acc to be 32-bit`:

- set `preferred_element_type=jnp.float32` in `lax.dot_general` for the kernel path, or
- set `jax_default_matmul_precision=highest` in benchmark scripts.

Prefer explicit kernel-side `preferred_element_type` for deterministic behavior.

### Scoped VMEM policy

Set `LIBTPU_INIT_ARGS` by TPU generation during microbench/tuning:

- `v5p` / `v5e`: `--xla_tpu_scoped_vmem_limit_kib=50000`
- `v6e`: `--xla_tpu_scoped_vmem_limit_kib=98304`
- `v4`: no special scoped-VMEM override

### Compiler diagnostics and dumps

Capture compiler diagnostics on serious benchmark/tuning runs:

- HLO dumps via `--xla-dump-dir`,
- compiler logs via `--compiler-log-path`,
- explicit `XLA_FLAGS` and `LIBTPU_INIT_ARGS` recorded with results.

Useful scripts:

- `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
- `lib/levanter/scripts/tune/tune_fused_cross_entropy_loss_block_sizes.py`

### Dump-driven diagnosis

When performance is unclear, run dump-first comparisons on one fixed shape:

- XLA/reference path,
- full Pallas path,
- decomposition variant(s) (temporary toggles).

Use separate dump dirs per variant (`hlo_*`, `llo_*`, `mosaic_*`) and compare:

- throughput,
- fusion/custom-call placement,
- schedule bundle counts,
- pressure signals (e.g. heavy `vrot`/`vsel`, spills, vreg pressure).

Prefer structural fixes before broad tile sweeps when decomposition variants indicate stage-structure issues.
For the full LLO workflow (flags, artifact layout, comparison checklist, and replication loop), see
`docs/reference/llo.md`.

## Definition of Done

- Values match reference within tolerance on tested grid.
- Gradients match reference on small shapes.
- Performance improves on at least one realistic target shape, or limitations are explicitly documented.
- Tuned table is checked in for requested hardware/shape regimes.
- Research artifacts (logbook updates, issue summary, snapshot links) follow `agent-research` workflow.

## Starter References

- `lib/levanter/src/levanter/kernels/pallas/template_kernel.py`
- `lib/levanter/tests/kernels/test_pallas_template_kernel.py`
- `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss`

## Tokamax Notes (Optional)

Tokamax kernels can be useful references for API and kernel structure comparisons.

- Typical install path in this repo's uv environment:
  `.venv/lib/python3.11/site-packages/tokamax/_src/ops`
- Compare numerics/perf on identical shapes/dtypes before drawing conclusions.
- Parse `absl.flags` before accessing Tokamax modules that depend on flags.
- Tokamax Mosaic kernels can OOM VMEM at larger shapes; reduce shape/tile sizes for controlled comparisons.

## Further Reading

- [JAX Pallas Overview](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX Pallas TPU Docs](https://docs.jax.dev/en/latest/pallas/tpu/index.html)
- [JAX Pallas Mosaic GPU Docs](https://docs.jax.dev/en/latest/pallas/gpu/index.html)
- [When XLA Isn't Enough: From Pallas to VLIW](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
- `docs/reference/llo.md`
- `docs/reference/profiling.md`
