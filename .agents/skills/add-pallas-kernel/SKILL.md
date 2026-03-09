---
name: add-pallas-kernel
description: Add a new TPU kernel using jax.experimental.pallas. Use when asked to implement, benchmark, or autotune a Pallas TPU/GPU kernel.
---

# Skill: Add a Pallas Kernel (Agent-Oriented)

This skill describes a repeatable workflow for adding a new TPU kernel using
`jax.experimental.pallas` in a way that is:

- **safe** (numerics and gradients checked),
- **fast** (microbench + profiling evidence),
- **reviewable** (clear baseline + clear diff),
- **agent-friendly** (explicit checkpoints; no silent guessing).

## What you produce

For a new kernel `K`, you should produce:
- A **vanilla JAX reference implementation** (readable, correct, stable API).
- A **Pallas kernel implementation** + wrapper function with the same API.
- A **test harness** that checks:
  - values match reference (within tolerance),
  - gradients match reference on small shapes,
  - the kernel is jittable and works on representative shapes/dtypes.
- A **perf harness** that compares speeds for configurable shapes and dtypes.
- **Autotuned block sizes** for a reasonable range of sizes/parameters.
- A **running report** in `.agents/projects/<short-topic>.md` that logs nontrivial iterations, perf numbers, tuning results, and any pitfalls.
- A **short report** (in the PR description) summarizing the tested shape/dtype grid, perf results, and known limitations.

## Recommended module layout (Tokamax-style)

For Levanter kernels:
- `reference.py`: readable vanilla JAX oracle
- `xla.py`: default implementation (often identical to reference)
- `pallas_tpu.py`: TPU/Pallas implementation (optional import)
- `pallas_gpu.py`: optional GPU/Pallas implementation (if applicable)
- `api.py`: stable user-facing function with `implementation=` override and fallback order

See `lib/levanter/src/levanter/kernels/pallas/template_kernel.py` for a minimal example.

## Batching convention (recommended)

- Implement **one** "true" kernel for the **batched** case.
- The public API accepts either batched inputs or a single unbatched example and normalizes by adding/removing a trivial leading batch dimension.

## Block size config (recommended)

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

Notes:
- For TPU Pallas, block sizes are typically required to be multiples of 128 (DMA alignment).
- If Mosaic reports a layout mismatch for a batched integer operand, align the batch block size to the XLA tile size.

## Workflow

### 1) Start from a reference

Pick one:
- existing implementation in the repo (best),
- pseudocode,
- PyTorch reference,
- Optax/JAX baseline.

### 2) Write the vanilla JAX baseline

Write a baseline in plain JAX that:
- makes shapes and dtypes explicit,
- is obviously correct,
- has a stable signature you want to keep.

Avoid cleverness here: the baseline is your oracle.

Use `jaxtyping` for public kernel APIs and reference implementations.

### 3) Build a correctness harness (value + grad)

At minimum:
- value match vs baseline on a grid of shapes/dtypes,
- grad match on small shapes (finite difference or `jax.grad` vs baseline),
- validate numerics on CPU and on accelerator backends.

When comparing outputs, prefer **pointwise deviation** metrics (e.g. max/mean absolute diff) in addition to `allclose`.

### 4) Implement the Pallas kernel + wrapper

Guidelines:
- keep the wrapper pure and explicit (no hidden global state),
- keep the "edit surface" obvious (tile sizes, block sizes),
- make failure modes actionable (shape constraints, alignment, etc.).

**Fallback semantics (recommended):**
- If a specific implementation is selected, **error** on unsupported shapes.
- If the default implementation order is used, **warn and fallback** to XLA/reference.

**Pallas cost estimates (required for new kernels):**
Add a `cost_estimate=` argument on each `pl.pallas_call`. Use `pl.estimate_cost` on a reference/body-equivalent JAX function.

```python
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

def _cost_estimate(q, k, v, *, kernel_inputs_specs, kernel_outputs_specs):
    body_cost = pl.estimate_cost(reference_impl, q, k, v)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )
```

### 5) Add a speed microbench + profiling hook

Minimum:
- microbench step time vs baseline on a representative shape,
- (optional but encouraged) a profiling run that produces an xprof artifact.

#### Running on TPU (Marin Ray infra)

```sh
RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \
  --cluster marin-us-central2-staging \
  --tpu v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT "marin-kernels" \
  -- python XXX
```

##### Scoped VMEM flag policy

- `v5p`/`v5e`: `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"`
- `v6e`: `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304"`
- `v4`: do **not** set a special scoped VMEM limit flag.

##### Profiling with Levanter (recommended)

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 50 \
  --trainer.profiler.perfetto_link false
```

Always report:
- time-to-first-step (includes compilation) vs steady-state step time
- the exact TPU type and shape/dtype grid tested

### 6) Iterate using evidence

`profile -> hypothesis -> change -> tests -> microbench -> profile`

Use `.agents/skills/agent-profiling/` tooling.

### 7) TPU dump-driven workflow (generic)

#### Required dump flags

```bash
export XLA_FLAGS="\
  --xla_dump_to=${HLO_DIR} \
  --xla_dump_hlo_as_text"

export LIBTPU_INIT_ARGS="\
  --xla_jf_dump_to=${LLO_DIR} \
  --xla_jf_dump_hlo_text=true \
  --xla_jf_dump_llo_text=true \
  --xla_jf_dump_llo_html=false \
  --xla_jf_dump_llo_static_gaps=true \
  --xla_jf_emit_annotations=true \
  --xla_jf_debug_level=2 \
  --xla_mosaic_dump_to=${MOSAIC_DIR} \
  --xla_mosaic_enable_dump_debug_info=true \
  --xla_mosaic_enable_llo_source_annotations=true"
```

#### XLA-LLO Replication Playbook

1. Start from one exact shape and freeze it
2. Dump both sides first (XLA/reference and Pallas)
3. Infer algorithm structure from XLA fusions
4. Replicate the simplest stage first (don't port everything at once)
5. Add complexity incrementally, re-checking correctness and throughput at each step
6. Use LLO counters as diagnostics, not just timings
7. Prefer structural fixes before tile sweeps
8. Validate numerics with more than one metric (abs_linf, rel_l2, loss delta)
9. Retune after structural convergence

## Autotuning (v0)

Produce a checked-in Python "tuned table" module:
- mapping from `(tpu_type, dtype, shape_bucket)` to best config,
- a single `infer_block_sizes(...)` helper,
- fallback to `BlockSizes.get_default()` if no match.

## Starter template

- `lib/levanter/src/levanter/kernels/pallas/template_kernel.py`
- `lib/levanter/tests/kernels/test_template_kernel.py`

## Definition of Done

- Correctness: value matches baseline within tolerance across the tested grid.
- Gradients: gradients match baseline on small shapes.
- Performance: measurable improvement on at least one realistic shape (or clear explanation why not).
- Documentation: PR includes a short "what we tested / what improved / what's next" summary.
- Block sizes: autotuned block sizes checked in for the requested shape/dtype grid and device types.

## Further Reading

- [JAX Pallas Overview](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX Pallas TPU Docs](https://docs.jax.dev/en/latest/pallas/tpu/index.html)
- Tokamax kernels: `.venv/lib/python3.11/site-packages/tokamax/_src/ops`
- JAX built-in kernels: `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops`
- `lib/levanter/docs/Performance-Guide.md`
