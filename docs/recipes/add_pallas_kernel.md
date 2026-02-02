# Add a Pallas Kernel (Agent-Oriented)

This recipe describes a repeatable workflow for adding a new TPU kernel using
`jax.experimental.pallas` in a way that is:

- **safe** (numerics and gradients checked),
- **fast** (microbench + profiling evidence),
- **reviewable** (clear baseline + clear diff),
- **agent-friendly** (explicit checkpoints; no silent guessing).

This is modeled after our other “agent workflows”, e.g. `docs/recipes/add_dataset.md`, but kernels are even more
snowflake-y: expect iteration.

## What you produce

For a new kernel `K`, you should produce:
- A **vanilla JAX reference implementation** (readable, correct, stable API). You may be given torch code or pseudocode to get started.
- A **Pallas kernel implementation** + wrapper function with the same API.
- A **test harness** that checks:
  - values match reference (within tolerance),
  - gradients match reference on small shapes,
  - the kernel is jittable and works on representative shapes/dtypes,
- A **perf harness** that compares speeds for configurable shapes and dtypes as appropriate. Should also be suitable for autotuning.
- **Autotuned block sizes** for a reasonable range of sizes/parameters, to be specified in request.
- A **short report** (in the PR description) summarizing:
  - the tested shape/dtype grid,
  - perf results,
  - known limitations and follow-ups.

## Workflow

## Recommended module layout (Tokamax-style)

Tokamax uses a pattern that works well for kernel development:
- a single public API entrypoint (`api.py`) that dispatches to implementations
- a default “best available” backend order (e.g. TPU Pallas, then XLA)
- optional imports for accelerated backends so CPU-only users can still import modules

For Levanter kernels, we recommend the same pattern:
- `reference.py`: readable vanilla JAX oracle
- `xla.py`: default implementation (often identical to reference)
- `pallas_tpu.py`: TPU/Pallas implementation (optional import)
- `pallas_gpu.py`: optional GPU/Pallas implementation (if applicable)
- `api.py`: stable user-facing function with `implementation=` override and fallback order

See `lib/levanter/src/levanter/kernels/pallas/template_kernel.py` for a minimal example of the dispatch pattern.

## Batching convention (recommended)

To avoid juggling vmap-aware vs non-vmap-aware versions, prefer this simple convention:

- Implement **one** “true” kernel for the **batched** case.
- The public API accepts either batched inputs or a single unbatched example and normalizes by adding/removing a
  trivial leading batch dimension.

This makes it easy to support:
- `op(x)` for a single example (debug/unit tests),
- `op(x_batched)` for real training,
- and it keeps the kernel implementation focused on one shape regime.

If you later need `vmap(op)` semantics, treat the vmap axis as part of the batch by reshaping leading dims into a
single batch dimension, then reshaping back.

## Block size config (recommended)

Expose tile sizes via a small dataclass in the public API, similar to
`jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel.BlockSizes`:

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
- For TPU Pallas, block sizes are typically required to be multiples of 128 (DMA alignment). Validate in the Pallas
  backend, not in the generic API, so XLA/reference paths can still run.
- If Mosaic reports a layout mismatch for a batched integer operand (e.g. labels), align the batch block size to the
  XLA tile size (often 1024 on v5p for `s32[B]`), or raise a clear error before lowering.
- Keep a `block_size` convenience arg if it already exists in a legacy API; map it to `block_sizes.v_block_size` and
  raise if both are set inconsistently.

## Flattening rule for token losses

For token-level losses that operate on `[B, H]` and `[H, V]`:
- Flatten all **non-Contract** axes of `pred_embeddings` into a single batch dimension `B`.
- Flatten `target_y` to match this same `B`.
- Run the kernel on `(x: [B, H], labels: [B], w: [H, V])`.
- Reshape the per-example outputs back to the original axes before applying masks/reductions.

### 1) Start from a reference

Pick one:
- existing implementation in the repo (best),
- pseudocode,
- PyTorch reference,
- Optax/JAX baseline.

For the first target, we want **fused softmax cross entropy** with:
- optional **z-loss** (logsumexp penalty),
- optional **unreduced** output (per-token/per-example loss).

### 2) Write the vanilla JAX baseline

Write a baseline in plain JAX that:
- makes shapes and dtypes explicit,
- is obviously correct,
- has a stable signature you want to keep.

Avoid cleverness here: the baseline is your oracle.

**Memory-efficient baseline (when the naive baseline is too big):**
If the naive baseline would materialize huge intermediates, write a streaming or
blockwise baseline that matches the math but avoids those allocations. This keeps
correctness/perf checks feasible on realistic sizes.

**Type annotations:**
Use `jaxtyping` for public kernel APIs and reference implementations to make
shape/dtype expectations explicit (e.g. `Float[Array, "B H"]`).

### 3) Build a correctness harness (value + grad)

At minimum:
- value match vs baseline on a grid of shapes/dtypes,
- grad match on small shapes (finite difference or `jax.grad` vs baseline),
- **backend numerics**: validate numerics on CPU and on accelerator backends (TPU/GPU) as applicable.

When comparing outputs, prefer **pointwise deviation** metrics (e.g. max/mean
absolute diff) in addition to `allclose` so you can quickly spot outliers.

Keep the harness small, deterministic, and fast enough to run locally.
For long-lived kernels, turn the numerics script into a proper pytest in
`lib/levanter/tests/kernels/` that compares the **default implementation** to the
reference baseline. Use small shapes on CPU, and on TPU/GPU pick shapes aligned
with your block sizes so the fast path runs.

### 4) Implement the Pallas kernel + wrapper

Implement the kernel and wrap it behind the same API as the baseline.

Guidelines:
- keep the wrapper pure and explicit (no hidden global state),
- keep the “edit surface” obvious (tile sizes, block sizes),
- make failure modes actionable (shape constraints, alignment, etc.).

**Fallback semantics (recommended):**
- If a specific implementation is selected (e.g. `implementation="pallas_tpu"`), **error** on unsupported shapes or
  non-TPU backends.
- If the default implementation order is used, **warn and fallback** to XLA/reference.

**Z-loss/logsumexp penalty:**
- Have the kernel return both per-example loss and `logsumexp`.
- Apply the penalty outside the kernel: `loss + logsumexp_weight * logsumexp**2`.
- In custom VJP, propagate both `dloss` and `dlogsumexp` so gradients remain correct.

**TPU matmul precision gotcha:**
Some Mosaic TPU matmul paths require 32-bit accumulation. If you see
`Expected matmul acc to be 32-bit`, set `preferred_element_type=jnp.float32` in
`lax.dot_general` or set `jax.config.update("jax_default_matmul_precision", "highest")`
in your benchmark script. Prefer the explicit `preferred_element_type` in kernels.

### 5) Add a speed microbench + profiling hook

Minimum:
- microbench step time vs baseline on a representative shape,
- (optional but encouraged) a profiling run that produces an xprof artifact so agents can reason about hotspots.

#### Running on TPU (Marin Ray infra)

For TPU kernels you generally want to run the microbench/profile on a real TPU VM via our Ray runner.
Use `--tpu` to pick the TPU generation/size and pass environment variables with `-e` (these get injected into the Ray
job runtime environment).

```sh
RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \
  --cluster marin-us-central2-staging \
  --tpu v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT "marin-kernels" \
  -- python XXX
```

Replace:
- `v5p-8` with the TPU type you care about (common options: `v4-8`, `v5e-8`, `v5p-8`, `v6e-8`, etc.)
- `XXX` with your benchmark entrypoint (ideally a small script that JIT-compiles once, then runs a fixed number of
  steps and prints/records timing).

The cluster you use will vary depending on TPU generation/availability; most kernel work should start with a single
node (e.g. `*-8`) unless you’re explicitly targeting multi-slice behavior.

##### Listing available TPU clusters

If you’re not sure which cluster has which TPU types, use the cluster CLI:

```sh
uv run ./scripts/ray/cluster.py list-configs
uv run ./scripts/ray/cluster.py --cluster <cluster-name> list-workers
```

Use `--cluster` to target a specific region (e.g. `marin-us-central1`) and inspect
which TPU types are currently attached.

##### Running jobs and tailing logs (unattended-friendly)

For long runs, submit with `--no_wait`, then poll status and stream logs:

```sh
RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \
  --cluster marin-us-central1 \
  --tpu v5p-8 \
  --no_wait \
  -- python path/to/bench.py

uv run ./scripts/ray/cluster.py --cluster marin-us-central1 list-jobs > /tmp/ray_jobs.json
uv run ./scripts/ray/cluster.py --cluster marin-us-central1 job-logs -n 400 <job_id>
uv run ./scripts/ray/cluster.py --cluster marin-us-central1 wait-job <job_id>
```

You can parse `/tmp/ray_jobs.json` to find the matching `submission_id` and check
`status` (`PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`) for automation.

##### Alternative: dev_tpu helper

If you prefer a simpler wrapper, `docs/dev-guide/dev_tpu.md` and
`scripts/ray/dev_tpu.py` provide a more guided TPU workflow. Use whichever fits
your setup best; both paths are compatible with the kernel recipe.

Quick start on the staging cluster we commonly use for kernel work:

```sh
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central2-staging.yaml allocate
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central2-staging.yaml execute -- python path/to/bench.py
```

##### Profiling with Levanter (recommended)

Whenever possible, use Levanter's built-in profiler wiring so profiles are captured consistently and uploaded via
Levanter trackers (e.g. Weights & Biases):

- See `lib/levanter/docs/Performance-Guide.md` for details.
- Levanter uses JAX profiling and uploads a `jax_profile` artifact to W&B when profiling is enabled.
- If your benchmark is implemented as (or inside) a Levanter training loop, prefer flags like:
  - `--trainer.profiler true`
  - `--trainer.profiler_start_step 5`
  - `--trainer.profiler_num_steps 50`
  - `--trainer.profiler_perfetto_link false` (enable if you want a Perfetto URL; see the guide)

If you’re writing a standalone microbench script, prefer invoking the profiler directly with
`levanter.callbacks.profile_ctx` (rather than trying to plumb trainer flags through a non-trainer script). See
`lib/levanter/src/levanter/main/sample_lm.py` for an example of wrapping a “steady-state” region (skipping the compile
round) with `profile_ctx(...)`, which then logs a `jax_profile` artifact via the active tracker.
You will need to set up wandb logging by initializing a Levanter [levanter.tracker.wandb.WandbTracker][] via
[levanter.tracker.wandb.WandbConfig][].

##### Measuring performance sanely

Always report at least:
- time-to-first-step (includes compilation) vs steady-state step time (after warmup)
- the exact TPU type and shape/dtype grid tested

##### Tokamax comparison (optional)

If a similar Tokamax op exists, it can be a useful performance reference.
Common gotchas:
- Tokamax uses `absl.flags`; you must parse flags before accessing any
  Tokamax modules (`absl_flags.FLAGS([argv0])`).
- Tokamax Mosaic kernels may OOM VMEM at large shapes; reduce tile sizes or
  use smaller shapes for a quick comparison.

GPU kernels are typically runnable locally (and can still follow the same “baseline vs fast kernel” harness pattern).

## Autotuning (v0)

Most Pallas kernels have “knobs” (tile sizes, block sizes, pipeline stages, etc.). The best values depend on:
- TPU generation (v4/v5e/v5p/v6e),
- dtypes,
- shapes (batch/hidden/vocab),
- sharding/layout assumptions.

For v0, keep autotuning deliberately simple and explicit, with a concrete deliverable:

- a checked-in **Python “tuned table” module** that records the best-performing tile/block sizes per
  `(hardware, dtype, shape bucket)` and provides a reasonable default fallback.

This mirrors how some JAX Pallas TPU ops ship tuned defaults, e.g.
`jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes`.

1) Define a small **configuration space** you’re willing to search (a list of candidate block/tile configs).
2) Define a small set of **user-defined “architectures” / shape regimes** you care about (e.g. common hidden sizes,
   vocab sizes, batch sizes).
3) Run a **repeatable benchmark** for each (architecture, config) pair and record:
   - compile time (optional),
   - steady-state step time / tokens/sec,
   - any correctness/NaN failures.
4) Emit a **raw results table** (CSV/JSON) keyed by:
   `(device_kind, tpu_type, dtype, shape_bucket, config_id) -> metrics` and store it as an artifact.

### Tuned block-size tables

Even a tiny, uninteresting tuned table is still useful as an exemplar. Prefer
storing a small `tuned_block_sizes.py` module that:
- defines shape buckets and tuned entries,
- provides a single `infer_block_sizes(...)` helper,
- falls back to `BlockSizes.get_default()` if no match is found.
5) Derive a **best-config table** and write it to a Python module in the codebase (the tuned table):
   - mapping from `(tpu_type, dtype, shape_bucket)` to the best config (tile sizes etc.)
   - **shape buckets are important**: do not key on every exact shape. Pick a small number of representative buckets
     (e.g. ranges of `B/H/V` or a small set of canonical architectures) so the table stays stable and reviewable.
   - if the kernel depends on additional invariants (e.g. sharding/layout assumptions), include those in the key or
     enforce them before lookup.
   - a small helper that selects the best known config (or falls back to a default)
6) Repeat:
   - pick a new architecture/shape bucket (or a new TPU generation)
   - run the grid
   - append/merge best configs into the tuned table
7) Pick a **reasonable default** strategy:
   - global default (works “ok” everywhere), and/or
   - per-architecture defaults (if performance is highly shape-dependent).

The raw results should be treated as an artifact (W&B artifact preferred). The tuned Python table is the
code-reviewed “source of truth” for runtime defaults.

The point is to make tuning reproducible and reviewable: “why did we pick tile size X?” should be answerable by looking
at the table.

### 6) Iterate using evidence

The iteration loop should look like:
`profile -> hypothesis -> change -> tests -> microbench -> profile`

Use `agent_driven_profiling` tooling as it matures.

## Starter template

Use the starter template under `lib/levanter/src/levanter/kernels/pallas/`:
- `template_kernel.py` for baseline + kernel scaffolding
- `tests/test_template_kernel.py` for the value/grad/speed harness pattern

See also the `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss` kernel for a more
complete/complex example.

## Definition of Done

- Correctness: value matches baseline within tolerance across the tested grid.
- Gradients: gradients match baseline on small shapes.
- Performance: measurable improvement on at least one realistic shape (or a clear explanation why not).
- Documentation: PR includes a short “what we tested / what improved / what’s next” summary.
- Block sizes: autotuned block sizes checked in for the requested shape/dtype grid and device types.

## Further Reading

### Pallas Docs

- [JAX Pallas Overview](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX Pallas TPU Docs](https://docs.jax.dev/en/latest/pallas/tpu/index.html)
- [JAX Pallas Mosaic GPU Docs](https://docs.jax.dev/en/latest/pallas/gpu/index.html)

### Tokamax

Generally a good idea to look at tokamax kernels for inspiration and patterns.
Assuming we're in a uv install, they should be at `.venv/lib/python3.11/site-packages/tokamax/_src/ops`

We take inspiration from Tokamax's public APIs. We prefer a bit less framework-y stuff overall and
favor a bit more copy-paste and explicitness for agents.

### JAX kernels

JAX has some built-in kernels that use Pallas under the hood; these can be
good references for patterns. `.venv/lib/python3.11/site-packages/jax/experimental/pallas/ops`


## Misc Tips

-  Try to add parallel dimension semantics to at least one axis (usually batch) for TPU kernels.
