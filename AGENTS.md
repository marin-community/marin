# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject section or its dedicated docs.
- Per-project sections below contain only project-specific conventions. Detailed operational docs live in the project's own `docs/` directory.

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- For PR descriptions, testing, specifications, and review workflow, follow [pull-request.md](docs/recipes/pull-request.md).
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- For long-running benchmark/research threads, follow [agent_research.md](docs/recipes/agent_research.md).
- For canary/daily ferry proposal, launch, and monitoring workflow, follow [ferries.md](docs/recipes/ferries.md).
- When making significant changes to Grug/Grugformer, follow [change_grug.md](docs/recipes/change_grug.md).
- For profiling ingestion and agent-driven optimization workflows, follow [agent_profiling.md](docs/recipes/agent_profiling.md).
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `./infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Agent-Generated GitHub Activity

- When an agent creates a PR or an issue using the user's auth token, it must add the `agent-generated` label.
- When an agent comments on a PR or issue using the user's auth token, the comment must begin with an `🤖` emoji unless the exact comment text was explicitly approved by the user. If it cannot be at the very beginning for formatting or workflow reasons, it should come as soon as possible.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Avoid `TYPE_CHECKING`; use real imports. If you hit a cycle, prefer refactoring or use a `Protocol` at the boundary.
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Separation of responsibilities: when a change introduces a new subsystem, encapsulate lifecycle/configuration in a dedicated module and have callers depend on the interface.
- Disprefer internal mutation of function arguments, especially config dataclasses; prefer returning a modified copy (e.g., via `dataclasses.replace`) so call sites remain predictable and side effects are explicit.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Document public APIs with concise Google-style docstrings.
- Prefer small, concrete helpers over abstraction that adds indirection without reuse.
- When defaults depend on environment/resource type, resolve them once and fail fast on unknown/ambiguous inputs rather than silently guessing.
- Keep environment detection logic minimal and explicit; prefer single strong signals over sprawling defensive checks.
- Prefer logging over `print` statements. `print` is fine for debugging and scripts.
- Prefer generic code parameterized with `TypeVar` over hard-coding concrete types where reuse is natural.
- Aim for deterministic behavior; avoid nondeterminism unless explicitly needed.

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.
- Write docs for readers who do not have conversational context: include enough problem framing, assumptions, commands, and results that the document stands on its own.

### Deprecation

**NO BACKWARD COMPATIBILITY**: Do NOT add deprecation warnings, fallback paths, or compatibility shims. Update all call sites instead. Only add backward compatibility if the user explicitly requests it.

## Comments

You write detailed comments when appropriate to describe code behavior as a
whole, e.g. at the module or class level, or when describing some subtle
behavior.

You don't generate comments that merely restate the code, e.g.

<bad>
     # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
</bad>

<good>
# We have found that each instance of a FlightServer can provide approximately 1GB/s
# of throughput. As our typical VMs run with 200Gbps NICs, running 16 parallel servers
# should be sufficient to saturate the network.
</good>

## Planning

- When planning, you produce detailed plans including code snippets.
- You ask questions up front when building a plan instead of guessing.
- When a request feels too large for one pass, capture a plan (for example in `.agents/projects/` when the subproject provides one) before pausing.

## Testing

- Always fix tests if you broke them.
- Do not fix tests by relaxing tolerances or hacking around them.
- Avoid "tautological" tests that merely restate implementation logic as asserts; prefer tests that validate externally-observable behavior, integration points, or realistic failure modes.
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject sections below for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.
- Do not relax numerical tolerances without prior agreement from a human. Prefer `assert_allclose` with 1e-4 for complex modules and 1e-5 for simpler ones.
- Always protect PyTorch-dependent tests with `@skip_if_no_torch`.

PREFER:

- Integration style tests which exercise behavior and test the output

DO NOT:

- Create tests which validate obvious features: if a type exists, a constant has a value, etc.

## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`

---

## Levanter

Levanter lives in `src/levanter`. For detailed docs see `docs/levanter/`.

### Testing

* Run `uv run pytest tests -m "not entry and not slow and not ray"` for the default suite.
* Batch sizes should generally be a low multiple (e.g. 1 or 2) of `len(jax.devices())` to ensure multi-device correctness.
* Mark long-running tests with `@pytest.mark.slow`.

### TPU VMEM Flags

* For TPU Pallas benchmark/tuning runs, set `LIBTPU_INIT_ARGS` by TPU generation:
  * `v5p`/`v5e`: `--xla_tpu_scoped_vmem_limit_kib=50000`
  * `v6e`: `--xla_tpu_scoped_vmem_limit_kib=98304`
* Do not set a special scoped VMEM limit flag on `v4` unless a user explicitly asks for it.

### Design Preferences

* **Named tensors:** Represent arrays with `NamedArray` and explicit `Axis` objects via Haliax. Operate over named axes whenever possible.
* **Configurations:** Dataclasses loaded via `draccus`. Keep them declarative and typed.
* **Datasets:** Use `AsyncDataset` or `SyncDataset` in `levanter.data.dataset`. Prefer `AsyncDataset` unless there is a concrete reason not to.
* **Logging and tracking:** Use the existing tracker hooks (W&B, TensorBoard) for metrics instead of ad-hoc logging.
* Prefer `Stacked` with `fold` or `scan` over hand-written loops to improve compile times and gradient-checkpointing behavior.
* Use Equinox and Haliax -- avoid Flax and Haiku layers in new code.
* Prefer functional-style JAX code with explicit PRNG keys.
* Maintain compatibility with both GPU and TPU backends.

### JIT Safety

* Avoid data-dependent Python control flow inside jitted code.
* Do not rely on dynamic shapes or dynamic lengths when indexing.
* Use `debug.print` if you need to inspect values; choose `jnp.where` or `hax.where` when branching on data.
* Do not call `jax.default_backend()` or `jax.devices()` at module import time; resolve lazily inside runtime functions.

Any method inside an `equinox.Module`, any function decorated with `jax.jit` (or variants like `eqx.filter_jit`, `jax.named_jit`), and any helpers they call must follow these rules.

### Recipes

* [Porting a Model to Levanter](docs/recipes/port-models.md)

---

## Marin

Marin pipeline code lives in `src/marin`. For detailed docs see `docs/marin/`.

### Data Access

* Do not special-case Google Cloud Storage unless absolutely necessary. Use `fsspec.open` and other fsspec helpers so code stays filesystem-agnostic.
* Do not copy data artifacts (`.json`, `.parquet`, etc.) to the local filesystem -- stream them through fsspec instead.
* Avoid hard-coding GCS paths like `gs://marin-us-central2/foo/bar`. Prefer referencing pipeline steps; if you must inject a literal path, wrap it with `InputName.hard_coded` and call out the follow-up risk.
* NEVER EVER EVER load GCS files from across region if they are more than a few MB.

### Testing

* Run `uv run --package marin pytest` targeting any suites that exercise the package before submitting changes.

---

## Iris

Iris lives in `src/iris`. For operational docs see `docs/iris/`, `OPS.md`, and `TESTING.md` in the Iris directory.

### API Style

* Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
* After changing `.proto` files, regenerate via `scripts/generate_protos.py`.
* Any functionality exposed by dashboards must also be available via RPC. Dashboards are a thin UI over the RPC API, not a second implementation path.

### Concurrency Model

* Platform operations (`terminate`, `create_slice`, etc.) shell out via `subprocess.run` and are thread-safe.
* For concurrent independent platform operations, use `concurrent.futures.ThreadPoolExecutor` (not asyncio) and apply hard timeouts.

### Time Utilities

Use `iris.time_utils` for all time-related operations instead of raw `datetime` or `time`. Key types: `Timestamp`, `Duration`, `Deadline`, `Timer`, `ExponentialBackoff`.

### Planning

Prefer spiral plans over linear plans: each stage should be independently testable (proto -> server stub -> client wiring -> end-to-end test), then iterate.

### Operational Details

Deployment topology, CoreWeave disk layout, light worker mode, multi-region image push/pull, and zone validation are documented in the Iris `docs/` directory (especially `docs/coreweave.md` and `docs/image-push.md`). Consult those when working on infrastructure changes.

### Testing

See `TESTING.md` in the Iris directory for the testing policy and commands.

---

## Haliax

Haliax lives in `src/haliax`. For detailed docs see `docs/haliax/` and [docs/primer.md](docs/primer.md) for an agent-oriented overview.

### Playbooks

* [Adding tensor typing annotations](.playbooks/add-types.md)
* [Wrapping standard JAX functions](.playbooks/wrap-non-named.md) so they operate on `NamedArray`

### Library Conventions

* Haliax revolves around `NamedArray` and named shapes via `Axis` objects or shape dicts. Prefer APIs that accept axes or axis names rather than hard-coding positional dimensions. Use `AxisSpec` and `AxisSelection` where possible.
* Utilities should work with arbitrary axis names; avoid relying on fixed axis orders.
* Use `haliax.nn` or Equinox when building neural network layers.
* Type annotations can use named shapes shorthand from `haliax.haxtyping`: e.g. `ht.f32[NamedArray, "batch"]`.
* Prefer `Stacked` with `fold` or `scan` over hand-written loops.
* Prefer frozen dataclasses over dictionaries for configuration.

### Type Checking

Haliax uses `mypy` for static type checking (configured in `pyproject.toml`, run via `infra/pre-commit.py`).

### Testing

* Default: `XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run pytest tests`
* When wrapping a new JAX function, add a reference to it in `docs/api.md`.
