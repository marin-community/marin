# Levanter Agent Guidelines

Levanter is incorporated into `lib/levanter`. Start with the shared instructions in `../../AGENTS.md`; the notes below cover Levanter-specific conventions.

## Workflow Notes

* Build on the shared practices in `../../AGENTS.md`; this file only captures Levanter-specific expectations.
* Capture multi-session Levanter work in `.agents/projects/` inside this directory so partial progress is easy to resume.
* Keep Levanter-specific recipes in `docs/recipes/` and expand them when you uncover repeatable tasks.

## Recipes

* [Porting a Model to Levanter](docs/recipes/port-models.md): A guide for porting model architectures to the Levanter ecosystem using Haliax and Equinox.

## Documentation

* Levanter documentation lives in `lib/levanter/docs/`. Update `mkdocs.yml` when adding or reorganizing sections.
* Use mkdocs-style symbol links (for example `[full.path.object][]`) so cross-references work in the generated site.
* Add a short intent paragraph to new recipes and link them from relevant guides or README sections.

## Testing

* Run `uv run pytest tests -m "not entry and not slow and not ray"` to execute the default Levanter suite.
* Do not relax numerical tolerances without prior agreement from a human. Prefer `assert_allclose` with 1e-4 for complex modules and 1e-5 for simpler ones unless a specific case demands otherwise.
* Batch sizes should generally be a low multiple (e.g. 1 or 2) of the number of `len(jax.devices())` to ensure multi-device correctness.
* Mark long-running tests with `@pytest.mark.slow` so they stay out of the default suite.
* Always protect PyTorch-dependent tests with `@skip_if_no_torch` so they skip cleanly when PyTorch is unavailable.
* **CI tip:** use `astral-sh/setup-uv` to install `uv`, run `uv python install`, and then run `uv sync`/`uv pip` to guarantee the expected Python version in workflows.

## Design Preferences

* **Named tensors:** Levanter relies heavily on [Haliax](https://github.com/stanford-crfm/haliax). Represent arrays with `NamedArray` and explicit `Axis` objects, and operate over named axes whenever possible.
* **Generic code:** Many utilities use Python generics and dataclasses. Prefer reusable functions/classes parameterized with `TypeVar` instead of hard-coding concrete types.
* **Configurations:** Configuration files are dataclasses loaded via `draccus`. Keep them declarative and typed.
* **Datasets:** Represent datasets as `AsyncDataset` or `SyncDataset` in `levanter.data.dataset`. Favor asynchronous datasets and ensure they support slicing, shuffling, and mapping. Prefer `AsyncDataset` unless there is a concrete reason not to.
* **Logging and tracking:** Use the existing tracker hooks (for example Weights & Biases or TensorBoard) for metrics instead of ad-hoc logging.
* **Reproducibility:** Aim for deterministic training; avoid nondeterminism unless explicitly needed.
* Prefer `Stacked` with `fold` or `scan` over hand-written loops to improve compile times and gradient-checkpointing behavior.

## JIT Safety

* Avoid data-dependent Python control flow inside jitted code.
* Do not rely on dynamic shapes or dynamic lengths when indexing.
* Use `debug.print` if you need to inspect values.
* Choose jit-safe variants like `jnp.where` or `hax.where` when branching on data.

Any method inside an `equinox.Module`, any function decorated with `jax.jit` or one of its variants (for example `eqx.filter_jit` or `jax.named_jit`), and any helpers they call must follow these jit-safety rules.

## Additional Tips

* Use `NamedArray` and `Axis` for model parameters and activations.
* Levanter uses Equinox and Haliax—avoid Flax and Haiku layers in new code.
* Prefer functional-style JAX code with explicit PRNG keys.
* Avoid hard-coding dataset paths; surface them through configuration.
* Maintain compatibility with both GPU and TPU backends when extending the library.
