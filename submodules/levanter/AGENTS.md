# Levanter LLM Agent Guidelines

This document summarizes important conventions for contributing code or documentation to the Levanter
repository. Follow these notes when implementing new features or fixing bugs.

## General Guidelines

* **Get better.** Whenever you discover something missing from these guidelines, or the requester
  suggests a better way to do something, please update this document. The goal is to make it easier for
  everyone to contribute and maintain the codebase. Generally speaking, you should add bullets or new sections.
  Be sure to do this when directed to. For example, if directed that you should never relax tolerances in
  floating point tests, add that to the list.
* **Playbooks.** Sometimes, there are repeatable tasks (e.g. porting models) for which we follow a standard set of steps.
  Please reference `.playbooks/` to see what playbooks are available, or see the list below. If you want to add a playbook
  write a markdown doc named e.g. `.playbooks/port-models.md` and add a pointer to it in the list below.
* **Make a plan before giving up.** When a request seems too large to complete in one go, create a markdown file in `.agents/projects/` that outlines a step-by-step plan. Break the work into agent-sized checklist items.
  If later asked to continue the task, read the file, complete an item, and mark it off. If you can't finish a step, split it into smaller tasks and update the checklist accordingly.

## Playbooks

* [Porting a Model to Levanter](.playbooks/port-models.md): A guide for porting model architectures to the Levanter ecosystem using Haliax and Equinox.


## Code Style

* **Python version**: the project targets Python >=3.10.
* **Formatting and Linting**: We use `ruff` via `pre-commit`.
* **Typing**: the code base uses `mypy` for static type checking. `mypy` is run by pre‑commit and the
  configuration is found in `pyproject.toml`.
* **Run `pre-commit run --all-files`** before committing. The CI workflows run the same checks.
* **Doc Strings**: All public functions, classes, and modules should have docstrings, unless
  their purpose is painfully obvious. Use
  [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for
  consistency.
* **Commenting**: Use comments to explain why something is done a certain way, especially if it is not
  immediately obvious. Avoid commenting on every line of code; focus on the intent and purpose of
  complex logic. Demarcating logical groups of code with comments is encouraged, unless it is better
  to refactor the code into smaller functions or classes.
* **Mkdocs**: We use [Mkdocs](https://www.mkdocs.org/) for documentation. The main documentation is in
  the `docs` directory. Use Markdown for writing docs, and follow the existing structure. When linking to
  symbols, prefer using mkdocs-style links (e.g. With a custom title: `[full.path.object2][]` or
  `[Object 1][full.path.object1]`)
* **Documentation**: When adding new features, ensure that the documentation is updated accordingly.
  This includes updating the Mkdocs files and any relevant docstrings. If you add a new module or
  significant functionality, consider adding a dedicated section in the documentation.

## Testing

* Tests are executed with `pytest`. The default workflow runs
  `uv run pytest tests -m "not entry and not slow and not ray"`.
* In general, never relax tolerances in floating point tests unless specifically discussed with the
  team. Use `assert_allclose` with appropriate tolerances for numerical comparisons. We typically use
  1e-4 for more complex modules, and 1e-5 for simpler ones.
* Tests should be reasonably fast. Mark long-running tests with @pytest.mark.slow so they are excluded from the default suite.
* Always mark tests that depend on pytorch with `@skip_if_no_torch` to ensure they are skipped
  when PyTorch is not available. This is particularly important for tests that require PyTorch-specific
  functionality.
* **CI Best Practice**: Use `astral-sh/setup-uv` to install `uv` in workflows and run `uv python install`
  before installing dependencies with `uv sync` or `uv pip`. This ensures the expected Python
  version is available during testing.


## Design Preferences

* **Named tensors**: Levanter relies heavily on the [Haliax](https://github.com/stanford-crfm/haliax)
  library. Arrays are usually represented by `NamedArray` with explicit `Axis` objects. Prefer writing
  operations over named axes rather than positional dimensions.
* **Generic code**: many utilities are written with Python generics and dataclasses. Where possible,
  write reusable functions or classes that operate over TypeVars instead of hard coding concrete types.
* **Configurations**: configuration files are dataclasses loaded via `draccus`. Keep configs
  declarative and typed.
* **Datasets**: datasets are represented as `AsyncDataset` or `SyncDataset` in `levanter.data.dataset`.
  When creating new data pipelines, prefer asynchronous versions and support slicing, shuffling and
  mapping operations. In general, Async is preferred over Sync.
* **Logging and tracking**: metrics and performance stats are logged via tracker hooks (e.g. WandB or
  TensorBoard). Use the existing callback/hook framework instead of ad-hoc logging.
* **Reproducibility**: Levanter aims for deterministic training where possible. Avoid sources of
  nondeterminism unless explicitly required.
* Prefer Stacked with fold or scan over writing custom loops, for better compile times and gradient checkpointing support

## JIT Safety

* Avoid data-dependent Python control flow inside jitted code.
* Do not rely on dynamic shapes.
* Do not use dynamic lengths when indexing.
* Use `debug.print` if you need to inspect values.
* Use jit-safe versions of `jnp.where`, `hax.where`, or similar operations where the number of returns depends on data.

Any method inside an `equinox.Module`, any function decorated with `jax.jit` or one of its variants (e.g. `eqx.filter_jit` or `jax.named_jit`), and any helpers they call must follow these jit-safety rules.

## Additional Tips

* Use `NamedArray` and `Axis` for model parameters and activations.
* We use Equinox and Haliax, not Flax and not Haiku, for neural network layers and models.
* Prefer functional style JAX code with explicit PRNG keys.
* Avoid hard coding dataset paths; accept them via configuration.
* When extending the library, maintain compatibility with both GPU and TPU backends.
