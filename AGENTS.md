# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject manual so the next agent stays aligned.
- Consult the subproject manuals when working in submodule trees:
  * `lib/levanter/AGENTS.md` for Levanter-specific conventions.
  * `lib/marin/AGENTS.md` for Marin-specific conventions
- When a recipe exists, follow it—the agent-friendly playbooks live in `docs/recipes/`. Some live in the individual `lib/*/docs` directories.

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `uv run python infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Disprefer internal mutation of function arguments, especially config dataclasses; prefer returning a modified copy (e.g., via `dataclasses.replace`) so call sites remain predictable and side effects are explicit.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Do not use `from future import ...` statements.
- Document public APIs with concise Google-style docstrings.

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.

### Deprecation

- Unless specifically requested, do not introduce deprecation or fallback paths—update all call sites instead.

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
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject guides for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.

PREFER:

- Integration style tests which exercise behavior and test the output

DO NOT:

- Create tests which validate obvious features: if a type exists, a constant has a value, etc.


## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`

## Ray Run Notes

- For Ray/TPU runs, prefer `TMPDIR=/tmp` and `RAY_TMPDIR=/tmp`.
- Avoid setting `UV_CACHE_DIR` in Ray launch commands unless you have a specific reason (use uv's default cache).
- For a brand new run, omit `--force-run-failed`; add it only when you explicitly want to rerun previously-failed steps.

### Ray Token Auth (Central1)

If you need to authenticate to the Ray dashboard (token auth):

```bash
make get_ray_auth_token
uv run scripts/ray/cluster.py --cluster us-central1 auth
```

Example (size sweep on Central1):

```bash
TMPDIR=/tmp RAY_TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars RAY_TMPDIR /tmp --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_v5p32 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python experiments/speedrun/olmoe_1b7b_size_speedrun_sweep.py --dataset nemotron_cc --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64
```

Example (OLMoE sizes sweep, composite mixture, eval during+after, 2 terminals, shared W&B group):

- Uses a deterministic W&B group based on the current git commit so you don't need to coordinate a timestamp across terminals.
- Run uniqueness comes from `--run-suffix`.
- Defaults to 4 sizes × 4 LR multipliers = 16 runs per command.

SwiGLU (regular experts), BS=64:

```bash
GROUP="olmoe_sizes_v5p32_b64_$(git rev-parse --short HEAD)" && SUFFIX="t$(date +%Y%m%d_%H%M%S)_$$" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_swiglu_b64 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_size_speedrun_sweep --dataset nemotron_dclm_fineweb_10b --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64 --eval-suite core --eval-suite-mode both --steps-per-task-eval 500 --wandb-group "$GROUP" --run-suffix "swiglu_${SUFFIX}" --sizes olmoe_1b7b
```

Bilinear experts, BS=64 (same group automatically):

```bash
GROUP="olmoe_sizes_v5p32_b64_$(git rev-parse --short HEAD)" && SUFFIX="t$(date +%Y%m%d_%H%M%S)_$$" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_bilinear_b64 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_size_speedrun_sweep --dataset nemotron_dclm_fineweb_10b --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64 --eval-suite core --eval-suite-mode both --steps-per-task-eval 500 --bilinear-mlp --wandb-group "$GROUP" --run-suffix "bilinear_${SUFFIX}" --sizes olmoe_1b7b
```

> This file will be expanded as agent workflows and best practices evolve.
