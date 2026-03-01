# Grug Project Principles

## Goal

Keep Grug explicit, copyable, and template-first.
Inspired by grugbrain.dev and NanoGPT: keep the code simple enough to read in one pass.

The canonical implementation lives in `experiments/grug/base/`:
- `model.py`
- `train.py`
- `launch.py`

New variants (for example MoE) should start as copies of `experiments/grug/base/` and be edited in place.

## Core Rules

- Use `equinox.Module` and explicit `init/__call__` patterns.
- Do not use `eqx.nn` modules in Grug core; build layers from raw JAX ops and explicit parameter arrays.
- Prefer plain `jax.Array` + explicit `PartitionSpec` sharding.
- Make mesh/sharding intent visible in types where practical (for example jaxtyping shape annotations plus explicit axis-resource docs at boundaries).
- Prefer `jnp.einsum` for contractions and weighted reductions.
- Avoid broadcast-noise patterns with inserted singleton dims when `einsum` is clearer.
- Keep configs small; prefer copy/paste in variants over growing framework abstractions.
- Keep metric parity with classic train logs where practical (`train/loss`, `throughput/*`, `eval/*`, `mixture/*`, watch stats).

## Organization

- Template code belongs in `experiments/grug/<variant>/`.
- Shared library code should only hold general-purpose helpers, not grug-specific framework layers.
- Remove stale experiment paths after upstreaming, and keep the trail in `docs/reports/grug-archive.md`.

## Testing

Minimum checks for grug template changes:
- `uv run pytest tests/test_grug_base_template.py`
- targeted additional tests for changed behavior
- `uv run python infra/pre-commit.py --all-files`

Testing copy-paste variants:
- For each new variant under `experiments/grug/<variant>/`, add a variant smoke test that mirrors `tests/test_grug_base_template.py` and exercises import + one-step train-step construction.
- Keep assertions on externally visible behavior (metric keys, step progression, eval wiring), not internal implementation details.

## Migration Notes

Legacy Grugformer-era trainer/model/wrapper paths were retired in favor of template-first code. Avoid reintroducing them.
