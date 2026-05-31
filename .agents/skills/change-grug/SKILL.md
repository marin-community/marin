---
name: change-grug
description: Modify or upstream a Grug/Grugformer experiment variant.
---

# Skill: Changing Grug (Template-First)

Grug is intentionally template-first: the canonical edit surface lives in `experiments/grug/base/`, not in a shared `levanter.grug` trainer stack.

This skill covers two steps: trying a change in an experiment copy, and upstreaming it into the base template when it proves out.

## Source Of Truth

- **Canonical template:** `experiments/grug/base/` — `model.py`, `train.py`, `launch.py`.
- **Variants:** `experiments/grug/<variant>/` — copy from `base` and modify locally (e.g. MoE).
- **One-off speedruns:** `experiments/speedrun/...` — useful for exploration, not canonical.
- **Reference branch for array-stacked grug variant wiring:** https://github.com/marin-community/marin/tree/codex/array-stacked-grug-variant-pointer — useful for perf-focused experiments, especially improving compile times and reducing peak HBM.

## Workflow

### 1) Pick one change bucket

Keep each pass scoped to one bucket:

- attention/masking
- block wiring/norm ordering
- MLP/activation
- loss kernel behavior
- optimizer/training loop behavior

### 2) Experiment in a copy

- Copy `experiments/grug/base` to a new variant directory.
- Keep edits local and explicit (copy/paste over abstraction).
- Avoid introducing reusable framework surface unless there's clear repeated use.

### 3) Record the experiment

Update `docs/reports/grug-archive.md` with: path, origin (`base`, `moe`, or another source variant), commit SHA (when known), purpose, status (`active`, `superseded`, `deleted`), and diff link (prefer the CI-posted PR comment link; fallback to local report path).

For PRs that add a new `experiments/grug/<variant>/`, CI posts a visual diff comment automatically — copy that link into the archive entry.

For a local fallback, generate the diff report manually and link the report in the archive entry:

```bash
uv run python scripts/grug_dir_diff.py \
  experiments/grug/base \
  experiments/grug/<variant> \
  --out /tmp/grug-diff
```

### 4) Upstream to base if it wins

Port the successful change back into `experiments/grug/base/model.py`, `train.py`, and `launch.py`. Keep it grug-style:

- plain JAX arrays and explicit sharding
- Equinox modules with `init` + `__call__`
- minimal config knobs
- legibility first; if a block gets hard to read, introduce a small local helper instead of framework indirection
- when HBM is tight, use `docs/references/hbm-optimization.md` before bespoke memory hacks
- when compile time or peak HBM is the bottleneck, evaluate an array-stacked variant first (see reference branch above)

### 5) Delete stale paths

After upstreaming, delete superseded experiment code; keep only the archive trail in `docs/reports/grug-archive.md`.

### 6) Validate

```bash
./infra/pre-commit.py --all-files
uv run pytest tests/test_grug_variant_contracts.py
```

Add focused tests for any behavior changes.

This workflow is inspired by modded-nanogpt: iterate quickly in copy-paste experiments, then upstream only what stays simple and useful.
