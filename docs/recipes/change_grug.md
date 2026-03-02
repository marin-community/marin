# Recipe: Changing Grug (Template-First)

Grug is intentionally template-first: the canonical edit surface lives in `experiments/grug/base/`, not in a shared `levanter.grug` trainer stack.

This recipe describes the workflow for:

1. trying a change in an experiment copy, and
2. upstreaming it into the base template when it proves out.

## Source Of Truth

- **Canonical template:** `experiments/grug/base/`
  - `model.py`
  - `train.py`
  - `launch.py`
- **Variants:** `experiments/grug/<variant>/`
  - copy from `base` and modify locally (for example MoE).
- **One-off speedruns:** `experiments/speedrun/...`
  - useful for exploration, not canonical.

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

Update `docs/reports/grug-archive.md` with:

- path
- origin (`base`, `moe`, or another source variant)
- commit SHA (when known)
- purpose
- status (`active`, `superseded`, `deleted`)
- diff link (prefer the CI-posted PR comment link; fallback to local report path)

For PRs that add a new `experiments/grug/<variant>/`, CI posts a visual diff
comment automatically. Copy that link into the archive entry.

If you need a local fallback, generate the diff report manually:

```bash
uv run python scripts/grug_dir_diff.py \
  experiments/grug/base \
  experiments/grug/<variant> \
  --out /tmp/grug-diff
```

Then include a link to the report in the archive entry so reviewers can inspect
template-copy changes quickly.

### 4) Upstream to base if it wins

Port the successful change back into:

- `experiments/grug/base/model.py`
- `experiments/grug/base/train.py`
- `experiments/grug/base/launch.py`

Keep it grug-style:

- plain JAX arrays and explicit sharding
- Equinox modules with `init` + `__call__`
- minimal config knobs
- keep legibility first; if a block gets hard to read, introduce a small local helper instead of adding framework indirection

### 5) Delete stale paths

After upstreaming:

- delete superseded experiment code,
- keep only the archive trail in `docs/reports/grug-archive.md`.

### 6) Validate

Run the relevant checks:

```bash
uv run python infra/pre-commit.py --all-files
uv run pytest tests/test_grug_base_template.py
```

Add any additional focused tests needed for behavior changes.

This workflow is inspired by modded-nanogpt: iterate quickly in copy-paste experiments, then upstream only what stays simple and useful.
