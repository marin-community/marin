# Collapse workspace to single package

**Status**: Implemented (Option B). See commit `59d30e6d0`.

**Context**: Russell (infra lead) wants to explore collapsing the 7-member uv workspace back into a single `marin` package. The workspace/members structure hasn't been paying for itself.

## Current state

7 workspace members under `lib/`:

```
Leaf nodes (no workspace deps):
  iris      — distributed task execution / cluster mgmt
  haliax    — named tensors for JAX
  rigging   — filesystem / logging utilities

Mid-level:
  fray      → iris, rigging     — distributed execution framework
  zephyr    → iris, rigging, fray  — dataset library

High-level:
  levanter  → haliax, fray, rigging, zephyr  — training framework
  marin     → everything above               — orchestration / pipelines
```

Plus `experiments/` (imports marin + levanter heavily), `data_browser/` (standalone, no workspace deps).

### What the workspace buys today

- **Selective install**: `uv sync --package iris` avoids pulling JAX/levanter deps. Useful for lightweight cluster-management-only environments.
- **CI scoping**: each member has a `dorny/paths-filter` workflow that skips tests on unrelated changes.
- **Layering enforcement**: dependency direction is explicit in each `pyproject.toml`.

### What it costs

- 8 `pyproject.toml` files to maintain (root + 7 members).
- 2MB `uv.lock` with complex override-dependencies for JAX compat.
- Cross-member optional deps, extras, and dependency groups are hard to reason about.
- Every CI workflow needs `--package X` / `--frozen` / `--extra` incantations.
- New contributors face a learning curve ("which package do I add this to?").
- `pyrefly` config has to enumerate all `lib/*/src` search paths.

## What "collapse" means

Merge all source under one top-level `src/marin/` (or keep subpackage identity as `marin.haliax`, `marin.levanter`, etc.) with a single `pyproject.toml`.

### Option A: Flat merge — everything becomes `marin.*`

```
src/
  marin/
    __init__.py
    core/         ← current lib/marin/src/marin/
    haliax/       ← current lib/haliax/src/haliax/ (renamed to marin.haliax)
    levanter/     ← current lib/levanter/src/levanter/
    iris/          ...
    fray/
    rigging/
    zephyr/
```

**Pros**: one package, one namespace, simple.
**Cons**: ~3,800+ import statements need rewriting (`import haliax` → `from marin import haliax` or `import marin.haliax as haliax`). Breaks external consumers of `haliax` and `levanter` on PyPI. CLI entry points (`iris`, `fray`, `zephyr`) need renaming or shim scripts.

### Option B: Keep top-level package names, single pyproject.toml

```
src/
  iris/
  fray/
  haliax/
  levanter/
  marin/
  rigging/
  zephyr/
```

One `pyproject.toml`, one `uv.lock`, all packages installed together. No import rewrites needed.

**Pros**: zero import changes, preserves `import haliax` / `import levanter` for external users, CLI entry points unchanged.
**Cons**: no namespace scoping (7 top-level packages from one project feels unusual), can't pip-install just one sub-package anymore.

### Option C: Keep workspace, reduce members

Merge the small utilities (rigging, iris) into their primary consumers (fray, marin). Reduce from 7 members to 3-4:

```
lib/
  haliax/     ← unchanged (published to PyPI, external users)
  levanter/   ← absorbs zephyr? (levanter already depends on it)
  marin/      ← absorbs rigging, iris, fray
```

**Pros**: keeps selective-install for haliax/levanter (real external packages), reduces member count, less pyproject.toml maintenance.
**Cons**: still a workspace, just smaller. Half-measure.

## Recommendation: Option B

**Option B** gets 90% of the simplification benefit with zero import churn:

### What changes
1. **Delete** all 7 `lib/*/pyproject.toml` files.
2. **Move** each `lib/*/src/<pkg>/` → `src/<pkg>/`.
3. **Single root `pyproject.toml`**: merge all deps, extras, dependency groups, entry points.
4. **Single `uv.lock`** (already exists, but simpler without workspace resolution).
5. **CI workflows**: remove `--package` flags, simplify to `uv sync` / `uv run pytest`.
6. **pyrefly**: simplify search paths to just `src/`.
7. **AGENTS.md**: update dependency-direction docs (still valid as convention, just not enforced by pyproject.toml).
8. **`experiments/`**: move under `src/` or keep at top level (either works).

### What doesn't change
- Zero import rewrites (all `import haliax`, `from levanter import ...` etc. still work).
- CLI entry points (`iris`, `fray`, `zephyr`) still work.
- External PyPI consumers can still `pip install marin` and get everything.
- `data_browser/` remains standalone.

### Risks / concerns
- **Selective install lost**: can't `uv sync --package iris` for lightweight envs anymore. Mitigation: use extras (`marin[iris]`) or dependency groups to control what gets installed.
- **haliax/levanter PyPI**: these are currently published as separate packages. If external users `pip install haliax`, that breaks. Need to either:
  - Continue publishing them separately (build from `src/haliax/` etc.), or
  - Deprecate standalone packages, point users to `pip install marin`.
  - Or publish compatibility shim packages.
- **Layering enforcement**: currently pyproject.toml prevents `iris` from importing `levanter`. With one package, this becomes convention-only. Mitigation: a CI lint rule or import-linter config.
- **Test isolation**: currently each member has its own test group. Merge these into a single test matrix or keep as named groups.

### Migration steps

1. Create a migration branch.
2. `mkdir src/` at repo root.
3. `mv lib/*/src/*/ src/` (move each package's source).
4. Merge all `pyproject.toml` deps/extras/groups into root.
5. Move `[project.scripts]` entries to root.
6. Delete `lib/` (or keep as empty marker).
7. Update `uv.lock` (`uv lock`).
8. Update CI workflows (remove `--package`, update paths-filters).
9. Update `pyrefly` config.
10. Run full test suite.
11. Update docs (AGENTS.md, README, etc.).

### Estimated scope
- ~0 import changes (the big win).
- ~8 files deleted (member pyproject.tomls).
- ~1 file heavily edited (root pyproject.toml — merging all deps).
- ~10-15 CI workflows updated (remove `--package` flags, update filters).
- ~1-2 config files updated (pyrefly, pre-commit).

## Decisions

1. **PyPI story for haliax/levanter**: Team is fine abandoning separate PyPI packages. Everything ships as `marin`.
2. **Approach**: Option B (single pyproject.toml, preserve top-level package names, zero import rewrites).

## Open questions

1. **Does Russell want `experiments/` inside `src/` too?** Currently it's a hatch build target from root.
2. **Import linting**: worth adding `import-linter` to enforce the dependency DAG without workspace boundaries?
3. **Extras for selective install**: worth defining `marin[training]`, `marin[data]`, `marin[cluster]` etc. to replace per-package install?
