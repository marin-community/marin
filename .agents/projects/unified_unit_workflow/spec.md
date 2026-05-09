# Spec — unified_unit_workflow

The contracts implied by `design.md`. Reviewers should be able to read
this and answer "would I actually build this exact API?"

The schema is **convention with overrides**: a baseline that covers most
packages with no per-package config, plus a tiny `[tool.marin.tests]`
table for the genuinely different packages. Most workspace members will
have *no* table.

## 0. Terminology

- **Workspace member** — one directory under `lib/`. Marin has 8.
  Identified by its short directory name (`levanter`) and packaged as
  `marin-<dir>` (`marin-levanter`). The marin scope is special — its
  test root is the repo-top-level `tests/`, owned by the root
  `pyproject.toml`.
- **Module** — a Python module name resolved from a source file.
  `lib/levanter/src/levanter/store/cache.py` → `levanter.store.cache`.
- **Test file** — a file under `lib/<member>/tests/` (or top-level
  `tests/` for the marin scope) that pytest collects.
- **Leg** — one entry in the orchestrator's matrix; one `pytest`
  invocation against one workspace member's selected tests.
- **Selection** is per-test-file (which files pytest runs).
  **Dispatch** is per-workspace-member (one leg per member that has any
  selected test). **Traversal** is per-module (the analyzer's
  reverse-dep BFS walks dotted module names).

## 1. Workspace baseline (the convention)

Every test leg runs as if these defaults applied — the orchestrator does
not look at any `[tool.marin.tests]` for these:

| Concern | Default |
|---|---|
| Python | `3.12` (workspace-wide; pinned in `pyproject.toml` `[tool.marin.tests.workspace]`) |
| `uv sync` | `uv sync --frozen --package marin-<dir> --group test` (where `<dir>` is the `lib/<dir>` directory name; for the marin scope = `--package marin --group test`) |
| `uv run` | `uv run` with no extra flags |
| `pytest` markers | `not slow and not tpu and not tpu_ci` |
| `pytest` extra args | `["--durations=5", "-n", "auto", "--tb=short"]` |
| `pythonpath` | unset (don't override) |
| `env` | empty (no extras) |
| `setup_scripts` | none |
| `setup_node` | none |
| Working directory | repo root (always — no `cd lib/<pkg>`) |
| `timeout_minutes` | 15 |

Workspace baseline lives in the **root** `pyproject.toml`:

```toml
[tool.marin.tests.workspace]
python = "3.12"
markers = "not slow and not tpu and not tpu_ci"
pytest_args = ["--durations=5", "-n", "auto", "--tb=short"]
sync_groups = ["test"]
```

A package opts out of any default by writing the corresponding field in
its own `[tool.marin.tests]` (see §2). Otherwise the workspace baseline
applies.

## 2. Per-package override schema

A new optional TOML table in `lib/<pkg>/pyproject.toml`. Each field
overrides the workspace baseline; absent fields inherit. The schema
exists so a small number of packages can express genuinely-needed
deltas, not so every package re-declares the baseline.

```toml
# Optional. Most packages do NOT need this table.
[tool.marin.tests]
# uv sync extras (--extra X, repeated). Default: [].
# levanter uses ["torch_test"]; marin uses ["cpu", "dedup"].
sync_extras = ["torch_test"]

# uv sync groups (--group X, repeated). Default: ["test"].
# Override is for packages that use a non-"test" group name (haliax: "dev",
# fray: "fray-test").
sync_groups = ["test"]

# Extra `uv sync` args beyond the standard ones (--frozen, --package, --extra,
# --group). Reach for this only when the workspace baseline + the typed
# fields above can't express what you need. levanter torch uses
# ["--no-install-package", "torch"]. Default: [].
sync_extra_args = []

# Args passed between `uv run` and `pytest`. Default: [].
# haliax temporarily uses ["--with", "jax[cpu]==0.9.2"] (see Phase 0 cleanup).
uv_run_args = []

# pytest -m. Default: workspace baseline.
markers = "not slow and not tpu"

# Extra pytest args appended to the workspace defaults. Default: [].
pytest_args = ["-n", "4", "--dist", "worksteal"]

# Process env for the pytest step. All values must be strings (TOML ints
# rejected loudly so JAX_NUM_CPU_DEVICES = 8 doesn't silently become a no-op).
# Default: {}.
env = { JAX_NUM_CPU_DEVICES = "8", PYTHONASYNCIODEBUG = "1" }

# Repo-relative shell scripts run before the pytest invocation, in order.
# Used for the levanter CPU-torch wheel install. Default: [].
setup_scripts = ["infra/test_setup/install_torch_cpu.sh"]

# Node version for `actions/setup-node`. Some packages need Node during
# `uv sync` for protobuf generation. Default: not installed.
setup_node = "22"

# Per-leg timeout. Default: 15.
timeout_minutes = 15
```

**Validation contract** (enforced by `tests/infra/test_marin_tests_config.py`):

- Workspace baseline at `[tool.marin.tests.workspace]` MUST exist with
  `python` set; missing baseline is a CI-blocking error.
- Per-package `[tool.marin.tests]` is optional; absent table = pure
  baseline.
- All env values are strings; TOML ints/bools rejected with
  `ConfigError("env value for <key> must be a string, got <type>")`.
- Every path in `setup_scripts` resolves to an existing executable file.
- `python` cannot be set in a per-package table — the workspace pins one
  version for everyone (rejected with a precise error).

## 3. Loader API

`infra/marin_tests_config.py` (~80 lines):

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class TestsConfig:
    """Resolved test config for one scope (workspace baseline merged with
    any per-package overrides). Frozen; the orchestrator never mutates it.
    """
    package: str                              # short name, e.g. "levanter"
    sync_package: str                         # uv name, e.g. "marin-levanter"
    python: str                               # always = workspace.python
    sync_extras: tuple[str, ...] = ()
    sync_groups: tuple[str, ...] = ("test",)
    sync_extra_args: tuple[str, ...] = ()
    uv_run_args: tuple[str, ...] = ()
    markers: str = ""
    pytest_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    setup_scripts: tuple[str, ...] = ()
    setup_node: str | None = None
    timeout_minutes: int = 15

# Maps the analyzer's package names to the pyproject that owns them.
# Every member's [tool.marin.tests] lives in its own pyproject — including
# marin, whose table is in lib/marin/pyproject.toml even though its tests
# directory is the repo-top-level `tests/` (the renderer carries that as a
# separate constant — see TEST_DIR_FOR_SCOPE below).
SCOPE_TO_PYPROJECT: dict[str, str] = {
    "rigging":  "lib/rigging/pyproject.toml",
    "finelog":  "lib/finelog/pyproject.toml",
    "haliax":   "lib/haliax/pyproject.toml",
    "iris":     "lib/iris/pyproject.toml",
    "fray":     "lib/fray/pyproject.toml",
    "levanter": "lib/levanter/pyproject.toml",
    "zephyr":   "lib/zephyr/pyproject.toml",
    "marin":    "lib/marin/pyproject.toml",
}

# Where each scope's tests live. All but marin's are conventional;
# marin's `tests/` is at the repo root rather than under lib/marin/.
TEST_DIR_FOR_SCOPE: dict[str, str] = {
    **{name: f"lib/{name}/tests" for name in SCOPE_TO_PYPROJECT
       if name != "marin"},
    "marin": "tests",
}

# The workspace baseline lives at the repo root, separately from any
# member's [tool.marin.tests]:
#   pyproject.toml  ->  [tool.marin.tests.workspace]   (baseline only)
#   lib/marin/pyproject.toml  ->  [tool.marin.tests]   (marin's overrides)

def resolve(scope: str, repo_root: Path) -> TestsConfig:
    """Load the workspace baseline, merge any per-package override, return
    the fully-resolved config. Raises ConfigError on schema violation."""

def resolve_all(repo_root: Path) -> dict[str, TestsConfig]: ...

class ConfigError(Exception): ...
```

`sync_package` is **derived**, not declared. The convention is
`marin-<dir>` for `lib/<dir>/pyproject.toml` and `marin` for the root.

## 4. Concrete per-member configs after Phase 0 cleanup

Each table is the *full* `[tool.marin.tests]` content; absent fields
inherit from the workspace baseline. Four members need no table; four
carry small deltas (haliax conditional on the Phase 0 §7 step-5
review).

```toml
# lib/rigging/pyproject.toml — no [tool.marin.tests] table.
# lib/finelog/pyproject.toml — no [tool.marin.tests] table.
# lib/zephyr/pyproject.toml — no [tool.marin.tests] table.
# lib/fray/pyproject.toml   — no [tool.marin.tests] table
#                             (fray_test group renames to test in Phase 0).

# lib/iris/pyproject.toml
[tool.marin.tests]
env = { PYTHONASYNCIODEBUG = "1" }

# lib/haliax/pyproject.toml — conditional on Phase 0 §7 step 5.
# Default outcome: no table. If the JAX override is ratified:
[tool.marin.tests]
uv_run_args = ["--with", "jax[cpu]==0.9.2"]
env = { JAX_NUM_CPU_DEVICES = "8" }

# lib/levanter/pyproject.toml
[tool.marin.tests]
sync_extras = ["torch_test"]
sync_extra_args = ["--no-install-package", "torch"]
setup_scripts = [
    "infra/test_setup/install_torch_cpu.sh",
    "infra/test_setup/install_ffmpeg_apt.sh",
]
setup_node = "22"

# lib/marin/pyproject.toml
[tool.marin.tests]
sync_extras = ["cpu", "dedup"]
pytest_args = ["-n", "4", "--dist", "worksteal"]
```

## 5. Orchestrator workflow shape (`marin-unit.yaml`)

Three jobs:

```yaml
on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      force_run_all:
        description: "Bypass the analyzer; run every package's full suite."
        type: boolean
        default: false

concurrency:
  group: marin-unit-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  select:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.analyze.outputs.matrix }}
    steps:
      - uses: actions/checkout@v5
        with: { fetch-depth: 0 }
      - uses: astral-sh/setup-uv@v7
      - run: uv sync --group dev   # installs grimp
      - id: analyze
        run: |
          # PR  -> github.event.pull_request.base.sha
          # push to main -> github.event.before
          # workflow_dispatch with force_run_all=true -> --force-run-all
          BASE_REF="${{ github.event.pull_request.base.sha || github.event.before }}"
          ARGS=(--base-ref "$BASE_REF" --emit-github-output)
          if [ "${{ inputs.force_run_all }}" = "true" ]; then
            ARGS=(--force-run-all --emit-github-output)
          fi
          uv run python infra/select_tests.py "${ARGS[@]}"

  unit:
    needs: select
    if: ${{ needs.select.outputs.matrix != '[]' }}
    strategy:
      fail-fast: false
      matrix:
        include: ${{ fromJSON(needs.select.outputs.matrix) }}
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
      - id: render
        # Reads [tool.marin.tests], writes:
        #   $RUNNER_TEMP/leg/setup.sh    (concatenated setup_scripts)
        #   $RUNNER_TEMP/leg/pytest.sh   (uv run [args] pytest [args] tests)
        # Emits: python_version, sync_args, node_version, env_lines.
        run: |
          uv run python infra/run_tests.py prepare \
            "${{ matrix.package }}" \
            --tests "${{ join(matrix.tests, ' ') }}" \
            --out-dir "$RUNNER_TEMP/leg"
      - uses: actions/setup-node@v4
        if: steps.render.outputs.node_version != ''
        with: { node-version: ${{ steps.render.outputs.node_version }} }
      - run: uv python install ${{ steps.render.outputs.python_version }}
      - run: |
          echo "${{ steps.render.outputs.env_lines }}" >> "$GITHUB_ENV"
      - run: bash "$RUNNER_TEMP/leg/setup.sh"
      - run: uv sync ${{ steps.render.outputs.sync_args }}
      - run: bash "$RUNNER_TEMP/leg/pytest.sh"

  aggregate:
    needs: unit
    if: always()
    runs-on: ubuntu-latest
    steps:
      - run: |
          case "${{ needs.unit.result }}" in
            success|skipped) exit 0 ;;
            *) exit 1 ;;
          esac
```

**Contract:**

- `marin-unit` is the single status check branch protection requires.
- `infra/select_tests.py` gains `--emit-github-output` and
  `--force-run-all` (~20 lines).
- `run_all: true` lowers to a matrix where every package appears with
  `tests: []` (full suite signal); orchestrator does not need a separate
  code path.
- Multi-line shell goes into `$RUNNER_TEMP/leg/*.sh`, never into
  `$GITHUB_OUTPUT` (levanter's CPU-torch wheel install is genuinely a
  25-line heredoc).
- The leg always runs from repo root; pytest is invoked with explicit
  paths (`lib/<pkg>/tests/<file>` from the analyzer or `lib/<pkg>/tests`
  for full suite). No `cd`.

**`infra/run_tests.py prepare <package>` outputs:**

- `python_version` — string, e.g. `"3.12"`. From the workspace
  baseline.
- `sync_args` — single space-separated string ready for shell
  interpolation. Built as: `--frozen --package marin-<dir>` + one
  `--extra X` per `sync_extras` entry + one `--group X` per
  `sync_groups` entry + `sync_extra_args` joined verbatim. Example for
  levanter:
  `--frozen --package marin-levanter --extra torch_test --group test --no-install-package torch`.
- `node_version` — string, possibly empty. From `setup_node` field;
  empty disables the `actions/setup-node` step via the `if:` guard.
- `env_lines` — newline-delimited `KEY=VALUE` ready for `>> $GITHUB_ENV`.
  Values are not quoted; the loader rejects values containing newlines
  or `=` to keep this format unambiguous.
- Two files written to `$OUT_DIR` (default `$RUNNER_TEMP/leg/`):
  - `setup.sh` — `#!/usr/bin/env bash\nset -euo pipefail\n` then each
    `setup_scripts` entry as a `bash <path>` line in declared order.
    Empty body if `setup_scripts == []` (executes successfully).
  - `pytest.sh` — `#!/usr/bin/env bash\nset -euo pipefail\n` then a
    single line: `uv run <uv_run_args> pytest -m "<markers>" <pytest_args> <test_paths>`,
    where `test_paths` is either the analyzer's explicit list or
    `lib/<pkg>/tests` (full suite) or `tests` (marin full suite).

## 6. File-path summary

| Path | Status | Purpose |
|---|---|---|
| `infra/select_tests.py` | exists | Analyzer; gains `--emit-github-output` and `--force-run-all` flags |
| `infra/marin_tests_config.py` | new | Loader + dataclass; ~80 lines |
| `infra/run_tests.py` | new | Renders `TestsConfig` + matrix entry into shell scripts; ~120 lines |
| `infra/test_setup/install_torch_cpu.sh` | new | Lifted from `levanter-unit.yaml:119-146` |
| `infra/test_setup/install_ffmpeg_apt.sh` | new | Lifted from `levanter-unit.yaml:149-151` |
| `tests/infra/test_select_tests.py` | exists | 25 tests |
| `tests/infra/test_marin_tests_config.py` | new | Schema + baseline + override-merge tests; ~80 lines |
| `tests/infra/test_run_tests.py` | new | Asserts rendered script content for fixture configs; ~50 lines |
| `lib/<pkg>/pyproject.toml` (3 of 7 lib packages) | edit | iris, haliax, levanter, fray, marin gain a small table; rigging/finelog/zephyr untouched |
| `pyproject.toml` (root) | edit | Adds `[tool.marin.tests.workspace]` baseline |
| `.github/workflows/marin-unit.yaml` | rewrite | Three-job orchestrator |
| `.github/workflows/{haliax,levanter,iris,zephyr,fray}-unit.yaml` | delete | 5 deletions |
| `.github/workflows/levanter-tpu-tests.yaml` | new | Extracted from `levanter-unit.yaml:159-227` verbatim |
| `.github/workflows/dupekit-unit.yaml` | unchanged | Rust+Python hybrid stays standalone |
| `.github/workflows/marin-integration.yaml` | edit | Absorbs `iris-e2e-smoke` from `iris-unit.yaml:65-131` |

## 7. Phase 0 cleanup (delete the cruft first)

Land **before** the orchestrator goes live; each is independently
defensible and easier to review without the orchestrator change.

1. **Delete `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`** from `marin-unit.yaml:69`
   and the "don't cd lib/zephyr so Ray uv integration doesn't freak out"
   comment from `zephyr-unit.yaml:65`. Confirmed dead: zero `import ray`
   or `ray.init()` calls anywhere in `lib/`.
2. **Delete `-n1`** from `iris-unit.yaml:63`. iris doesn't actually need
   single-process; switch to `-n auto` like the rest of the workspace.
3. **Standardize on Python 3.12.** `haliax-unit.yaml:47` and
   `levanter-unit.yaml:47` set 3.11; every package's `requires-python`
   allows 3.12 (`lib/haliax/pyproject.toml:8`, `lib/levanter/pyproject.toml:8`,
   etc.). Bump both workflows to 3.12.
4. **Standardize the test dependency-group name to `test`** across every
   `lib/<pkg>/pyproject.toml` `[dependency-groups]` table. Today's
   layout (verified):
   - **`test` already exists**: levanter, zephyr, marin. No change.
   - **`fray_test` rename → `test`**: `lib/fray/pyproject.toml:28`. Also
     update **both** `include-group` references — `fray_tpu_test`
     (line 33) and `dev` (line 35) currently include `fray_test`.
     Regenerate `uv.lock` after the rename; the lockfile's `fray-test`
     group entries (uv.lock:5359, 5398) become `test`.
   - **Split `dev` into `dev` (tooling only) + `test` (test-only deps)**:
     haliax, iris, rigging, finelog. Move pytest, pytest-asyncio,
     pytest-xdist, pytest-cov, etc. from `dev` to a new `test` group;
     leave linters, type checkers, and editor helpers in `dev`. If any
     entry serves both roles, duplicate it (uv resolves dups
     idempotently). haliax's `dev` is the largest and most heterogeneous
     — flag in the PR description that humans should sanity-check the
     split.
   After this step, `uv sync --group test` works uniformly and no
   per-member `[tool.marin.tests]` overrides `sync_groups`.
5. **Re-evaluate haliax's `--with "jax[cpu]==0.9.2"` runtime pin
   (`haliax-unit.yaml:55`).** The lockfile already pins JAX. If the
   override is intentional (test haliax against the JAX version levanter
   uses), tighten haliax's `pyproject.toml` `jax >= 0.8.0` to
   `jax >= 0.9.2,<0.11` and drop the override. If it's not, just drop
   it. If it's neither and reviewers want to keep the override,
   `uv_run_args` in §2 carries it forward.
6. **Drop `-c pyproject.toml`** from `haliax-unit.yaml:55`. Pytest
   auto-discovers config; the explicit `-c` is redundant.
7. **Verify `PYTHONPATH` overrides become unnecessary.** Today four
   workflows set `PYTHONPATH=tests:src:.` or `PYTHONPATH=tests:.`
   (haliax/levanter/marin) — a vestige of `cd lib/<pkg>/` discipline
   that the unified workflow eliminates (legs always run from repo
   root with explicit `lib/<pkg>/tests/...` paths). Run the orchestrator
   in shadow mode for one pass; if a test breaks because of `sys.path`
   expectations, the per-member workaround is `env = { PYTHONPATH =
   "lib/<pkg>/tests:lib/<pkg>/src" }`, but the better fix is updating
   the test or its `conftest.py` to not rely on `sys.path` munging in
   the first place.

After Phase 0, the per-package YAMLs are simpler and the diff that
introduces `marin-unit.yaml` becomes much easier to review.

## 8. Errors

- `ConfigError("missing [tool.marin.tests.workspace] in pyproject.toml")`
  — root baseline must exist; surfaced by `test_marin_tests_config.py`.
- `ConfigError("env value for <key> must be a string, got <type>")` —
  reject TOML ints/bools loudly.
- `ConfigError("env value for <key> contains '=' or newline; not
  representable in $GITHUB_ENV")` — the renderer can't encode such
  values in the `env_lines` newline-delimited `KEY=VALUE` format used
  to populate `$GITHUB_ENV`.
- `ConfigError("setup_scripts entry not found: <path>")` — typo guard.
- `ConfigError("python may only be set in workspace baseline, not in
  lib/<pkg>/pyproject.toml")` — packages cannot diverge on Python.

## 9. Out of scope

- **`dupekit-unit.yaml`** stays as written.
- **`levanter-tpu-tests.yaml`** stays separate. Future migration onto
  the Iris prod cluster can adopt this schema later.
- **`marin-canary-*.yaml`, `marin-release-*.yaml`, `iris-smoke-*.yaml`**,
  scheduled triage — not unit tests.
- **Coverage-based test selection** (testmon, etc.) — explicitly
  rejected.
- **Per-package Python version matrix** — workspace pins one version.
  Multi-version testing is a `dupekit` need handled by `dupekit-unit.yaml`.
- **Test sharding within a package** — one matrix leg per package.
- **`uv sync` cache key tuning** — `astral-sh/setup-uv`'s default cache
  is good enough.

## 10. Migration plan

**Phase 0 — cleanup PRs (7 small).** Each is an independently mergeable
change; land them before the orchestrator (§7). These shrink the
existing seven YAMLs and make the orchestrator diff small. A separate
in-progress workspace effort to remove vestigial lazy imports is
landing in parallel — that effort is independent (it doesn't block the
orchestrator) but improves analyzer precision: the analyzer sees only
top-level imports today, so eliminating dead lazy imports brings the
test-impact graph closer to the runtime-import graph.

**Phase 1 — shadow mode (1 PR).** Land `marin-unit.yaml` with
`continue-on-error: true` on every step, alongside the existing six
remaining `*-unit.yaml` files (still required). Workflow_dispatch input
`force_run_all` is available for spot-checks. ~1 week of live data.

**Phase 2 — required, parallel (1 PR).** Drop `continue-on-error`. New
`marin-unit` aggregate becomes a *non-required* check; old YAMLs still
gate. Compare false-positive / false-negative rates on real PRs.
~3-7 days. Audit step: grep all workflows for `needs:` references to
job names that are about to disappear.

**Phase 3 — switch (1 PR + admin steps).** GitHub-admin sequence is
load-bearing:

1. Admin removes `haliax-unit / levanter-unit / iris-unit / zephyr-unit
   / fray-unit / marin-unit (old)` from branch-protection
   required-checks.
2. PR merges: deletes the five obsolete YAMLs, adds
   `levanter-tpu-tests.yaml`, edits `marin-integration.yaml` to absorb
   `iris-e2e-smoke`.
3. Admin adds `marin-unit` (new aggregate) as a required check.

**Rollback.** Tag `legacy/unit-workflows-<YYYYMMDD>` before Phase 3.
Cherry-pick the old YAMLs back from the tag if the new workflow needs
to be reverted.
