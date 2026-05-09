# Spec — unified_unit_workflow

The contracts implied by `design.md`. Reviewers should be able to read
this and answer "would I actually build this exact API?"

The schema is **convention with overrides**: a baseline that covers most
packages with no per-package config, plus a tiny `[tool.marin.tests]`
table for the genuinely different packages. Most workspace members will
have *no* table.

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
SCOPE_TO_PYPROJECT: dict[str, str] = {
    "rigging":  "lib/rigging/pyproject.toml",
    "finelog":  "lib/finelog/pyproject.toml",
    "haliax":   "lib/haliax/pyproject.toml",
    "iris":     "lib/iris/pyproject.toml",
    "fray":     "lib/fray/pyproject.toml",
    "levanter": "lib/levanter/pyproject.toml",
    "zephyr":   "lib/zephyr/pyproject.toml",
    "marin":    "pyproject.toml",            # root owns top-level tests/
}

def resolve(scope: str, repo_root: Path) -> TestsConfig:
    """Load the workspace baseline, merge any per-package override, return
    the fully-resolved config. Raises ConfigError on schema violation."""

def resolve_all(repo_root: Path) -> dict[str, TestsConfig]: ...

class ConfigError(Exception): ...
```

`sync_package` is **derived**, not declared. The convention is
`marin-<dir>` for `lib/<dir>/pyproject.toml` and `marin` for the root.

## 4. Concrete per-package configs after Phase 0 cleanup

This is what each package's table actually looks like under the new
scheme. Four packages need no table; the rest carry small deltas.

| Package | `[tool.marin.tests]` |
|---|---|
| `rigging` | (no table — pure baseline) |
| `finelog` | (no table) |
| `zephyr` | (no table) |
| `fray` | (no table — `fray_test` group renamed to `test` in Phase 0) |
| `iris` | `env = { PYTHONASYNCIODEBUG = "1" }` |
| `haliax` | `uv_run_args = ["--with", "jax[cpu]==0.9.2"]` + `env = { JAX_NUM_CPU_DEVICES = "8" }` if the Phase 0 review keeps the JAX override; otherwise no table |
| `levanter` | `sync_extras = ["torch_test"]`, `sync_extra_args = ["--no-install-package", "torch"]`, `setup_scripts = ["infra/test_setup/install_torch_cpu.sh", "infra/test_setup/install_ffmpeg_apt.sh"]`, `setup_node = "22"` |
| `marin` | `sync_extras = ["cpu", "dedup"]`, `pytest_args = ["-n", "4", "--dist", "worksteal"]` |

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
  `$GITHUB_OUTPUT` (levanter's torch wheel install is genuinely 25
  lines).
- The leg always runs from repo root; pytest is invoked with explicit
  paths (`lib/<pkg>/tests/<file>` from the analyzer or `lib/<pkg>/tests`
  for full suite). No `cd`.

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
   `lib/<pkg>/pyproject.toml` `[dependency-groups]` table. Today:
   levanter, zephyr, marin already have a `test` group; haliax, iris,
   rigging, finelog put test deps under `dev` (or have no test group at
   all); fray uses `fray_test`. Split or rename so `uv sync --group
   test` works uniformly — pure dev-tooling stays under `dev`, test-only
   deps move to `test`. After this, `sync_groups` defaults to `["test"]`
   and no package overrides it.
5. **Re-evaluate haliax's `--with "jax[cpu]==0.9.2"` runtime pin
   (`haliax-unit.yaml:55`).** The lockfile already pins JAX. If the
   override is intentional (test haliax against the JAX version levanter
   uses), tighten haliax's `pyproject.toml` `jax >= 0.8.0` to
   `jax >= 0.9.2,<0.11` and drop the override. If it's not, just drop
   it. If it's neither and reviewers want to keep the override,
   `uv_run_args` in §2 carries it forward.
6. **Drop `-c pyproject.toml`** from `haliax-unit.yaml:55`. Pytest
   auto-discovers config; the explicit `-c` is redundant.

After Phase 0, the per-package YAMLs are simpler and the diff that
introduces `marin-unit.yaml` becomes much easier to review.

## 8. Errors

- `ConfigError("missing [tool.marin.tests.workspace] in pyproject.toml")`
  — root baseline must exist; surfaced by `test_marin_tests_config.py`.
- `ConfigError("env value for <key> must be a string, got <type>")` —
  reject TOML ints/bools loudly.
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

**Phase 0 — cleanup PRs (5 small).** Each is an independently mergeable
change; land them before the orchestrator (§7). These shrink the
existing seven YAMLs and make the orchestrator diff small.

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
