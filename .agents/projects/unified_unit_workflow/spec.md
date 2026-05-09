# Spec — unified_unit_workflow

The contracts implied by `design.md`. Reviewers should be able to read
this and answer "would I actually build this exact API?"

## 1. `[tool.marin.tests]` schema

A new TOML table in each `lib/<pkg>/pyproject.toml` and the root
`pyproject.toml` (which owns the top-level `tests/` directory the
analyzer attributes to the `marin` scope — see §7). The schema is flat,
typed, and small. Fields are optional unless marked required.

```toml
[tool.marin.tests]
# uv package to sync (--package). Different from the directory name —
# every workspace member is `marin-<dir>`. Required.
sync_package = "marin-levanter"

# Python interpreter version. Default = workspace default = "3.12".
python = "3.11"

# Working directory for the pytest invocation, relative to repo root.
# "." (default) avoids the Ray uv-integration breakage that bites zephyr;
# fray/iris today require "lib/<pkg>" — declare explicitly.
working_dir = "."

# Pass-through to `uv sync`. Anything legal for `uv sync` is legal here:
# --frozen is added automatically; opt out via sync_frozen=false.
# Examples: ["--extra", "torch_test", "--group", "test", "--dev",
#           "--no-install-package", "torch"]
sync_args = ["--extra", "torch_test", "--group", "test"]
sync_frozen = true                          # default true; haliax sets false

# Pass-through to `uv run` (NOT pytest). Goes between `uv run` and
# `pytest`. Used by haliax for the runtime jax pin:
#   ["--with", "jax[cpu]==0.9.2"]
uv_run_args = []

# pytest options. `markers` becomes `-m <markers>`; `pytest_args` is
# everything else.
markers = "not slow and not tpu and not tpu_ci"
pytest_args = ["--durations=20", "-n", "4", "--dist", "worksteal",
               "-c", "pyproject.toml"]

# Repo-relative directories joined with ':' and exported as PYTHONPATH.
# Empty list (default) leaves PYTHONPATH untouched.
pythonpath = ["tests", "src", "."]

# Process environment for the pytest step. TOML lets you write integers
# but every value MUST be a string here — the loader rejects non-string
# values loudly so engineers don't write `JAX_NUM_CPU_DEVICES = 8` and
# silently get a no-op.
env = { RAY_ENABLE_UV_RUN_RUNTIME_ENV = "0", PYTHONASYNCIODEBUG = "1" }

# Optional. Repo-relative shell scripts run in order before the pytest
# invocation. The CPU-torch-from-pytorch-wheelhouse dance lives at
# infra/test_setup/install_torch_cpu.sh; new shared scripts go next to
# it. NOT raw shell — the path is the contract; the script's content is
# normal version-controlled bash that the orchestrator just executes.
setup_scripts = ["infra/test_setup/install_torch_cpu.sh",
                 "infra/test_setup/install_ffmpeg_apt.sh"]

# Optional. Node version to install via actions/setup-node before the
# pytest step. levanter, iris, and zephyr need this for their protobuf
# generation in `uv sync`. Empty/missing skips Node setup.
setup_node = "22"

# Per-leg timeout in minutes. Default 15.
timeout_minutes = 15
```

**Validation contract** (enforced by `tests/infra/test_marin_tests_config.py`):

- Every `lib/<pkg>/pyproject.toml` and the root `pyproject.toml` MUST
  have `[tool.marin.tests]`. A missing table is a CI-blocking error,
  not a silent skip.
- `python` matches `^3\.\d+$`.
- All elements of `sync_args`, `uv_run_args`, `pytest_args`,
  `pythonpath`, `setup_scripts` are non-empty strings.
- All values in `env` are strings (TOML ints/bools rejected with a
  precise error: "env value for `<key>` must be a string, got `<type>`").
- Every path in `setup_scripts` resolves to an existing executable file
  under the repo.

## 2. Loader API

`infra/marin_tests_config.py` (~120 lines):

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class TestsConfig:
    """Parsed [tool.marin.tests] for one scope (a lib package or marin-root).

    Frozen; the orchestrator never mutates it.
    """
    sync_package: str
    python: str = "3.12"
    working_dir: str = "."
    sync_args: tuple[str, ...] = ()
    sync_frozen: bool = True
    uv_run_args: tuple[str, ...] = ()
    markers: str = ""
    pytest_args: tuple[str, ...] = ()
    pythonpath: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    setup_scripts: tuple[str, ...] = ()
    setup_node: str | None = None
    timeout_minutes: int = 15

# Maps the analyzer's package names to the pyproject that owns them.
# "marin" routes to the repo-root pyproject (which owns top-level tests/);
# every other workspace package routes to lib/<pkg>/pyproject.toml.
SCOPE_TO_PYPROJECT: dict[str, str] = {
    "rigging": "lib/rigging/pyproject.toml",
    "finelog": "lib/finelog/pyproject.toml",
    "haliax": "lib/haliax/pyproject.toml",
    "iris": "lib/iris/pyproject.toml",
    "fray": "lib/fray/pyproject.toml",
    "levanter": "lib/levanter/pyproject.toml",
    "zephyr": "lib/zephyr/pyproject.toml",
    "marin": "pyproject.toml",
}

def load(scope: str, repo_root: Path) -> TestsConfig: ...
def load_all(repo_root: Path) -> dict[str, TestsConfig]: ...

class ConfigError(Exception):
    """Raised when [tool.marin.tests] is missing or malformed."""
```

Note the deliberate **one-config-per-scope** rule: `lib/marin/pyproject.toml`
does NOT carry a `[tool.marin.tests]` table; the analyzer attributes
top-level `tests/` to the `marin` scope, and the root pyproject owns
that scope. The validation test enforces both directions (every scope
in `SCOPE_TO_PYPROJECT` has a table; no extras).

## 3. Orchestrator workflow shape (`marin-unit.yaml`)

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
          # Resolve base ref:
          #   PR  -> github.event.pull_request.base.sha
          #   push to main -> github.event.before
          #   workflow_dispatch with force_run_all -> --force-run-all
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
    timeout-minutes: ${{ fromJSON(needs.select.outputs.matrix)[matrix.index].timeout_minutes || 15 }}
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
      - id: render
        # Reads [tool.marin.tests], writes:
        #   $RUNNER_TEMP/test-leg/setup.sh    (concatenated setup_scripts)
        #   $RUNNER_TEMP/test-leg/pytest.sh   (cd + uv run pytest one-liner)
        # Emits: python_version, sync_args (as a single string), node_version,
        # env_lines (KEY=VAL\nKEY=VAL...).
        run: |
          uv run python infra/run_tests.py prepare \
            "${{ matrix.package }}" \
            --tests "${{ join(matrix.tests, ' ') }}" \
            --out-dir "$RUNNER_TEMP/test-leg"
      - uses: actions/setup-node@v4
        if: steps.render.outputs.node_version != ''
        with: { node-version: ${{ steps.render.outputs.node_version }} }
      - run: uv python install ${{ steps.render.outputs.python_version }}
      - run: |
          # Lift env into $GITHUB_ENV so every subsequent step sees it.
          echo "${{ steps.render.outputs.env_lines }}" >> "$GITHUB_ENV"
      - run: bash "$RUNNER_TEMP/test-leg/setup.sh"
      - run: uv sync ${{ steps.render.outputs.sync_args }}
      - run: bash "$RUNNER_TEMP/test-leg/pytest.sh"

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
  `aggregate` exits 0 on `success` (some legs ran, all passed) or
  `skipped` (analyzer emitted empty matrix); exits 1 on `failure` /
  `cancelled`.
- `infra/select_tests.py` gains two flags: `--emit-github-output`
  (writes `matrix=<json>` to `$GITHUB_OUTPUT`) and `--force-run-all`
  (skips the diff, emits a full-suite matrix as if every package were
  affected).
- `run_all: true` lowers to a matrix where every package appears with
  `tests: []` (full-suite signal); the orchestrator does not need a
  separate code path.
- `infra/run_tests.py prepare` (~120 lines) is the renderer. It writes
  shell scripts into a temp dir rather than emitting them as outputs —
  GitHub Actions outputs cannot reliably carry multi-line shell, and
  levanter's torch wheel install is genuinely 25 lines. Scripts are
  executable, traceable in CI logs, and invokable locally.

## 4. File-path summary

| Path | Status | Purpose |
|---|---|---|
| `infra/select_tests.py` | exists | Analyzer; gains `--emit-github-output` and `--force-run-all` flags (~20 lines) |
| `infra/marin_tests_config.py` | new | Loader + dataclass; ~120 lines |
| `infra/run_tests.py` | new | Renders `TestsConfig` + matrix entry into shell scripts; ~120 lines |
| `infra/test_setup/install_torch_cpu.sh` | new | Lifted from `levanter-unit.yaml:119-146` verbatim |
| `infra/test_setup/install_ffmpeg_apt.sh` | new | Lifted from `levanter-unit.yaml:149-151` |
| `tests/infra/test_select_tests.py` | exists | 25 tests; unchanged |
| `tests/infra/test_marin_tests_config.py` | new | Schema validation; asserts every `SCOPE_TO_PYPROJECT` entry has a table; ~70 lines |
| `tests/infra/test_run_tests.py` | new | Renders a fixture `TestsConfig`, asserts the script content; ~50 lines |
| `lib/<pkg>/pyproject.toml` (×7 lib packages) | edit | Each gains `[tool.marin.tests]` |
| `pyproject.toml` (root) | edit | Gains `[tool.marin.tests]` for the top-level `tests/` dir (the marin scope) |
| `.github/workflows/marin-unit.yaml` | rewrite | New three-job orchestrator |
| `.github/workflows/{haliax,levanter,iris,zephyr,fray}-unit.yaml` | delete | 5 deletions; subsumed by `marin-unit` |
| `.github/workflows/levanter-tpu-tests.yaml` | new | Extracted from `levanter-unit.yaml:159-227` verbatim |
| `.github/workflows/dupekit-unit.yaml` | unchanged | Rust+Python hybrid stays standalone |
| `.github/workflows/marin-integration.yaml` | edit | Absorbs `iris-e2e-smoke` from `iris-unit.yaml:65-131` |

## 5. Errors

- `ConfigError("missing [tool.marin.tests] in <path>")` — every scope in
  `SCOPE_TO_PYPROJECT` must opt in. Surfaced by
  `test_marin_tests_config.py`; blocks merge.
- `ConfigError("env value for <key> must be a string, got <type>")` —
  GitHub Actions env values are strings; reject TOML ints/bools at
  config-load time.
- `ConfigError("setup_scripts entry not found: <path>")` — scripts must
  exist on disk; typo-guard.
- `ConfigError("invalid python version: <value>")` — must match
  `^3\.\d+$`.

## 6. Out of scope

- **`dupekit-unit.yaml`** stays as written. Rust toolchain + 3-version
  Python matrix doesn't fit the analyzer.
- **`levanter-tpu-tests.yaml`** stays separate. Future migration onto
  the Iris prod cluster can adopt `[tool.marin.tests]` later.
- **`marin-canary-*.yaml`, `marin-release-*.yaml`, `iris-smoke-*.yaml`**,
  scheduled triage jobs — not unit tests.
- **Coverage-based test selection** — explicitly rejected per prior art
  (testmon-style cache invalidation hides regressions).
- **Per-package Python version matrix** — each scope picks one Python.
  Multi-version testing is a `dupekit` need that lives in
  `dupekit-unit.yaml`.
- **Caching of `uv sync` between matrix legs** — `astral-sh/setup-uv`'s
  built-in cache handles it; no orchestrator-level cache logic.
- **Test sharding within a package** — one matrix leg per package. If a
  single package's full suite ever outgrows the runner's wall clock,
  sharding can be added as a `shard_count` field; not needed today.

## 7. Migration plan

Phased rollout, each phase reversible until the last.

**Phase 1 — shadow mode (1 PR).** Land `marin-unit.yaml` with
`continue-on-error: true` on every step, alongside the existing seven
`*-unit.yaml` files (which remain required). Analyzer + matrix shape
verified against real PRs; failures don't block merge. Workflow_dispatch
input `force_run_all` available for spot-checks. Live for ~1 week.

**Phase 2 — required, parallel (1 PR).** Drop `continue-on-error`. New
workflow's `marin-unit` aggregate status becomes a *non-required* check
in branch protection — visible to reviewers but not gating. Old YAMLs
still run and still gate. Live for ~3-7 days; compare false-positive /
false-negative rates against the old workflows on real PRs.

**Phase 3 — switch (1 PR + 1 admin step).** GitHub-admin sequence is
load-bearing and cannot be a single PR:

1. **Admin** removes `haliax-unit / levanter-unit / iris-unit /
   zephyr-unit / fray-unit / marin-unit (old name) / dupekit-unit` from
   branch-protection required-checks.
2. **PR merges**: deletes the five obsolete `*-unit.yaml` files; adds
   `levanter-tpu-tests.yaml`; edits `marin-integration.yaml` to absorb
   `iris-e2e-smoke`. CI on the PR runs `marin-unit` + `marin-integration`
   + the unrelated checks.
3. **Admin** adds the new `marin-unit` aggregate status as required on
   `main`.

**Rollback.** Before Phase 3, tag the pre-deletion main branch as
`legacy/unit-workflows-<YYYYMMDD>`. If the new workflow is unstable in
production, cherry-pick the seven YAMLs back from the tag in a single
PR; admin re-adds the old required checks.

**Audit before Phase 3.** Search the repo for `needs: <old-job-name>`
and `requires: <old-job-name>` in case any scheduled / canary / release
workflow has a status-check dependency on a soon-to-be-deleted job
name. Currently I see none, but verifying is part of Phase 2.
