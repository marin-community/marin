# Unified Unit-Test Workflow

_Why are we doing this? What's the benefit?_

We have seven `*-unit.yaml` workflows
(`.github/workflows/{haliax,levanter,iris,zephyr,fray,marin,dupekit}-unit.yaml`)
that each encode the same shape — `dorny/paths-filter` → `uv sync` →
`pytest` — with sharp per-package divergence in install commands, env
vars, marker filters, and working directories. The path filters under-
or over-approximate the real import dependencies, and adding a new
workspace package requires authoring (and forgetting to update) yet
another YAML. We already have an analyzer (`infra/select_tests.py:243`)
that computes a precise per-package test matrix from any diff. The next
step is to delete the per-package YAMLs and let one orchestrator drive
all unit tests, with each package declaring how to run its tests in its
own `pyproject.toml`. End state: `marin-unit.yaml`, `marin-integration.yaml`,
`dupekit-unit.yaml`, and a separate `levanter-tpu-tests.yaml` — five
files deleted and one rewritten.

## Background

The seven workflows diverge on Python version, `uv sync` extras/groups,
markers, parallelism, and per-package quirks (CPU-torch wheel dance,
`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, "don't `cd lib/zephyr`"); five of seven
path filters miss transitive dependencies. `infra/select_tests.py` already
emits a `{run_all, matrix}` JSON ready to drive a matrix job. Prior art
(Bazel `py_test`, Pants `python_tests`, Nx `project.json`, Turborepo
`turbo.json`) all converge on **declarative per-target config with a
small typed schema**. Coverage-based test selection (testmon-style)
loses on cache-invalidation. Full digest in `research.md`.

## Challenges

**Per-package quirks must survive consolidation.** Levanter's torch suite
needs the CPU-pinned torch wheel from the pytorch wheelhouse extracted
out of `uv.lock` (`levanter-unit.yaml:119-146`); zephyr must not `cd
lib/zephyr` because Ray's uv integration breaks (`zephyr-unit.yaml:65`);
marin needs `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` (`marin-unit.yaml:69`);
haliax wants a runtime JAX pin via `--with` (`haliax-unit.yaml:55`); iris
wants single-process (`-n1`); marin wants `-n 4 --dist=worksteal`. A
unified workflow must let each package declare these without smuggling
package-specific shell into the orchestrator.

**The orchestrator is on the critical path for every PR.** Today, six of
the seven workflows skip themselves entirely if the path filter doesn't
match. The new orchestrator runs analyzer + matrix-emit on every PR.
The analyzer takes ~3s once `uv sync --group dev` has installed `grimp`;
the matrix can be empty (zero matrix legs run) so the wall-clock cost
on a docs-only PR is `uv sync` + 3s. That's still more than the current
~5s "filter-says-skip" path.

## Costs / Risks

- **Migration churn.** Five workflow files deleted, one rewritten, every
  scope gets a new `[tool.marin.tests]` table. Branch-protection rules
  currently name jobs like `levanter-unit` / `iris-unit`; an admin must
  remove those and add the new `marin-unit` aggregate as a required
  check (sequenced — see `spec.md` §7).
- **`grimp` is a hard CI dependency now.** If the analyzer fails (a
  transitive import error, a malformed package) every PR's unit tests
  are dead. The pressure-relief is a `--force-run-all` flag on
  `select_tests.py` plus a `workflow_dispatch` input on `marin-unit.yaml`
  so a human can bypass the analyzer in a panic.
- **Cold-start floor on every PR.** The current `dorny/paths-filter` skip
  costs ~5s; the new `select` job (checkout + setup-uv + `uv sync --group
  dev` + grimp graph build) costs ~30–60s even when the resulting matrix
  is empty. Net regression on docs-only PRs.
- **Levanter sub-suite parallelism is lost.** Today `levanter-unit`,
  `levanter-entry`, and `levanter-torch` run on three runners
  concurrently; collapsed into one matrix leg they serialize. Wall-clock
  hit on PRs that touch `levanter` source.
- **No immediate user-visible improvement.** PR contributors notice
  faster CI on small changes (analyzer skips most legs) and more
  reliable CI on cross-package changes (transitive deps now caught) —
  both real but not earth-moving.

## Design

**Package-local test config in `pyproject.toml`.** Each
`lib/<pkg>/pyproject.toml` (and the root `pyproject.toml` for the marin
scope, which owns top-level `tests/`) gains a small `[tool.marin.tests]`
table:

```toml
[tool.marin.tests]
sync_package = "marin-levanter"     # uv --package
python = "3.11"
working_dir = "."
sync_args = ["--extra", "torch_test", "--group", "test"]
uv_run_args = []                    # haliax sets ["--with", "jax[cpu]==0.9.2"]
markers = "not slow and not tpu and not tpu_ci"
pytest_args = ["--durations=20"]
pythonpath = ["tests", "src"]
env = { RAY_ENABLE_UV_RUN_RUNTIME_ENV = "0" }
setup_scripts = ["infra/test_setup/install_torch_cpu.sh"]
setup_node = "22"
```

The schema is flat by design. `sync_args` and `uv_run_args` accept any
flag the underlying tool understands (so haliax's runtime jax pin and
levanter's `--no-install-package torch` don't need new fields).
`setup_scripts` is a list of repo-relative paths to shell scripts —
not raw shell, so the genuinely weird stuff (CPU-torch wheel install,
ffmpeg apt-install) lives in version-controlled, lintable, locally-
runnable scripts under `infra/test_setup/`. Field semantics and
validation rules in `spec.md`.

**Single orchestrator: `marin-unit.yaml`.** Three jobs, with
`fail-fast: false` and a workflow-level `concurrency: marin-unit-<pr>`
that cancels in-flight runs on push:

1. `select` — checks out, `uv sync --group dev` (installs grimp), runs
   `python infra/select_tests.py --emit-github-output` against the
   resolved base ref. Base ref is `pull_request.base.sha` on PR,
   `event.before` on push to main, or `--force-run-all` when the
   workflow_dispatch input asks. Emits the matrix as a step output.
2. `unit` — `strategy.matrix.include: ${{ fromJSON(...) }}` consumes the
   matrix. Each leg invokes `infra/run_tests.py prepare <package>`
   which loads the `[tool.marin.tests]` table and writes two scripts to
   `$RUNNER_TEMP/test-leg/` (setup.sh and pytest.sh) — outputs only
   carry scalars (python version, sync args, env-as-KEY=VAL lines). The
   leg then sets up Python via `uv python install`, optionally runs
   `actions/setup-node`, lifts env into `$GITHUB_ENV`, runs setup.sh,
   `uv sync ...`, then pytest.sh. If the analyzer's `tests` list is
   non-empty, pytest runs on that explicit list (precise); empty list
   = forced full suite.
3. `aggregate` — depends on `unit`, exits 0 on success or skipped
   (empty matrix is fine — no test-affecting changes), 1 otherwise.
   Single status check `marin-unit` is what branch protection requires.

The "writes scripts to a temp dir" pattern is deliberate: GitHub Actions
step outputs cannot reliably carry multi-line shell, and levanter's torch
wheel install is a 25-line heredoc.

**`marin-integration.yaml` absorbs `iris-e2e-smoke`.** Playwright +
Claude-screenshot-verification isn't unit-shaped (boots a server, drives
a real browser, calls an external API). It moves to integration where
heavier multi-component scenarios already live.

**`dupekit-unit.yaml` stays as-is.** Rust+Python hybrid doesn't fit the
analyzer (which only sees Python imports). Folding cargo into either
marin-unit or marin-lint blurs the boundary.

**`levanter-tpu-tests` extracted.** The Docker-TPU job in
`levanter-unit.yaml:159-226` becomes `levanter-tpu-tests.yaml`,
unchanged in content. Stays separate "for now" per the user's framing;
folds into the unified policy later when it can run on the Iris prod
cluster.

**Sub-suites collapse.** Levanter's three Python sub-jobs (unit / entry
/ torch) become one matrix leg with one marker filter. The AST-driven
test selection is the safety net the marker-split used to be: a
torch-only test only fires when its imported source appears in the
affected set.

## Testing

The analyzer is already covered by 25 unit tests in
`tests/infra/test_select_tests.py`. Two new test surfaces:

- **A `[tool.marin.tests]` schema test per package.** Loads each
  package's table, asserts required fields are present and types check.
  Lives at `tests/infra/test_marin_tests_config.py`. Failing this test
  in CI prevents merging a malformed table.
- **Orchestrator dry-run.** A `marin-unit.yaml` PR can be exercised in
  draft against this branch using GitHub's `workflow_dispatch`; success
  criteria: matrix shape matches the analyzer JSON for known diffs
  (e.g. a haliax-only change emits 1 leg), every leg's pytest runs
  green, status check `marin-unit` aggregates correctly.

Migration order: land the orchestrator behind `workflow_dispatch` first,
verify against three or four real PRs, then flip the trigger to
`pull_request` and delete the six old YAMLs in the same PR (so the
branch-protection rename happens atomically).

## Open Questions

- **Cold-start cost on docs-only PRs.** The `select` job runs on every
  PR including ones whose final matrix is empty. ~30–60s floor. Worth
  it for the precision wins, but I want a sanity check that the docs/
  PR experience won't deteriorate enough to bother people.
- **Branch-protection migration.** Phase 3 of `spec.md` §7 requires a
  three-step admin sequence (remove old required checks, merge the
  switch PR, add `marin-unit` as required). Should this happen with
  someone in the room, or do we trust the runbook?
- **Should `marin` scope live at the root pyproject or `lib/marin/`?**
  Today the analyzer attributes top-level `tests/` to `marin`, but
  `lib/marin/tests/` is empty. `spec.md` puts the marin scope's
  `[tool.marin.tests]` in the *root* pyproject. If anyone plans to
  populate `lib/marin/tests/` later, the routing needs another scope.
- **Should the analyzer become a reusable composite action?** Inline in
  the orchestrator keeps blast radius small; composite would let other
  workflows consult the matrix. Starting inline; reconsider if a second
  consumer shows up.
