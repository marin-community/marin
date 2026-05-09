# Unified Unit-Test Workflow

_Why are we doing this? What's the benefit?_

We have seven `*-unit.yaml` workflows
(`.github/workflows/{haliax,levanter,iris,zephyr,fray,marin,dupekit}-unit.yaml`)
that each encode the same shape — `dorny/paths-filter` → `uv sync` →
`pytest` — with sharp per-package divergence in install commands, env
vars, marker filters, and working directories. The path filters under-
or over-approximate the real import dependencies, and adding a new
workspace package requires authoring (and forgetting to update) yet
another YAML. We already have an analyzer (`infra/select_tests.py`)
that computes a precise per-package test matrix from any diff, with a
top-level-only AST filter that respects the codebase's "no lazy
imports" convention. The next step is to delete the per-package YAMLs
and let one orchestrator drive all unit tests, with a small set of
defaults baked in workspace-wide and a tiny `[tool.marin.tests]` table
in only those packages that genuinely diverge. End state:
`marin-unit.yaml` + `marin-integration.yaml` cover the python lib
packages; `dupekit-unit.yaml` keeps the Rust+Python hybrid;
`levanter-tpu-tests.yaml` extracts the Docker-TPU job. Five workflow
files deleted, one rewritten.

This is also a **cleanup pass.** Several of the per-package quirks I
cataloged turn out to be cruft, not real differences — Ray was removed
from the codebase, iris doesn't actually need `-n1`, packages have
drifted onto different Python versions for no real reason. The design
deletes those before encoding the rest.

## Background

The seven workflows diverge on Python version (3.11/3.12/3.11–13
matrix), `uv sync` extras/groups, markers, parallelism, and per-package
quirks. After auditing each one, the genuine differences (after
deleting cruft — see `spec.md` §7) are: levanter needs the CPU-torch
wheel + Node 22; marin needs `cpu`/`dedup` extras + `-n 4`; iris wants
`PYTHONASYNCIODEBUG=1`; haliax has a JAX runtime pin under review;
fray uses a non-`test` group name. Every other package collapses to
the workspace baseline. `infra/select_tests.py` already emits a
`{run_all, matrix}` JSON ready to drive a matrix job.

Prior art (Bazel `py_test`, Pants `python_tests`, Nx `project.json`,
Turborepo `turbo.json`) all converge on declarative per-target config,
but every one of them lets the *baseline* do the work — per-target
config is for what differs, not what's the same. Coverage-based test
selection (testmon-style) loses on cache-invalidation. Full digest in
`research.md`.

## Challenges

**Avoiding scope creep into "preserve everything verbatim."** The
seven workflows have accreted quirks over time and several of them no
longer matter. The temptation is to encode all of them as schema
fields. The right move is to delete what's dead first
(`RAY_ENABLE_UV_RUN_RUNTIME_ENV`, the no-cd-zephyr rationale, iris's
`-n1`, possibly the haliax JAX `--with` pin) and let the schema only
carry differences that are actually load-bearing.

**The orchestrator is on the critical path for every PR.** Today, six
of the seven workflows skip themselves entirely if the path filter
doesn't match in ~5 seconds. The new orchestrator runs analyzer +
matrix-emit on every PR, including docs-only ones. The analyzer takes
~3s once `uv sync --group dev` has installed `grimp`; the matrix can
be empty so the wall-clock cost on a docs-only PR is `uv sync` + 3s.
That's a real ~30–60s floor regression on docs-only PRs.

## Costs / Risks

- **Migration churn.** Five workflow files deleted, one rewritten,
  five small Phase 0 cleanup PRs land first. Branch-protection rules
  currently name jobs like `levanter-unit` / `iris-unit`; an admin
  must remove those and add the new `marin-unit` aggregate as a
  required check (sequenced — see `spec.md` §10).
- **`grimp` is a hard CI dependency now.** If the analyzer fails (a
  transitive import error, a malformed package) every PR's unit tests
  are dead. The pressure-relief is a `--force-run-all` flag on
  `select_tests.py` plus a `workflow_dispatch` input on
  `marin-unit.yaml` so a human can bypass the analyzer in a panic.
- **Cold-start floor on every PR.** ~30–60s `uv sync --group dev` +
  grimp graph build, even when the resulting matrix is empty. Net
  regression on docs-only PRs.
- **Levanter sub-suite parallelism is lost.** Today `levanter-unit`,
  `levanter-entry`, and `levanter-torch` run on three runners
  concurrently; collapsed into one matrix leg they serialize. The
  AST-driven test selection covers correctness; the loss is wall-clock.

## Design

**Workspace baseline in the root `pyproject.toml`** (one place, owns
Python version, default markers, default pytest args, default `uv sync
--group test`):

```toml
[tool.marin.tests.workspace]
python = "3.12"
markers = "not slow and not tpu and not tpu_ci"
pytest_args = ["--durations=5", "-n", "auto", "--tb=short"]
sync_groups = ["test"]
```

**Per-package overrides** are optional. Most packages need no table.
Only the genuinely-different packages declare a `[tool.marin.tests]`
in their `lib/<pkg>/pyproject.toml`:

```toml
# lib/levanter/pyproject.toml
[tool.marin.tests]
sync_extras = ["torch_test"]
sync_extra_args = ["--no-install-package", "torch"]
setup_scripts = ["infra/test_setup/install_torch_cpu.sh",
                 "infra/test_setup/install_ffmpeg_apt.sh"]
setup_node = "22"
```

After Phase 0 cleanup the actual per-package configs are tight:
**four packages need no table at all** (rigging, finelog, zephyr,
fray), **three or four carry a small delta** (iris, levanter, marin,
and possibly haliax — its JAX `--with` pin is on the chopping block),
**zero declare a Python version, working directory, or sync group**
(workspace pins one Python; everyone runs from repo root; everyone
syncs with `--group test`). Full table in `spec.md` §4.

**Single orchestrator: `marin-unit.yaml`.** Three jobs, with
`fail-fast: false` and a workflow-level `concurrency: marin-unit-<pr>`
that cancels in-flight runs on push:

1. `select` — checks out, `uv sync --group dev` (installs grimp), runs
   `python infra/select_tests.py --emit-github-output` against the
   resolved base ref. Base ref is `pull_request.base.sha` on PR,
   `event.before` on push to main, or `--force-run-all` when the
   workflow_dispatch input asks. Emits the matrix as a step output.
2. `unit` — `strategy.matrix.include: ${{ fromJSON(...) }}` consumes
   the matrix. Each leg invokes `infra/run_tests.py prepare <package>`
   which loads the resolved `TestsConfig` (workspace baseline merged
   with any per-package override) and writes two scripts to
   `$RUNNER_TEMP/leg/` (setup.sh and pytest.sh) — outputs only carry
   scalars (sync args, env-as-`KEY=VAL` lines). The leg sets up Python
   via `uv python install`, optionally runs `actions/setup-node`,
   lifts env into `$GITHUB_ENV`, runs setup.sh, `uv sync ...`, then
   pytest.sh. Always from repo root; no `cd`. If the analyzer's
   `tests` list is non-empty, pytest runs on that explicit list
   (precise); empty list = forced full suite.
3. `aggregate` — depends on `unit`, exits 0 on success or skipped
   (empty matrix is fine — no test-affecting changes), 1 otherwise.
   Single status check `marin-unit` is what branch protection
   requires.

The "writes scripts to a temp dir" pattern is deliberate: GitHub
Actions step outputs cannot reliably carry multi-line shell, and
levanter's torch wheel install is genuinely 25 lines.

**`marin-integration.yaml` absorbs `iris-e2e-smoke`.** Playwright +
Claude-screenshot-verification isn't unit-shaped (boots a server,
drives a real browser, calls an external API). `dupekit-unit.yaml`
stays as-is — Rust+Python hybrid doesn't fit the analyzer.
`levanter-tpu-tests.yaml` extracts the Docker-TPU job from
`levanter-unit.yaml:159-227` unchanged; folds into the unified policy
later when it can run on the Iris prod cluster.

**Sub-suites collapse.** Levanter's three Python sub-jobs (unit /
entry / torch) become one matrix leg with one marker filter. The
AST-driven test selection is the safety net the marker-split used to
be: a torch-only test only fires when its imported source appears in
the affected set.

## Testing

The analyzer is already covered by 25 unit tests in
`tests/infra/test_select_tests.py`. Two new test surfaces:

- **A baseline + override resolution test.** Loads the workspace
  baseline and each package's optional `[tool.marin.tests]`, asserts
  required fields are present, types check, and the merged
  `TestsConfig` is what we expect for known packages
  (`tests/infra/test_marin_tests_config.py`). Failing this test in CI
  prevents merging a malformed table or a missing baseline.
- **Renderer test.** Asserts the shell scripts produced by
  `infra/run_tests.py prepare <pkg>` for fixture configs match
  expected content (`tests/infra/test_run_tests.py`).

Migration is staged so each phase is reversible until the last (full
detail in `spec.md` §10): five Phase 0 cleanup PRs delete the cruft
first; Phase 1 lands the orchestrator behind `continue-on-error`;
Phase 2 makes it a non-required parallel check next to the old
workflows; Phase 3 deletes the old workflows in a single PR and an
admin updates branch-protection in three steps.

## Open Questions

- **Phase 0 scope.** `spec.md` §7 lists six cleanup steps the design
  treats as obvious wins (Ray refs gone, `-n1` gone, Python unified at
  3.12, every package's test deps move to a uniform `test` group,
  haliax JAX `--with` re-evaluated, redundant `-c pyproject.toml`
  gone). Anything in that list someone wants to defend?
- **haliax's runtime JAX pin.** `--with "jax[cpu]==0.9.2"` overrides
  the lockfile. Is this an intentional "test haliax against the JAX
  version levanter uses" check? If yes, tighten the dep in
  `lib/haliax/pyproject.toml` and drop the override. If no, just drop
  it. If neither and reviewers want it preserved, `uv_run_args` carries
  it.
- **Cold-start cost on docs-only PRs.** ~30–60s floor on every PR. Worth
  it for the precision wins, but I want a sanity check that the docs/
  PR experience won't deteriorate enough to bother people.
- **Branch-protection runbook.** Phase 3 needs a three-step admin
  sequence (remove old required checks → merge switch PR → add new
  required check). Should this happen with someone in the room, or do
  we trust the runbook?
- **Should the analyzer become a reusable composite action?** Inline in
  the orchestrator keeps blast radius small; composite would let other
  workflows consult the matrix. Starting inline; reconsider if a second
  consumer shows up.
