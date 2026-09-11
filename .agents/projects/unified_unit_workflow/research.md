# Research — unified_unit_workflow

In-repo and prior-art notes that shaped the design. Written before the
1-pager; left here as a sibling so reviewers can dig if they want depth.

## In-repo: today's seven `*-unit.yaml` workflows

All seven follow the same shell — `dorny/paths-filter` gates the run, then
one or more jobs run `uv sync` and `pytest`. They diverge sharply on the
details:

| Workflow | Python | `uv sync` | `pytest` quirk | Sub-jobs |
|---|---|---|---|---|
| `haliax-unit.yaml:32-55` | 3.11 | `--package marin-haliax --dev` | `--with "jax[cpu]==0.9.2"` runtime pin | none |
| `levanter-unit.yaml:33-226` | 3.11 | `--package marin-levanter --dev --group test --frozen` (+`--extra torch_test --no-install-package torch` for torch leg) | three marker-filtered legs (`unit`/`entry`/`torch`); torch leg installs CPU torch from the pytorch wheelhouse via uv.lock version extraction (lines 119-146); `levanter-tpu-tests` runs in a docker container with TPU device mounts | `levanter-unit`, `levanter-entry`, `levanter-torch`, `levanter-tpu-tests` |
| `iris-unit.yaml:34-131` | 3.12 | `--group dev` | `cd lib/iris && pytest -n1` | `iris-unit`, `iris-e2e-smoke` (Playwright + Claude screenshot verify) |
| `zephyr-unit.yaml:33-67` | 3.12 | `--package marin-zephyr --group test --frozen` | "**don't cd into lib/zephyr** so that ray uv integration doesn't freak out" (line 65) | none |
| `fray-unit.yaml:32-54` | 3.12 | `--group=fray-test` | `cd lib/fray && pytest -s` | none |
| `marin-unit.yaml:35-72` | 3.12 | `--package marin --extra cpu --extra dedup --group test --frozen` | `pytest -n 4 --dist=worksteal`; `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` to stop Ray re-uv-syncing on workers | none |
| `dupekit-unit.yaml:31-89` | 3.11 / 3.12 / 3.13 matrix | `--frozen --group test` | `cd rust/dupekit && pytest`; also runs cargo fmt/clippy/test | `check-user-mode`, `unit-test`, `rust-checks` |

Every workflow's path filter is the same template (`lib/<pkg>/**`, `uv.lock`,
`.github/workflows/<pkg>-*.yaml`) plus a few cross-package adds (zephyr
also fires on `lib/iris/**`, `lib/fray/**`; levanter fires on
`lib/haliax/**`). Five of seven are missing transitive dependencies — see
the audit table earlier in the conversation.

## In-repo: the analyzer that already exists

`infra/select_tests.py:243` is the entry point. It enumerates module names
via `grimp.build_graph`, AST-scans every source file's *top-level* imports
(skipping `def`/`class`/`if TYPE_CHECKING:`), builds a reverse-dep map,
walks downstream of each changed module, and AST-scans test files for any
import that touches the affected set. Output:

```json
{"run_all": false, "reason": "diff-driven",
 "matrix": [{"package": "levanter", "tests": ["lib/levanter/tests/test_metrics.py"]}, ...]}
```

Broad triggers (`uv.lock`, root `pyproject.toml`, the analyzer file
itself, unrecognised workflow files) short-circuit to `run_all: true`.
Test coverage in `tests/infra/test_select_tests.py` (25 tests) covers the
broad triggers, no-op cases, direct test-file edits, forced full-suite
behaviour, the lazy-import filter, and helper functions. Lives at
`/Users/power/code/marin/.worktrees/marin-test-deps/infra/select_tests.py`.

## In-repo: composite actions and lint

`.github/actions/` has two composites: `claude-triage/action.yaml`
(canary-triage skill wrapper) and `notify-slack/action.yaml`. Neither is
unit-test relevant. `infra/pre-commit.py:1-735` is the unified linter
(ruff/black/license/pyrefly/...); `marin-lint.yaml:1-34` is a thin
wrapper that just calls it on `--all-files`. Lint is already
consolidated; tests are the outlier.

No prior design docs in `.agents/projects/` cover test selection,
workflow consolidation, or build-graph orchestration.

## Prior art

Four established systems converge on the same shape: **declarative
per-target config with a small typed schema**.

- **Bazel `py_test`** — fields: `srcs`, `deps`, `data`, `args`, `env`,
  `tags`, `size`, `timeout`, `shard_count`, `flaky`. Test target is a
  first-class typed BUILD record. ([reference](https://bazel.build/reference/be/python))
- **Pants `python_tests`** — superset of Bazel's fields plus
  `extra_env_vars`, `batch_compatibility_tag`, an `env` field naming a
  defined environment. ([reference](https://www.pantsbuild.org/stable/reference/targets/python_tests))
- **Nx `project.json` `targets.test`** — `executor`, `options`,
  `outputs`, `configurations` for env overrides. JSON-Schema validated.
  Per-package file co-located with code.
- **Turborepo `turbo.json` `tasks.test`** — `dependsOn`, `inputs`,
  `outputs`, `env`, `cache`. Centralised at root by default; per-package
  overrides via package-local `turbo.json` that extends root.
  ([reference](https://turborepo.dev/docs/reference/configuration))

None pick "pure convention." All four pick declarative + typed. The
schemas are small (5-10 fields). Per-package locality wins (Bazel,
Pants, Nx) over central config (Turborepo).

`pytest-testmon` and `pytest-incremental` offer line-level test
selection from coverage history. The recurring complaint in blog
postmortems is cache-invalidation bugs that hide regressions — teams
revert to package-level granularity. **Implication:** keep our analyzer
coarse (package-level test selection within each suite); declarative
config stays small.

`workflow_call` + matrix is the de facto GitHub Actions monorepo
pattern: a generator job emits a JSON matrix, a downstream job consumes
`strategy.matrix` and dispatches one leg per entry. Marin's analyzer
already produces matrix-shaped output, so the orchestrator is largely a
glue job.

## Q&A summary

User decisions captured during interrogate (2026-05-09):

- **Config home**: `[tool.marin.tests]` in each `lib/<pkg>/pyproject.toml`. Co-located with the package; matches the established Bazel/Nx pattern.
- **Sub-suites (levanter unit/entry/torch)**: collapse into one job per package. The AST analyzer already filters tests narrowly; per-marker job splits no longer earn their orchestration cost.
- **iris-e2e-smoke**: move into `marin-integration.yaml`. Browser + external-API verification isn't unit-shaped.
- **dupekit-unit.yaml**: keep as-is. Rust+Python hybrid doesn't fit the analyzer (which only sees Python imports), and folding cargo into either marin-unit or marin-lint blurs the boundary.

End state: `marin-unit.yaml`, `marin-integration.yaml`, `dupekit-unit.yaml`, `levanter-tpu-tests.yaml` (the TPU job split out), plus the existing `marin-lint.yaml` and the canary/release workflows that aren't in scope. Five-and-change unit/test workflows down from twelve-ish.
