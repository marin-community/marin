# Workflow Scripts

Marin's GitHub Actions workflows should be thin triggers around local, reproducible workflow programs. Today, the same operational logic appears as long YAML blocks across smoke tests, canaries, ferries, wheel publishing, diagnostics, and issue/PR automation, which makes failures harder to reproduce locally and makes workflow names hard to reason about in the Actions UI. This design introduces `scripts/workflows/` as the home for workflow-owned Python CLIs, standardizes workflow/job naming, and sets a policy for when GitHub Actions YAML is allowed to contain logic.

Success means every workflow has a documented local reproduction command, new workflow logic is covered by the `scripts/workflows` audit, all non-allowlisted third-party actions are SHA-pinned, and the copied Iris polling/diagnostics blocks are removed from the smoke, ferry, and integration workflows. The migration should reduce long inline `run: |` blocks to short workflow glue, not merely move names around.

## Challenges

The difficult part is not moving shell into Python one block at a time; it is drawing boundaries that stay useful after the first cleanup PR. Some workflow behavior is repo-local and should run inside the Marin `uv run` environment, while other operational tools are deliberately standalone uv scripts and should remain usable without installing the repo. GitHub Actions also has native reuse tools, especially reusable workflows, but they reuse job topology rather than making behavior locally reproducible. We need a design that uses Python scripts for behavior first, then only adds reusable workflows when duplicated job structure remains after the scripts exist.

The second challenge is naming. Issue #5067 proposes `type-domain-test`; prior triage noted that most existing files already start with the domain. This design chooses domain-first naming because it keeps team-owned workflows adjacent in `ls`, GitHub's Actions sidebar, and branch-protection discussions.

## Costs / Risks

- Renaming workflow files, workflow names, or job names can affect required-check configuration. The cleanup needs an explicit branch-protection coordination step before changing any status context that gates merges.
- Moving scripts under stricter linting and type checking will create some initial friction, especially because `scripts/**` is currently excluded from ruff and pyrefly.
- Replacing third-party logic actions with local scripts creates code Marin owns and must maintain. That is a good trade for load-bearing automation, but not for basic primitives like checkout or tool setup.
- This is mostly infrastructure hygiene. The immediate user-visible benefit is faster diagnosis and more reliable local reproduction, not new product behavior.

## Design

Create `scripts/workflows/` for Python CLIs that implement behavior currently embedded in `.github/workflows/*.yaml`. Workflow YAML remains responsible for triggers, permissions, runner selection, matrices, secrets binding, artifact upload/download, and calling scripts. Long `run:` blocks should become one-line invocations unless the block is only shell glue that GitHub Actions itself owns.

Prefer the `gh` CLI for GitHub state changes (creating PRs, opening issues, posting comments, editing labels, cutting releases, querying branch protection). `gh` is pre-installed and pre-authenticated on GitHub-hosted runners, works the same way locally, and removes a class of Marin-owned wrapper code. A short documented `git + gh pr create/edit` snippet replaces `peter-evans/create-pull-request` without introducing a Python helper. Reach for a Python script only when the behavior involves real logic that benefits from tests: path/glob matching across many groups, YAML auditing, subprocess polling with timeout and JSON parsing, multi-step provider-specific diagnostics. The bar is "would this be ugly or untestable in 10–15 lines of `gh` + shell." If the answer is no, keep it in YAML and call `gh` directly.

The default execution model for the Python scripts is repo-local:

```bash
uv run python scripts/workflows/<domain>_<command>.py <subcommand> ...
```

Repo-local scripts may import Marin workspace packages and shared helpers. This is the right default for Iris, Zephyr, Marin, Levanter, Haliax, and Fray workflows because their behavior usually depends on checked-out repo code and the locked workspace environment. Standalone uv scripts remain appropriate for tools that do not depend on the repo, such as `scripts/logscan.py` and operational notification utilities like `scripts/ops/discord.py`; those can keep PEP 723 metadata and `#!/usr/bin/env -S uv run --script` when that is what makes them portable.

All new `scripts/workflows/**` CLIs use Click. Each script exposes a small set of explicit commands and options, writes machine-readable outputs when a workflow step needs them, and keeps computation separate from I/O where practical. `scripts/rust_package.py` is the nearest existing model for workflow-owned logic: it documents local usage and prerequisites, owns subprocess orchestration in Python, and is called from `.github/workflows/dupekit-wheels.yaml` instead of embedding build/release logic in YAML. The main differences for new workflow scripts are Click instead of argparse and stricter lint/type expectations. See [research.md](research.md) for file references and prior art.

Workflow names use:

```text
<domain>-<type>[-<variant>]
```

`domain` is the owning area: `iris`, `zephyr`, `marin`, `levanter`, `haliax`, `fray`, `dupekit`, or `ops`. `ops` is the bucket for repository automation such as stale issue handling, CodeQL, nightshift jobs, Claude automation, and workflow maintenance. `type` is the gate or automation kind: `unit`, `smoke`, `canary`, `integration`, `release`, `lint`, `docs`, or `dev`. `variant` is optional and should only remain when a matrix or script option cannot express the difference cleanly.

Examples:

- `iris-smoke` for a cross-provider Iris smoke workflow, with `cluster=gcp|coreweave` as a matrix or script option.
- `iris-smoke-gcp` only if the GCP implementation must remain a separate status context.
- `marin-canary-ferry` for the current Marin canary ferry.
- `zephyr-integration-shuffle` instead of `zephyr-shuffle-itest`.
- `haliax-unit` instead of `haliax-run_tests`.
- `levanter-integration-gpt2-small` instead of `levanter-gpt2_small_itest`.
- `dupekit-release-wheels` for wheel building and publishing.
- `ops-stale`, `ops-codeql`, or `ops-nightshift-docs` for repository automation that is not owned by a product domain.

Job names should follow the same shape when they create branch-protection contexts, with a short action suffix if needed: `iris-smoke / test (gcp)`, `dupekit-release / build (linux)`, `marin-canary-ferry / validate`. Step names should be imperative and stable: `Submit Iris job`, `Wait for Iris job`, `Collect Iris diagnostics`, `Create update PR`.

Third-party actions are allowed in two tiers. A small explicit allowlist of trusted primitives may remain tag-pinned when that is already the project norm: checkout, setup, cache, artifact upload/download, CodeQL, GitHub app token creation, uv setup, Google auth/setup-gcloud, and Docker setup/login/build actions. All other third-party actions should be pinned to a full commit SHA. This follows GitHub's security guidance that SHA pinning is the only immutable reference for third-party actions and that tag pinning is a trust decision.

The largest duplicated behavior is Iris job orchestration: submit/wait/status inspection and failure diagnostics are embedded across canary, smoke, and integration workflows. The first operational extraction is a single `scripts/workflows/iris_monitor.py` with `status`, `wait`, and `collect` subcommands, not a reusable workflow. Wait and diagnostics share Iris CLI plumbing and are always invoked as a pair (wait, then collect on failure), so one module is the right shape. Before that operational PR lands, the foundation PR creates the script/audit surface and converts lower-risk workflows so the live-infrastructure extraction has a stable pattern to follow.

`scripts/workflows/**` should be included in ruff and pyrefly even though broader `scripts/**` remains excluded today. These scripts are CI infrastructure, not one-off local utilities, and breakage affects the development loop for everyone.

Do not create a shared workflow package on day one. Start with concrete importable modules such as `scripts/workflows/iris_monitor.py`, `scripts/workflows/changes.py`, and `scripts/workflows/github_actions.py`. Add `scripts/workflows/lib/` only after at least two modules need the same non-trivial helper and the helper has a stable contract. The repo currently has `scripts/__init__.py`, but not a meaningful scripts package hierarchy; premature packaging would add more structure than the first extraction needs.

The full migration should land as a small number of large PRs, not a long tail of tiny workflow edits:

1. **Foundation and low-risk normalization.** Add `scripts/workflows/`, `.github/workflows/README.md` (including the canonical `git + gh pr create/edit` recipe), the workflow inventory/audit script, path-change filtering, and SHA pinning for non-trusted third-party actions. Replace `peter-evans/create-pull-request@v7` in `dupekit-wheels.yaml` with an inline `gh` snippet rather than a Python wrapper. Delete `marin-metrics.yaml` (dead conda-based weekly job) instead of migrating it; this also eliminates the only `conda-incubator/setup-miniconda` pin from the repo. Bring `scripts/workflows/**` under ruff and pyrefly. Convert unit/docs/lint/release workflows that mostly run tests or packaging scripts, because they do not launch live infrastructure.
2. **Iris and ferry behavior extraction.** Add `iris_monitor.py` with `status`, `wait`, and `collect` subcommands, then migrate `iris-cloud-smoke-gcp`, `iris-coreweave-ci`, `marin-canary-ferry*`, `marin-datakit-*`, and `zephyr-shuffle-itest` to call it. This is one large operational PR because these workflows share the same failure modes and should be reviewed together.
3. **Names, file renames, and consolidation.** Rename workflow files and displayed workflow/job names to `domain-type[-variant]`, update branch-protection contexts with `gh api` where required, and consolidate workflows only where the scripts have made the provider/domain differences parameterizable. This is the right point to decide whether `iris-smoke-gcp` and `iris-smoke-coreweave` become one `iris-smoke` workflow.

## Testing

Each workflow script gets focused pytest coverage for argument parsing, output shape, and pure computation. I/O-heavy paths should be tested at the boundary with temporary files and subprocess fakes, not by mocking internal helper calls. Scripts that wrap Iris or GitHub should include a dry-run or fixture-driven mode where practical so CI can test command construction and status parsing without launching live infrastructure.

The rollout test is staged. First, land scripts that can be run locally while workflows still call the old YAML blocks. Second, switch one low-risk workflow step to call the script and compare logs/output shape against the previous run. Third, migrate repeated blocks across workflows.

Branch-protection-sensitive renames should be handled with `gh`, not manual clicking. `gh api repos/marin-community/marin/branches/main/protection` currently exposes the required contexts for `main`: `lint-and-format`, `build-docs`, `marin-tests`, `levanter-tests`, `levanter-entry-tests`, `levanter-torch-tests`, `haliax-tests`, `iris-tests`, `zephyr-tests`, `fray-tests`, and `marin-itest`. If a PR changes any required job context, the rollout first adds the target contexts while keeping the old ones, merges only after the renamed checks appear and pass, then removes the old contexts after the renamed workflows have run on `main`.

## Open Questions

- Should `iris-smoke-gcp` and `iris-smoke-coreweave` eventually merge into one `iris-smoke` workflow after script extraction, or stay separate because the providers are operationally different enough to deserve separate status contexts?
