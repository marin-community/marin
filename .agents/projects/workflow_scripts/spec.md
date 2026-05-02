# Workflow Scripts Spec

This spec defines the target contracts for moving Marin's GitHub Actions workflows to the workflow-scripts model. The implementation can land in large staged PRs, but each stage must preserve the contracts below.

## File Layout

| Path | Contract |
| --- | --- |
| `scripts/workflows/__init__.py` | Empty package marker for workflow-owned Python modules. No exported convenience API. |
| `scripts/workflows/changes.py` | Repo-local replacement for path-filter logic used by unit/docs/lint workflows. |
| `scripts/workflows/pull_request.py` | Repo-local replacement for third-party create-PR workflow logic. |
| `scripts/workflows/github_actions.py` | Workflow inventory and policy checks: naming, `.yaml`, third-party SHA pins, and required-status inventory. |
| `scripts/workflows/iris_job.py` | Iris job status, wait, and GitHub Actions output helpers. |
| `scripts/workflows/iris_diagnostics.py` | Failure diagnostics collection for Iris-backed workflows. |
| `.github/workflows/README.md` | Human index of workflows: target name, trigger, gate type, owner domain, and local reproduction command. |
| `pyproject.toml` | Includes `scripts/workflows/**` in ruff and pyrefly while leaving unrelated legacy scripts excluded. |

Do not add `scripts/workflows/lib/` in the first migration. If two or more modules need the same non-trivial helper, add a concrete helper module named for the concept, not a generic utility module. Docker image build orchestration can be added later if the cleanup shows real duplicated behavior after trusted Docker setup/login/build actions remain in YAML.

## CLI Contracts

All `scripts/workflows/*.py` modules are repo-local commands invoked as:

```bash
uv run python scripts/workflows/<name>.py <command> [options]
```

All new workflow scripts use Click. Commands must fail non-zero on operational failure, print concise human-readable progress to stderr, and write machine-readable data either to stdout or to `$GITHUB_OUTPUT` when `--github-output` is supplied. Commands that can affect GitHub state must support `--dry-run`.

### `changes.py`

```python
@dataclass(frozen=True)
class PathDecision:
    name: str
    matched: bool
    paths: tuple[str, ...]


def changed_paths(base_ref: str, head_ref: str, *, repo: Path) -> tuple[str, ...]:
    """Return paths changed between two refs, sorted and relative to repo root.

    Uses `git diff --name-only --diff-filter=ACMRTUXB base_ref...head_ref` for pull requests
    and `base_ref..head_ref` for push comparisons. Raises ValueError when either ref is missing.
    """


def match_groups(paths: Iterable[str], groups: Mapping[str, Sequence[str]]) -> tuple[PathDecision, ...]:
    """Match changed paths against named glob groups.

    Patterns use pathlib-style `PurePath.match` semantics with `**` support. Negated patterns are
    accepted with a leading `!` and are applied after positive patterns in declaration order.
    """
```

CLI:

```bash
uv run python scripts/workflows/changes.py match \
  --base "$BASE_SHA" \
  --head "$HEAD_SHA" \
  --group marin='lib/marin/**,tests/**,pyproject.toml,uv.lock' \
  --github-output
```

Output contract:

- stdout JSON: `{"groups": [{"name": "marin", "matched": true, "paths": ["..."]}]}`
- `$GITHUB_OUTPUT`: one lower-case boolean per group, e.g. `marin=true`

For `pull_request` events, callers pass the PR base SHA and head SHA and the command uses merge-base diff semantics. For `push` events, callers pass before/after SHAs and the command uses direct range semantics. For `schedule` and `workflow_dispatch`, workflows must either skip `changes.py` or pass `--always-match`, which marks every group as matched and records `reason="manual-or-scheduled"` in stdout JSON. Deleted-only files do not trigger groups because the command excludes `D` from the diff filter; renames trigger on the new path.

### `pull_request.py`

```python
@dataclass(frozen=True)
class PullRequestRequest:
    branch: str
    title: str
    body: str
    commit_message: str
    labels: tuple[str, ...]
    base: str = "main"
    author_name: str = "github-actions[bot]"
    author_email: str = "41898282+github-actions[bot]@users.noreply.github.com"
    draft: bool = False


def create_or_update_pull_request(request: PullRequestRequest, *, repo: str, dry_run: bool) -> str | None:
    """Commit current worktree changes, push `request.branch`, and create or update a pull request.

    Returns the PR URL, or None in dry-run mode. Raises ValueError for an empty diff or invalid request.
    """
```

CLI:

```bash
uv run python scripts/workflows/pull_request.py create \
  --branch auto/update-dupekit-wheels \
  --title "chore: update dupekit wheels" \
  --body-file pr-body.md \
  --commit-message "chore: pin dupekit wheels to $GITHUB_SHA" \
  --label agent-generated
```

This command replaces `peter-evans/create-pull-request@v7` in `dupekit-wheels.yaml` and any future in-workflow PR creation.

Supported behavior is intentionally narrower than the third-party action: commit the current worktree diff, push or force-with-lease the named branch, create a PR if none exists, update title/body/labels on an existing open PR from the same branch, and exit zero with `pr_created=false` when the diff is empty. Unsupported behavior: reviewers, assignees, branch deletion, signed commits, path-scoped commits, multi-commit preservation, and automatic base-branch rebasing. The workflow must grant `contents: write` and `pull-requests: write`.

### `github_actions.py`

```python
@dataclass(frozen=True)
class WorkflowRecord:
    path: Path
    workflow_name: str
    jobs: tuple[WorkflowJob, ...]
    third_party_actions: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowJob:
    job_id: str
    job_name: str | None
    matrix_context: str | None
    required_context: str | None


def workflow_records(workflows_dir: Path) -> tuple[WorkflowRecord, ...]:
    """Parse workflow files and return workflow names, jobs, matrix context shapes, and action refs."""


def required_status_contexts(repo: str, branch: str) -> tuple[str, ...]:
    """Return branch-protection required status contexts using the GitHub API."""
```

CLI:

```bash
uv run python scripts/workflows/github_actions.py audit --workflows-dir .github/workflows
uv run python scripts/workflows/github_actions.py required-contexts --repo marin-community/marin --branch main
```

Audit failures:

- workflow file does not end in `.yaml`
- workflow name does not follow `Domain - Type [- Variant]`
- job id does not follow lower-case kebab-case
- non-trusted third-party action is not pinned to a 40-character SHA
- required branch-protection context is missing from the workflow inventory

Trusted tag-pinned actions:

- `actions/`
- `github/codeql-action/init`
- `github/codeql-action/analyze`
- `actions/checkout`
- `actions/setup-python`
- `actions/setup-node`
- `actions/cache`
- `actions/upload-artifact`
- `actions/download-artifact`
- `actions/create-github-app-token`
- `astral-sh/setup-uv`
- `google-github-actions/auth`
- `google-github-actions/setup-gcloud`
- `docker/setup-buildx-action`
- `docker/login-action`
- `docker/build-push-action`

All other external actions must be SHA-pinned.

### `iris_job.py`

```python
class IrisJobState(StrEnum):
    PENDING = "JOB_STATE_PENDING"
    BUILDING = "JOB_STATE_BUILDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLED = "JOB_STATE_CANCELLED"


@dataclass(frozen=True)
class IrisJobStatus:
    job_id: str
    state: IrisJobState
    error: str | None


def job_status(job_id: str, *, iris_config: Path | None, prefix: str | None = None) -> IrisJobStatus:
    """Return the exact Iris job status by running `iris job list --json --prefix <prefix>`."""


def wait_for_job(
    job_id: str,
    *,
    iris_config: Path | None,
    prefix: str | None,
    poll_interval: int,
    timeout: int | None,
) -> IrisJobStatus:
    """Poll until the Iris job reaches a terminal state.

    `poll_interval` and `timeout` are seconds. Raises TimeoutError on timeout and RuntimeError for
    FAILED, CANCELLED, missing job, malformed JSON, or Iris CLI failure.
    """
```

CLI:

```bash
uv run python scripts/workflows/iris_job.py wait \
  --job-id "$JOB_ID" \
  --iris-config "$IRIS_CONFIG" \
  --poll-interval 30 \
  --github-output
```

The command shells out to `.venv/bin/iris` when present, otherwise `uv run iris`. It passes `--config <iris_config>` when supplied, selects the exact `job_id` from JSON output, and treats unknown terminal states as failure. `--github-output` writes `job_id`, `state`, and `succeeded` before exiting on both success and known terminal failure; it may be absent only when the command cannot parse status at all. `SIGINT` and `SIGTERM` stop polling and exit non-zero without cancelling the remote job.

### `iris_diagnostics.py`

```python
@dataclass(frozen=True)
class DiagnosticsRequest:
    job_id: str
    output_dir: Path
    iris_config: Path | None
    provider: Literal["gcp", "coreweave"]
    project: str | None
    controller_label: str | None
    namespace: str | None


def collect_diagnostics(request: DiagnosticsRequest) -> Path:
    """Collect Iris controller, job tree, and provider-specific task diagnostics into `output_dir`."""
```

CLI:

```bash
uv run python scripts/workflows/iris_diagnostics.py collect \
  --job-id "$JOB_ID" \
  --iris-config "$IRIS_CONFIG" \
  --provider gcp \
  --output-dir "$DIAG_DIR"
```

Output directory contract:

```text
<output-dir>/
  controller-process.log
  job-tree.json
  controller-<name>.log          # GCP only, when controller SSH succeeds
  kubernetes-pods.json           # CoreWeave only, when namespace is supplied
  summary.json
```

`summary.json` contains `job_id`, `provider`, `files`, `required_files`, `missing_required_files`, and `errors`.

Required artifacts:

- All providers: `job-tree.json` and `summary.json`
- GCP: at least one `controller-<name>.log`
- CoreWeave: `kubernetes-pods.json`

Diagnostics collection is best-effort for optional artifacts, but the command exits non-zero when no required provider artifact can be collected. Workflows should run the diagnostics step under `if: failure() || cancelled()` and `continue-on-error: true`, then always upload whatever files were written so diagnostics regressions are visible without hiding the original workflow failure.

## Workflow Naming Contracts

Workflow files use `.yaml` and lower-case kebab-case. Display names use title case with the same semantic parts.

| Current file | Target file | Target workflow name |
| --- | --- | --- |
| `claude-review.yml` | `ops-claude-review.yaml` | `Ops - Claude Review` |
| `claude.yml` | `ops-claude.yaml` | `Ops - Claude` |
| `docker-images.yaml` | `ops-docker-images.yaml` | `Ops - Docker Images` |
| `dupekit-unit-tests.yaml` | `dupekit-unit.yaml` | `Dupekit - Unit` |
| `dupekit-wheels.yaml` | `dupekit-release-wheels.yaml` | `Dupekit - Release Wheels` |
| `fray-unit-tests.yaml` | `fray-unit.yaml` | `Fray - Unit` |
| `grug-variant-diff.yaml` | `ops-grug-variant-diff.yaml` | `Ops - Grug Variant Diff` |
| `haliax-run_tests.yaml` | `haliax-unit.yaml` | `Haliax - Unit` |
| `iris-cloud-smoke-gcp.yaml` | `iris-smoke-gcp.yaml` | `Iris - Smoke - GCP` |
| `iris-coreweave-ci.yaml` | `iris-smoke-coreweave.yaml` | `Iris - Smoke - CoreWeave` |
| `iris-dev-restart.yaml` | `iris-dev-restart.yaml` | `Iris - Dev Restart` |
| `iris-iap-proxy.yaml` | `iris-release-iap-proxy.yaml` | `Iris - Release IAP Proxy` |
| `iris-unit-tests.yaml` | `iris-unit.yaml` | `Iris - Unit` |
| `levanter-gpt2_small_itest.yaml` | `levanter-integration-gpt2-small.yaml` | `Levanter - Integration - GPT-2 Small` |
| `levanter-launch_small_fast.yaml` | `levanter-dev-launch-small-fast.yaml` | `Levanter - Dev - Launch Small Fast` |
| `levanter-tests.yaml` | `levanter-unit.yaml` | `Levanter - Unit` |
| `marin-canary-ferry-cw.yaml` | `marin-canary-ferry-coreweave.yaml` | `Marin - Canary Ferry - CoreWeave` |
| `marin-canary-ferry.yaml` | `marin-canary-ferry.yaml` | `Marin - Canary Ferry` |
| `marin-codeql.yml` | `ops-codeql.yaml` | `Ops - CodeQL` |
| `marin-datakit-nemotron-ferry.yaml` | `marin-canary-datakit-nemotron.yaml` | `Marin - Canary - Datakit Nemotron` |
| `marin-datakit-smoke.yaml` | `marin-smoke-datakit.yaml` | `Marin - Smoke - Datakit` |
| `marin-docs.yaml` | `marin-docs.yaml` | `Marin - Docs` |
| `marin-infra-dashboard.yaml` | `ops-infra-dashboard.yaml` | `Ops - Infra Dashboard` |
| `marin-itest.yaml` | `marin-integration.yaml` | `Marin - Integration` |
| `marin-libs-wheels.yaml` | `marin-release-libs-wheels.yaml` | `Marin - Release Libs Wheels` |
| `marin-lint-and-format.yaml` | `marin-lint.yaml` | `Marin - Lint` |
| `marin-metrics.yaml` | `marin-dev-metrics.yaml` | `Marin - Dev Metrics` |
| `marin-unit-tests.yaml` | `marin-unit.yaml` | `Marin - Unit` |
| `nightshift-cleanup.yml` | `ops-nightshift-cleanup.yaml` | `Ops - Nightshift Cleanup` |
| `nightshift-doc-drift.yml` | `ops-nightshift-doc-drift.yaml` | `Ops - Nightshift Doc Drift` |
| `stale.yml` | `ops-stale.yaml` | `Ops - Stale` |
| `zephyr-shuffle-itest.yaml` | `zephyr-integration-shuffle.yaml` | `Zephyr - Integration - Shuffle` |
| `zephyr-unit-tests.yaml` | `zephyr-unit.yaml` | `Zephyr - Unit` |

Required status contexts are job contexts, not file names. Any rename of required job ids must include a branch-protection update for `main`. Current required contexts are:

```text
lint-and-format
build-docs
marin-tests
levanter-tests
levanter-entry-tests
levanter-torch-tests
haliax-tests
iris-tests
zephyr-tests
fray-tests
marin-itest
```

Target required contexts after final renaming:

```text
marin-lint
marin-docs
marin-unit
levanter-unit
levanter-entry
levanter-torch
haliax-unit
iris-unit
zephyr-unit
fray-unit
marin-integration
```

Required-context mapping:

| Current workflow | Current job id / context | Target workflow | Target job id / context |
| --- | --- | --- | --- |
| `marin-lint-and-format.yaml` | `lint-and-format` | `marin-lint.yaml` | `marin-lint` |
| `marin-docs.yaml` | `build-docs` | `marin-docs.yaml` | `marin-docs` |
| `marin-unit-tests.yaml` | `marin-tests` | `marin-unit.yaml` | `marin-unit` |
| `levanter-tests.yaml` | `levanter-tests` | `levanter-unit.yaml` | `levanter-unit` |
| `levanter-tests.yaml` | `levanter-entry-tests` | `levanter-unit.yaml` | `levanter-entry` |
| `levanter-tests.yaml` | `levanter-torch-tests` | `levanter-unit.yaml` | `levanter-torch` |
| `haliax-run_tests.yaml` | `haliax-tests` | `haliax-unit.yaml` | `haliax-unit` |
| `iris-unit-tests.yaml` | `iris-tests` | `iris-unit.yaml` | `iris-unit` |
| `zephyr-unit-tests.yaml` | `zephyr-tests` | `zephyr-unit.yaml` | `zephyr-unit` |
| `fray-unit-tests.yaml` | `fray-tests` | `fray-unit.yaml` | `fray-unit` |
| `marin-itest.yaml` | `marin-itest` | `marin-integration.yaml` | `marin-integration` |

Required checks should use stable job ids as the context source. If a job also sets a display `name:`, it must either equal the job id for required jobs or be verified in GitHub before branch protection is patched. Matrix jobs must not become required contexts unless the matrix-expanded context names are explicitly listed in this table.

## Landing Sequence

The work should land in three large PRs.

### PR 1: Foundation and Low-Risk Workflows

Required content:

- Add `scripts/workflows/changes.py`, `pull_request.py`, and `github_actions.py`.
- Add `.github/workflows/README.md`.
- Update `pyproject.toml` so `scripts/workflows/**` is linted and type checked.
- Replace `dorny/paths-filter` usages in unit/docs/lint/release workflows with `changes.py`.
- Replace `peter-evans/create-pull-request@v7` in `dupekit-wheels.yaml` with `pull_request.py`.
- Pin all non-trusted third-party actions to full SHAs, including `dorny/paths-filter`, `peaceiris/actions-gh-pages`, `actions/github-script` if retained, `actions/stale`, `anthropics/claude-code-action`, and `conda-incubator/setup-miniconda`.
- Keep existing required job ids unchanged in this PR unless branch protection is updated in the same PR window.

Workflow scope:

- `dupekit-unit-tests.yaml`
- `dupekit-wheels.yaml`
- `fray-unit-tests.yaml`
- `haliax-run_tests.yaml`
- `iris-unit-tests.yaml`
- `levanter-tests.yaml`
- `marin-docs.yaml`
- `marin-lint-and-format.yaml`
- `marin-unit-tests.yaml`
- `zephyr-unit-tests.yaml`
- `grug-variant-diff.yaml`
- `stale.yml`

### PR 2: Iris, Ferries, and Live-Infrastructure Workflows

Required content:

- Add `scripts/workflows/iris_job.py` and `iris_diagnostics.py`.
- Migrate all copied Iris wait loops to `iris_job.py wait`.
- Migrate all copied GCP/CoreWeave diagnostics to `iris_diagnostics.py collect`.
- Convert workflow-specific notification shell to existing `scripts/ops/discord.py` where practical.
- Preserve provider-specific setup in YAML until the behavior is proven scriptable. Do not restart or bounce Iris clusters as part of this migration.

Workflow scope:

- `iris-cloud-smoke-gcp.yaml`
- `iris-coreweave-ci.yaml`
- `iris-dev-restart.yaml`
- `marin-canary-ferry.yaml`
- `marin-canary-ferry-cw.yaml`
- `marin-datakit-smoke.yaml`
- `marin-datakit-nemotron-ferry.yaml`
- `zephyr-shuffle-itest.yaml`
- `levanter-gpt2_small_itest.yaml`
- `levanter-launch_small_fast.yaml`

### PR 3: Names, Branch Protection, and Remaining Consolidation

Required content:

- Rename workflow files to the target table above.
- Rename displayed workflow names and non-required job ids to the target convention.
- Rename required job ids with a two-step branch-protection rollout:

1. Before merging the rename PR, add the target contexts to branch protection while keeping the current contexts. This is safe because the target contexts will be pending until the rename PR runs them, so the PR itself should not be merged until those checks appear and pass.
2. Merge the rename PR after both old and new required checks are satisfied or after maintainers confirm that old contexts are no longer emitted for that branch.
3. After the renamed workflows have run once on `main`, remove the old contexts from branch protection and leave only the target contexts.

Verification commands:

```bash
gh api repos/marin-community/marin/branches/main/protection \
  --jq '.required_status_checks.contexts'
gh pr checks <rename-pr-number> --required
```

Patch command:

```bash
gh api --method PATCH repos/marin-community/marin/branches/main/protection \
  --input branch-protection.json
```

`branch-protection.json` must preserve all existing protection settings and replace only `required_status_checks.contexts` / `required_status_checks.checks` as needed.

Rollback: if renamed workflows fail to emit the expected contexts, restore the previous workflow/job names in git and patch branch protection back to the previous context list captured from the verification command. The active `protect main` ruleset should be checked with `gh api repos/marin-community/marin/rulesets`; if required checks move into rulesets before implementation, the same add-new-before-remove-old sequence applies through the rulesets API instead of classic branch protection.

- Consolidate workflows only when scripts have made differences parameterizable. The main candidate is `iris-smoke-gcp.yaml` plus `iris-smoke-coreweave.yaml` into `iris-smoke.yaml`; this is optional and should not block naming cleanup.
- Run `github_actions.py audit` in CI so new workflows follow the model.

Workflow scope:

- All 33 workflows.

## Out of Scope

- Replacing trusted setup/auth/build primitives such as `actions/checkout`, `actions/setup-python`, `actions/cache`, `actions/upload-artifact`, `astral-sh/setup-uv`, `google-github-actions/*`, and Docker setup/login/build actions.
- Introducing reusable workflows before scripts remove duplicated behavior.
- Changing what tests run inside unit/integration/smoke workflows, except where required by script extraction.
- Changing live Iris cluster lifecycle policy.
- Creating a generic `scripts/workflows/lib/` package before repeated helper pressure exists.
