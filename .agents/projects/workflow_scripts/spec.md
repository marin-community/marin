# Workflow Scripts Spec

This spec defines the target contracts for moving Marin's GitHub Actions workflows to the workflow-scripts model. The implementation can land in large staged PRs, but each stage must preserve the contracts below.

## File Layout

| Path | Contract |
| --- | --- |
| `scripts/workflows/__init__.py` | Empty package marker for workflow-owned Python modules. No exported convenience API. |
| `scripts/workflows/changes.py` | Repo-local replacement for path-filter logic used by unit/docs/lint workflows. |
| `scripts/workflows/github_actions.py` | Workflow inventory and policy checks: naming, `.yaml`, third-party SHA pins, and required-status inventory. |
| `scripts/workflows/iris_monitor.py` | Iris job `status`, `wait`, and `collect` (failure diagnostics) for Iris-backed workflows. |
| `.github/workflows/README.md` | Human index of workflows: target name, trigger, gate type, owner domain, and local reproduction command. Includes the canonical `git + gh pr create/edit` recipe used in place of third-party PR-creation actions. |
| `pyproject.toml` | Includes `scripts/workflows/**` in ruff and pyrefly while leaving unrelated legacy scripts excluded. |

GitHub state changes (PR creation/update, issue creation, comments, labels, releases, branch-protection queries) are performed by calling the pre-installed `gh` CLI directly from workflow YAML, not by Python wrappers under `scripts/workflows/`. The bar for promoting a `gh`-based snippet into a Python module is real logic that benefits from tests: not a wrapper that adds `--dry-run` to a single API call.

Do not add `scripts/workflows/lib/` in the first migration. If two or more modules need the same non-trivial helper, add a concrete helper module named for the concept, not a generic utility module. Docker image build orchestration can be added later if the cleanup shows real duplicated behavior after trusted Docker setup/login/build actions remain in YAML.

## CLI Contracts

All `scripts/workflows/*.py` modules are repo-local commands invoked as:

```bash
uv run python scripts/workflows/<name>.py <command> [options]
```

All new workflow scripts use Click. Commands must fail non-zero on operational failure, print concise human-readable progress to stderr, and write machine-readable data either to stdout or to `$GITHUB_OUTPUT` when `--github-output` is supplied. Python commands that affect external state (writing files outside the workspace, mutating cloud or Iris resources) must support `--dry-run`. `gh`-driven workflow steps are exempt; rely on workflow-level dispatch gating and `--dry-run`-equivalent `gh` flags (`gh api -X GET`, `gh pr list`) when previewing.

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
# Diff-driven (pull_request, push):
uv run python scripts/workflows/changes.py match \
  --base "$BASE_SHA" \
  --head "$HEAD_SHA" \
  --group marin='lib/marin/**,tests/**,pyproject.toml,uv.lock' \
  --github-output

# Manual or scheduled (workflow_dispatch, schedule): force every group on.
uv run python scripts/workflows/changes.py match \
  --always-match \
  --group marin='lib/marin/**,tests/**,pyproject.toml,uv.lock' \
  --github-output
```

`--always-match` is mutually exclusive with `--base`/`--head`; passing both is a usage error.

Output contract:

- stdout JSON: `{"groups": [{"name": "marin", "matched": true, "paths": ["..."]}]}`
- `$GITHUB_OUTPUT`: one lower-case boolean per group, e.g. `marin=true`

For `pull_request` events, callers pass the PR base SHA and head SHA and the command uses merge-base diff semantics. For `push` events, callers pass before/after SHAs and the command uses direct range semantics. For `schedule` and `workflow_dispatch`, workflows must either skip `changes.py` or pass `--always-match`, which marks every group as matched and records `reason="manual-or-scheduled"` in stdout JSON. Deleted-only files do not trigger groups because the command excludes `D` from the diff filter; renames trigger on the new path.

### Pull-request creation via `gh`

`peter-evans/create-pull-request@v7` is replaced by an inline `git + gh` snippet in workflow YAML, not a Python wrapper. **The canonical, copyable version of this recipe lives in `.github/workflows/README.md`; the snippet below is illustrative and may drift — the README is the source of truth.**

```bash
set -euo pipefail

# Set DESIRED_LABELS to the exact label set the bot should own on this PR
# (space-separated). Anything else stays as-is. To clear, set DESIRED_LABELS="".
: "${BRANCH:?}" "${TITLE:?}" "${BODY_FILE:?}" "${COMMIT_MESSAGE:?}"
: "${DESIRED_LABELS:=agent-generated}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Reset onto the existing remote branch when present so --force-with-lease
# has a real lease, then layer the new commit on top. Falls back to a fresh
# branch when origin/$BRANCH does not exist yet.
git fetch origin "$BRANCH" --depth=1 2>/dev/null || true
if git rev-parse --verify --quiet "refs/remotes/origin/$BRANCH" >/dev/null; then
  git checkout -B "$BRANCH" "refs/remotes/origin/$BRANCH"
  # Carry the working-tree edits from the build step over the reset.
  git checkout "$GITHUB_SHA" -- .
else
  git checkout -B "$BRANCH"
fi

git add -A
if git diff --cached --quiet; then
  echo "pr_created=false" >>"$GITHUB_OUTPUT"
  exit 0
fi
git commit -m "$COMMIT_MESSAGE"
git push --force-with-lease origin "$BRANCH"

# Capture URL via --json so we never parse gh's human banner.
PR_URL=$(gh pr list --head "$BRANCH" --state open --json url --jq '.[0].url // ""')
if [[ -z "$PR_URL" ]]; then
  gh pr create --base main --head "$BRANCH" --title "$TITLE" --body-file "$BODY_FILE" >/dev/null
  PR_URL=$(gh pr view "$BRANCH" --json url --jq .url)
else
  gh pr edit "$PR_URL" --title "$TITLE" --body-file "$BODY_FILE" >/dev/null
fi

# Reconcile labels: peter-evans semantics (replace), expressed via add+remove.
CURRENT_LABELS=$(gh pr view "$PR_URL" --json labels --jq '[.labels[].name] | join(" ")')
for label in $DESIRED_LABELS; do
  case " $CURRENT_LABELS " in *" $label "*) ;; *) gh pr edit "$PR_URL" --add-label "$label" >/dev/null ;; esac
done
for label in $CURRENT_LABELS; do
  case " $DESIRED_LABELS " in *" $label "*) ;; *) gh pr edit "$PR_URL" --remove-label "$label" >/dev/null ;; esac
done

echo "pr_url=$PR_URL" >>"$GITHUB_OUTPUT"
echo "pr_created=true" >>"$GITHUB_OUTPUT"
```

Three details that bit the v1 of this recipe and must stay fixed:

- **URL capture.** `gh pr create` writes a human banner before the URL on non-TTY runners; capture from `gh pr view --json url` (or `gh pr list --json url`) instead of parsing `gh pr create` stdout.
- **Stale branch + `--force-with-lease`.** `git checkout -B` against `HEAD` with no knowledge of `origin/$BRANCH` yields an empty lease, and the push will be rejected. Always `git fetch` the branch first and reset onto it when present, then re-apply the build-step edits before staging.
- **Label semantics.** `gh pr edit --add-label` accumulates; peter-evans `labels:` replaced. The reconcile loop above expresses replace-semantics explicitly. If callers want accumulate-only, they should set `DESIRED_LABELS` to the union and skip the second loop.

`gh` reads `GITHUB_TOKEN` (or `GH_TOKEN`) from the workflow environment. The default `GITHUB_TOKEN` issued by Actions cannot trigger downstream workflow runs on PRs it creates — the same limitation peter-evans has. Callers that need downstream CI on the auto-PR must use a PAT or a GitHub App token (e.g. via `actions/create-github-app-token`) bound to `GH_TOKEN`. Workflows must grant `contents: write` and `pull-requests: write`; `actions/checkout` must keep its default `persist-credentials: true` for the `git push` to succeed.

Reviewers, assignees, branch deletion, signed commits, multi-commit preservation, and automatic base rebasing are explicitly out of scope; if a future caller needs them, add `gh` flags or promote to a Python helper at that point — not preemptively.

Issue creation (`gh issue create`), PR/issue comments (`gh pr comment`, `gh issue comment`), label edits (`gh pr edit --add-label`/`--remove-label`), and branch-protection queries (`gh api repos/.../branches/.../protection`) follow the same rule: invoke `gh` directly from YAML.

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

### `iris_monitor.py`

One CLI entry point exposes `status`, `wait`, and `collect` subcommands. The shared Iris CLI plumbing (`.venv/bin/iris` when present, otherwise `uv run iris`; `--config` plumbing; JSON parsing) is small (~10 lines), but the provider-specific diagnostics (GCP SSH + controller logs vs CoreWeave kubectl/pod listing) are not. Internally, the module is split:

```text
scripts/workflows/
  iris_monitor.py            # Click entry point + status/wait logic
  _iris_cli.py               # Shared: locate iris binary, parse `iris job list --json`, IrisJobStatus
  _iris_diagnostics_gcp.py   # GCP-only: gcloud SSH, controller log fetch
  _iris_diagnostics_coreweave.py  # CoreWeave-only: kubectl, pod listing
```

`_iris_cli.py` is the first concrete helper module. `lib/` is still not introduced — these are sibling modules with leading underscores, and they exist because at least two callers (wait, collect) share them. If a third workflow script needs the same Iris CLI plumbing, that's the trigger to consider promoting `_iris_cli.py` to a public module, not creating `lib/`.

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


@dataclass(frozen=True)
class DiagnosticsRequest:
    job_id: str
    output_dir: Path
    iris_config: Path | None
    provider: Literal["gcp", "coreweave"]
    project: str | None
    controller_label: str | None
    namespace: str | None


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


def collect_diagnostics(request: DiagnosticsRequest) -> Path:
    """Collect Iris controller, job tree, and provider-specific task diagnostics into `output_dir`."""
```

CLI:

```bash
uv run python scripts/workflows/iris_monitor.py wait \
  --job-id "$JOB_ID" \
  --iris-config "$IRIS_CONFIG" \
  --poll-interval 30 \
  --github-output

uv run python scripts/workflows/iris_monitor.py collect \
  --job-id "$JOB_ID" \
  --iris-config "$IRIS_CONFIG" \
  --provider gcp \
  --output-dir "$DIAG_DIR"
```

`wait` selects the exact `job_id` from JSON output and treats unknown terminal states as failure. `--github-output` writes `job_id`, `state`, and `succeeded` before exiting on both success and known terminal failure; it may be absent only when the command cannot parse status at all. `SIGINT` and `SIGTERM` stop polling and exit non-zero without cancelling the remote job.

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
| `marin-metrics.yaml` | *(deleted in PR 1)* | *(deleted in PR 1)* |
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

- Add `scripts/workflows/changes.py` and `scripts/workflows/github_actions.py`.
- Add `.github/workflows/README.md`, including the canonical `git + gh pr create/edit` recipe.
- Update `pyproject.toml` so `scripts/workflows/**` is linted and type checked.
- Replace `dorny/paths-filter` usages in unit/docs/lint/release workflows with `changes.py`.
- Replace `peter-evans/create-pull-request@v7` in `dupekit-wheels.yaml` with the inline `git + gh` snippet from `.github/workflows/README.md`. No Python wrapper is added for this.
- Delete `marin-metrics.yaml` outright (dead conda-based weekly job; no migration). This removes the only `conda-incubator/setup-miniconda` usage from the repo.
- Pin all remaining non-trusted third-party actions to full SHAs, including `dorny/paths-filter`, `peaceiris/actions-gh-pages`, `actions/github-script` if retained, `actions/stale`, and `anthropics/claude-code-action`.
- Keep existing required job ids unchanged in this PR unless branch protection is updated in the same PR window.

Workflow scope is grouped by what changes, since the risk profile differs:

**`paths-filter` → `changes.py` conversions** (path-filter logic moves into Python; behavior should be identical):

- `dupekit-unit-tests.yaml`
- `fray-unit-tests.yaml`
- `haliax-run_tests.yaml`
- `iris-unit-tests.yaml`
- `levanter-tests.yaml`
- `marin-docs.yaml`
- `marin-lint-and-format.yaml`
- `marin-unit-tests.yaml`
- `zephyr-unit-tests.yaml`

**Release-state-changing workflow** (separate review focus — this is the only PR-1 workflow that performs GitHub state changes; gate-test against the actual auto-PR branch before merge):

- `dupekit-wheels.yaml` (peter-evans → `gh` snippet)

**SHA-pin-only conversions** (no `changes.py`, no PR-creation rewrite — only third-party action pinning):

- `grug-variant-diff.yaml`
- `stale.yml`

**Deletion**:

- `marin-metrics.yaml` (dead conda weekly; safe to remove — `experiments/metrics/exp446_metrics.py` no longer exists in the tree, no docs/AGENTS.md references)

### PR 2: Iris, Ferries, and Live-Infrastructure Workflows

Required content:

- Add `scripts/workflows/iris_monitor.py` with `status`, `wait`, and `collect` subcommands.
- Migrate all copied Iris wait loops to `iris_monitor.py wait`.
- Migrate all copied GCP/CoreWeave diagnostics to `iris_monitor.py collect`.
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
