# GitHub Actions Workflows

This directory contains thin trigger YAML around behavior implemented in `scripts/workflows/`. See the design at `.agents/projects/workflow_scripts/design.md` and contracts at `.agents/projects/workflow_scripts/spec.md`.

## Inventory

| File | Workflow Name | Trigger | Gate Type | Owner Domain | Local Reproduction |
| --- | --- | --- | --- | --- | --- |
| `dupekit-release-wheels.yaml` | Dupekit - Release Wheels | PR + push to main + workflow_dispatch | release | dupekit | see job steps |
| `dupekit-unit.yaml` | Dupekit - Unit | PR + push to main | unit | dupekit | `cd rust/dupekit && uv run --frozen --group test pytest tests/ -v` |
| `fray-unit.yaml` | Fray - Unit | PR + push to main | unit | fray | `cd lib/fray && uv run --group=fray-test pytest --durations=5 --tb=short -m 'not slow and not tpu_ci' -v -s tests/` |
| `haliax-unit.yaml` | Haliax - Unit | PR + push to main | unit | haliax | `JAX_NUM_CPU_DEVICES=8 uv run --package marin-haliax pytest -c pyproject.toml tests` |
| `iris-dev-restart.yaml` | Iris - Dev Restart | schedule (daily) + workflow_dispatch | ops | iris | see job steps |
| `iris-release-iap-proxy.yaml` | Iris - Release IAP Proxy | PR + push to main | release | iris | see job steps |
| `iris-smoke-coreweave.yaml` | Iris - Smoke - CoreWeave | PR + issue_comment + workflow_dispatch | smoke | iris | see job steps |
| `iris-smoke-gcp.yaml` | Iris - Smoke - GCP | PR + issue_comment + workflow_dispatch | smoke | iris | see job steps |
| `iris-unit.yaml` | Iris - Unit | PR + push to main | unit | iris | `cd lib/iris && uv run --group dev python -m pytest -n1 --durations=5 --tb=short -m 'not slow and not docker and not e2e' tests/` |
| `levanter-dev-launch-small-fast.yaml` | Levanter - Dev - Launch Small Fast | workflow_dispatch | dev | levanter | see job steps |
| `levanter-integration-gpt2-small.yaml` | Levanter - Integration - GPT-2 Small | workflow_dispatch | integration | levanter | see job steps |
| `levanter-unit.yaml` | Levanter - Unit | PR + push to main | unit | levanter | `uv run --package marin-levanter --frozen --with "jax[cpu]==0.9.2" pytest tests -m "not entry and not slow and not tpu and not torch"` |
| `marin-canary-datakit-nemotron.yaml` | Marin - Canary - Datakit Nemotron | schedule + workflow_dispatch | canary | marin | see job steps |
| `marin-canary-ferry-coreweave.yaml` | Marin - Canary Ferry - CoreWeave | schedule + workflow_dispatch | canary | marin | see job steps |
| `marin-canary-ferry.yaml` | Marin - Canary Ferry | schedule + workflow_dispatch | canary | marin | see job steps |
| `marin-docs.yaml` | Marin - Docs | PR + push to main | docs | marin | `uv run python infra/check_docs_source_links.py` |
| `marin-integration.yaml` | Marin - Integration | PR + push to main + workflow_dispatch | integration | marin | `uv run pytest tests/integration/iris/` |
| `marin-lint.yaml` | Marin - Lint | PR + push to main | lint | marin | `./infra/pre-commit.py --all-files` |
| `marin-release-libs-wheels.yaml` | Marin - Release Libs Wheels | PR + push to main + schedule + workflow_dispatch | release | marin | see job steps |
| `marin-smoke-datakit.yaml` | Marin - Smoke - Datakit | schedule + workflow_dispatch | smoke | marin | see job steps |
| `marin-unit.yaml` | Marin - Unit | PR + push to main | unit | marin | `uv run --package marin --extra cpu --frozen pytest -n 4 --dist=worksteal --durations=5 --tb=short -m 'not slow and not tpu_ci and not integration' -v tests/` |
| `ops-claude-review.yaml` | Ops - Claude Review | PR + issue_comment | ops | claude | see job steps |
| `ops-claude.yaml` | Ops - Claude | issue_comment + issues | ops | claude | see job steps |
| `ops-codeql.yaml` | Ops - CodeQL | PR + push to main + schedule | ops | ops | see job steps |
| `ops-docker-images.yaml` | Ops - Docker Images | schedule (weekly) + workflow_dispatch | ops | ops | see job steps |
| `ops-grug-variant-diff.yaml` | Ops - Grug Variant Diff | PR | ops | ops | see job steps |
| `ops-infra-dashboard.yaml` | Ops - Infra Dashboard | PR + push to main | ops | ops | see job steps |
| `ops-nightshift-ci-tests.yaml` | Ops - Nightshift CI Tests | schedule + workflow_dispatch | integration | ops | see job steps |
| `ops-nightshift-cleanup.yaml` | Ops - Nightshift Cleanup | schedule + workflow_dispatch | ops | ops | see job steps |
| `ops-nightshift-doc-drift.yaml` | Ops - Nightshift Doc Drift | schedule + workflow_dispatch | docs | ops | see job steps |
| `ops-stale.yaml` | Ops - Stale | schedule + workflow_dispatch | ops | ops | see job steps |
| `zephyr-integration-shuffle.yaml` | Zephyr - Integration - Shuffle | workflow_dispatch | integration | zephyr | see job steps |
| `zephyr-unit.yaml` | Zephyr - Unit | PR + push to main | unit | zephyr | `uv run --package marin-zephyr --frozen pytest --durations=5 --tb=short -m 'not slow and not tpu_ci' -v lib/zephyr/tests/` |

## Canonical recipe: open or update a bot PR with `git + gh`

This recipe replaces `peter-evans/create-pull-request@v7`. It creates the PR if missing, updates it (force-with-lease) if present, reconciles labels, and writes `pr_url` and `pr_created` to `$GITHUB_OUTPUT`.

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

### Three details that bit v1

- **URL capture.** `gh pr create` writes a human banner before the URL on non-TTY runners; capture from `gh pr view --json url` (or `gh pr list --json url`) instead of parsing `gh pr create` stdout.
- **Stale branch + `--force-with-lease`.** `git checkout -B` against `HEAD` with no knowledge of `origin/$BRANCH` yields an empty lease, and the push will be rejected. Always `git fetch` the branch first and reset onto it when present, then re-apply the build-step edits before staging.
- **Label semantics.** `gh pr edit --add-label` accumulates; peter-evans `labels:` replaced. The reconcile loop above expresses replace-semantics explicitly. If callers want accumulate-only, they should set `DESIRED_LABELS` to the union and skip the second loop.

### `gh` token and permissions notes

`gh` reads `GITHUB_TOKEN` (or `GH_TOKEN`) from the workflow environment. The default `GITHUB_TOKEN` issued by Actions cannot trigger downstream workflow runs on PRs it creates — the same limitation peter-evans has. Callers that need downstream CI on the auto-PR must use a PAT or a GitHub App token (e.g. via `actions/create-github-app-token`) bound to `GH_TOKEN`. Workflows must grant `contents: write` and `pull-requests: write`; `actions/checkout` must keep its default `persist-credentials: true` for the `git push` to succeed.

### Required workflow boilerplate

```yaml
permissions:
  contents: write
  pull-requests: write

steps:
  - uses: actions/checkout@v5
    # persist-credentials must remain default (true) for the git push to succeed
  - name: Open or update PR
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      BRANCH: bot/auto-update
      TITLE: "Auto-update from CI"
      BODY_FILE: pr_body.md
      COMMIT_MESSAGE: "Auto-update artifacts"
      DESIRED_LABELS: "agent-generated automated"
    run: ./scripts/workflows/_open_pr.sh   # or inline the snippet
```

The README is the source of truth — when fixing the recipe, fix it here. The spec.md copy is illustrative.
