# GitHub Actions Workflows

This directory contains thin trigger YAML around behavior implemented in `scripts/ci/`. See the design at `.agents/projects/workflow_scripts/design.md` and contracts at `.agents/projects/workflow_scripts/spec.md`.

## Agent prose cleanup

`ops-agent-prose-cleanup.yaml` cleans issue and PR descriptions carrying the
`agent-generated` label. It runs when an item is opened, edited, reopened, or
labeled. The workflow executes `scripts/ci/github_prose_cleanup.py` from the
default branch; `pull_request_target` never checks out the PR head.

Claude reads the common writing-style guide, the AI-writing checklist, and the
applicable issue or pull request guide from `.agents/skills/writing-style/`, then
rewrites the description as a permanent record for a technical reader. The
workflow prompt narrows that policy to a soft edit: preserve source facts,
measurements, caveats, links, and useful code examples; do not import facts from
the diff or comments; do not target a word count; and leave compliant text and
titles unchanged.

The model job has read-only repository permissions, receives only the filesystem
`Read` tool, and returns a schema-validated body. A separate write job runs the
Python finalizer, which applies deterministic presentation checks outside fenced
and inline code, rejects empty or oversized rewrites, and prepares the GitHub
update.

Before an edit, the workflow stores the exact prior description in a collapsed
`github-actions[bot]` comment. The edited description ends with an `Original
description` link to that comment. Content hashes make archive creation
idempotent across retries, and the workflow rechecks the current description
before writing so a queued run cannot overwrite a newer edit. It skips cleanup
when the archive or updated body would exceed GitHub's size limit.

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
- **Stale branch +** `--force-with-lease`**.** `git checkout -B` against `HEAD` with no knowledge of `origin/$BRANCH` yields an empty lease, and the push will be rejected. Always `git fetch` the branch first and reset onto it when present, then re-apply the build-step edits before staging.
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
    run: |
      # inline the open-or-update-PR snippet from the section above
```

The README is the source of truth — when fixing the recipe, fix it here. The spec.md copy is illustrative.
