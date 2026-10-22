# GitHub Actions Workflows

This directory contains thin trigger YAML around behavior implemented in `scripts/ci/`. See the design at `.agents/projects/workflow_scripts/design.md` and contracts at `.agents/projects/workflow_scripts/spec.md`.

## Pulumi service rollouts

`ops-pulumi-rollout.yaml` runs
`uv run --all-packages --extra deploy --frozen marin-deploy <service> rollout` for
registered service projects after their source paths change on `main`.
`scripts/ci/pulumi_rollouts.py` maps each service to its Pulumi directory, deploy
identity, tests, and additional source roots.
The registry currently covers Ducky, Echo, EvalDash, Grafana, and XProf. Add one `Rollout` entry when
another service should deploy through this workflow.

For an `IrisService`, every registered shared source root must also be present in that
service's `code_paths`. The rollout selector starts the shared deploy command;
`code_paths` changes the `command.local.Command` input that resubmits the Iris job.

Pull requests run each affected service's registered pytest path. Manual dispatch selects one
registered service and accepts a deploy-generation override for Ducky and XProf. Ducky and
XProf still read their runtime credentials from GitHub Actions secrets; the rollout job exposes
those values only to the matching matrix entry.

Run one stack manually from `main` with:

```bash
gh workflow run ops-pulumi-rollout.yaml --ref main -f service=echo
```

## Agent prose cleanup

`ops-agent-prose-cleanup.yaml` launches a Loom session for an issue or PR carrying
the `agent-generated` label whose author has write access or is `marin-ops-agent[bot]`
or `loom-oa-dev[bot]`; external issues and PRs are not cleaned up. Its goal directs the session to follow the
writing-style guide and apply the archive, validation, and stale-body checks
in `scripts/ci/github_prose_cleanup.py` before any update. Actions checks out
only the trusted default-branch launch and access-check actions. The workflow runs on open, reopen, and label
events, not edits. The launch uses the low-effort `prose-cleanup` Loom profile published as
`LOOM_PROSE_CLEANUP_PROFILE`, and its idempotency key is the issue or PR number, so later label
and reopen events return the existing run instead of starting another. A body that already carries
the `marin-prose-cleanup` marker skips the launch entirely.

## Agentic lint

`agentic-lint.yaml` runs a best-effort Loom lint review on a pull request that
lacks the `agentic-lint` label. The label records that the PR has had a lint
pass: the `commit` skill applies it at PR creation after the author's local
`./infra/pre-commit.py --review`, and the Loom session adds it after reporting
its findings. The launcher runs when a non-draft PR from this repository opens,
reopens, or becomes ready for review, and only when its author has write access
or is `marin-ops-agent[bot]`. Pushes to a PR do not launch another review;
removing the label requests a fresh review of the current head. The idempotency
key is the PR number and head SHA, so repeated events on one commit reuse a run.
The review is not a required check, so a failed Loom run does not block merging.
Loom's `agentic-lint` profile and the workflow's OIDC federation are declared in
`infra/loom/Pulumi.marin-loom.yaml`; the GitHub profile variable is published
from `infra/pulumi/github`.

Other GitHub agent entry points use the `github-automation` Loom profile.

Loom launchers that respond to a user-authored issue, comment, review, or pull
request decide who may trigger them with the `check-write-access` action
rather than the event payload's `author_association`. The payload hides
private organization membership, so a regular member with private visibility
appears as `CONTRIBUTOR` and would be skipped. The action asks GitHub for the
author's effective repository permission with the workflow token and allows
`write`, `maintain`, or `admin`; a lookup failure logs the API error and denies
the launch. Bot accounts do not resolve through that endpoint, so the
pull-request launchers pass `marin-ops-agent[bot]` through the action's
`allowed-bots` input. Scheduled and label-driven launchers such as prose
cleanup do not use the check.

Canary failure workflows launch an independent triage session and continue to
send the immediate Slack fallback notification. The Iris smoke test still
uploads screenshots for inspection through the `Marin - Unit` run.

## Holistic review and deslop

`ops-loom-holistic-review.yaml` launches a whole-change review session for a
non-draft pull request from this repository whose author has write access or
is `marin-ops-agent[bot]` or `loom-oa-dev[bot]`. A gate step reads the pull
request's size and file patches through the API: the review runs when the
change has at least 250 changed lines or adds a class, dataclass, or
`Protocol` under `lib/*/src/` or `experiments/`. The session runs the
`review-pr-holistic` skill with the `pr-holistic-review` profile and maintains
one compact overview ending in
`<!-- marin-holistic-review:<sha> -->`. Re-reviews replace the original
overview by comment ID, preserving finding numbers and updating their status.
A separate new top-level comment may announce only new or materially changed
actionable findings; it links to the overview and uses
`<!-- marin-holistic-followup:<sha> -->` so it cannot become the overview.
Unlike the correctness review, the workflow also launches on `synchronize`;
runs for an earlier head are cancelled, and the session posts nothing when the
delta since the last reviewed head is under 100 changed lines, touches only
fixtures or documentation, or consists only of `Deslop` commits. The review is
advisory and is not a required check.

The `deslop` job in `ops-loom-github.yaml` launches a `github-automation`
session when a holistic packet is posted on a pull request carrying
`agent-generated`, or when a maintainer review contains `deslop`. The session
applies only the packet's mechanical items and the `noslop` skill, pushes one
`Deslop` commit after the narrow checks pass, and replies with
`<!-- marin-deslop:<sha> -->`. The launcher's per-PR idempotency key permits
one automatic deslop session; follow-up findings are advisory.
Pull requests without `agent-generated` receive
committable suggestions and no push. The holistic session ignores a head whose
delta is only `Deslop` commits, which closes the review-then-deslop loop.

## Nightshift area reviews

`ops-nightshift-cleanup.yaml` runs on weekdays at 10:00 UTC and by manual
dispatch. It launches one `github-automation` session. Follow this procedure
for that assignment; CI-test and documentation-drift jobs have separate scopes.

1. **Choose one area.** Read applicable `AGENTS.md` files and use
   `consult-echo` for prior work. Inspect the last 20 PRs labeled `nightshift`,
   including closed PRs and relevant feedback:

   ```bash
   gh pr list --state all --label nightshift --limit 20 \
     --json number,title,state,createdAt,url,files
   ```

   Choose a bounded subsystem under `lib/marin/src/marin`,
   `lib/iris/src/iris`, `lib/zephyr/src/zephyr`, or
   `lib/levanter/src/levanter`. Explain why it deserves review: recurring
   friction, recent changes, a critical operational boundary, or neglected
   coverage. Prefer an area recent runs have not reviewed unless new evidence
   justifies returning.

2. **Understand the system before proposing edits.** Apply the whole-system
   questions from [holistic review](../../.agents/skills/review-pr-holistic/SKILL.md):
   fit with existing concepts, better abstractions, consequences across
   boundaries, hidden tests, claims versus evidence, related PRs, test value,
   and scope. This is an audit of existing behavior; the PR-specific head,
   delta, posting, and telemetry procedures apply only to PR reviews.
   Trace entry points, current callers, configuration defaults, state and
   error transitions, neighboring abstractions, tests, and recent history.
   Read `TESTING.md` and module testing guidance when assessing coverage.
   Write a short walkthrough of how the area works.

3. **Compare up to three supported improvements.** Do not fill a quota. Each
   candidate needs exact source paths and lines, an affected user or operator,
   a concrete consequence, a replacement using existing concepts, and a safe,
   falsifiable regression or measurement. Establish why the affected case
   matters now: a non-test caller exercising it, an active operational path,
   a recent failure or request, repeated CI observations, or a measurement on
   a representative workload. Accompany a synthetic reproducer with evidence
   of priority. Public-API completeness or unused option combinations alone are
   insufficient reasons to spend a nightshift run.

   Trace the proposed replacement through the ordinary execution path. Account
   for added bytes read, peak memory, network requests, repeated passes, and
   latency. A format or edge-case fix that introduces full-file materialization
   or extra work for common callers needs evidence that its benefit outweighs
   that cost.

   Check open PRs by listing their changed files and filtering by the
   candidate's paths; `gh pr list --search` does not search changed paths.
   Read relevant issues and PR discussions. Defer work already covered by an
   open PR, rejected without new evidence, or requiring a maintainer decision.

4. **Choose by demonstrated benefit.** Record a candidate table with evidence
   of current use, consequence or measured cost, proposed change, validation,
   effort, risk, and selection or rejection reasons. Distinguish observed
   impact from possible severity; mark frequency and savings unknown when
   unmeasured. Prefer removal of recurring cost or failure in exercised paths
   over isolated edge-case completeness. Explain why the winner is worth more
   than the alternatives relative to its implementation and review burden.
   Avoid style-only changes, trivial extraction, and speculative performance
   claims. Defer changes needing live migrations, distributed ownership
   redesign, multiple PRs, or validation unavailable to this run.

   If no candidate has enough demonstrated value for one coherent PR, report
   what was reviewed and why no change is warranted. Do not switch areas
   repeatedly to force a PR.

5. **Validate the premise, then implement one improvement.** Reproduce the
   defect or measure the cost before editing. Drop disproved candidates and
   reconsider the remaining ones. Follow applicable implementation and
   testing skills, and compare the changed behavior with the baseline using
   the promised regression or measurement. Measure newly added I/O or
   materialization on representative input sizes. Passing existing tests alone
   does not establish the improvement. Use `update-docs` when guidance changes and
   `commit` to publish a PR with `agent-generated` and `nightshift` labels.
   Do not merge it or file issues for unselected candidates.

Publish the report in the session's Loom channel with message kind `result`:

```bash
loom channels send --kind result "<review and outcome>"
```

Include the source revision, reviewed area, walkthrough, candidate table,
selection or no-change rationale, validation results and limits, and PR URL
when present. Keep the full review in the session; keep the PR description
focused on its resulting behavior.

## SkyRL runtime checks

`Marin - Unit` selects `skyrl-wheel` for SkyRL, its editable Marin and Harbor
dependencies, or CI tooling changes. It checks the nested lock and runs the
packaging, launcher, and verifier tests in SkyRL's CPU environment. GPU smoke
tests run separately; see `docs/dev-guide/skyrl.md`.

`Marin - SkyRL Nightly` runs OPD, Grug/Megatron and asynchronous GPU CatCount
at 09:00 UTC. The Grug CUTLASS case runs weekly or by manual selection.
`Marin - SkyRL CPU Nightly` runs CatCount learning and checkpoint/resume gates
at 06:00 UTC. Both use the checked-out nested runtime and its independent lock.

## External dependency updates

Harbor is maintained under `vendor/harbor`. The `harbor-unit` lane and local
affected-test runner use `infra/ci/run_harbor_tests.py` to check its standalone
lock before running offline runtime tests. Changes to its local Rigging and
FineStore dependencies also select this lane.

Evalchemy is maintained under `vendor/evalchemy` with its own lock and dedicated
`evalchemy-unit` lane. Both vendored evaluators use local Rigging and FineStore.

`ops-external-dependencies.yaml` is manual: it regenerates packaged external
requirements and opens an update only when generated state changes. Vendored
SkyRL, Evalchemy and Harbor source updates land directly in Marin.

`ops-native-package-dependencies.yaml` applies the same policy to Marin's
published native wheels. A successful non-PR `Marin - Release Packages` run
triggers the updater from the default branch, where the protected environment
can release the app key. The updater reads the newest published DupeKit,
Finelog, and Iris native versions from PyPI, advances their compatibility floors
and `uv.lock`, and merges only after the same identity, file, SHA, and CI gates
pass. Manual dispatch retries a failed or missed update without republishing the
wheels.

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
