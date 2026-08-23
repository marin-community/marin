---
name: commit
description: Use only when a change is ready to commit, push, or open or update its pull request; do not activate during implementation.
---

# Commit and PR

Before authoring a commit or PR title or body, read:

- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/pull-requests.md`
- `.agents/skills/writing-style/ai-writing-donts.md`

Clean and validate the diff, commit it, run the advisory review over the
committed branch, push, update the PR, then monitor it. A WIP checkpoint may
stop after clean-up, lint, staging, commit, and push.

## 1. Clean up your own diff

Read the complete branch diff and remove dead code, weak tests, stale prose,
debugging artifacts, and unrelated changes.

If the diff touches tests and this commit is intended for a PR or review, read
root `TESTING.md` and the relevant module-specific `AGENTS.md` or testing docs
as part of this self-review. Apply `TESTING.md`'s behavioral-value gate to every
changed test. Delete disposable probes and low-value tests before a PR-ready
commit.

## 2. Lint and format

```bash
./infra/pre-commit.py --changed-files --fix   # diff-scoped; use --all-files for a full sweep
```

This is the required entry point; do not substitute `uv run pre-commit`, skip
hooks, or weaken checks.

## 3. Tests and docs checks (when relevant)

- `uv run --no-project infra/ci/run_tests.py` to run affected safe unit tests.
  Do not override the repository's default marker expression.
- If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`.
- If the change is docs-heavy: `uv run mkdocs build --strict`.

## 4. Stage changes

Review `git status` and `git diff`, then stage only this task's explicit files.
Never stage secrets or unrelated work.

## 5. Commit

Use an imperative subject of at most 72 characters with an optional `[scope]`
prefix. The optional body records behavior, reason, evidence, and caveats; it
does not inventory files or tests. Do not use a conventional-commit prefix,
emoji, attribution, provider name, or session trailer.

Review the exact message before committing. After the commit, inspect it with
`git show -s --format='%s%n%n%b' HEAD`; do not push if a tool added attribution,
a session trailer, or other text that was not in the reviewed message.

If a hook fails, fix the issue and create a new commit. Do not amend unless the
user asks.

If there are no changes to commit, say so and stop.

## 6. Lint-catalog review (before every PR)

```bash
./infra/pre-commit.py --review --agent-command='<your headless CLI>'
```

Run this after the initial commit and before publishing the PR. It reviews the
whole branch diff against the merge base, including uncommitted work. Resolve
every finding and put fixes in a new commit. Inspect the matching `infra/lint`
rule and apply findings when they improve the change.

The advisory review is read-only: it does not edit, stage, commit, or push.

Do not recursively rerun `--review` after small, targeted touch-ups made in
response to its findings. Validate behavioral edits with the normal mechanical
checks and relevant tests. For a formatter-only mechanical edit, rerun formatting
and lint checks only. Rerun the advisory review only when the follow-up materially
changes the branch's implementation approach or scope, or when the user asks for
another pass.

Artifacts are written under `/tmp/marin-linter/<branch>/<timestamp>-<uniq>/`.

## 7. Push

Push when requested or when the branch has an upstream. Use
`git push -u origin HEAD` when needed. Stop on rejection; do not force-push
without explicit authorization.

## 8. Open or update the PR

The PR body becomes the squash-merge commit message. Follow
`.agents/skills/writing-style/pull-requests.md`.

Add `Fixes #NNNN` or `Part of #NNNN` only when an existing issue applies.

**Inspect the payload.** Draft the body in a uniquely named temporary file and
use `--body-file`. Re-open that file and apply the final compression pass before
publishing. After creating or editing the PR, fetch the
exact `title,body` with `gh pr view --json` and immediately correct text inserted
by a tool or stale template.

Push to the main repository unless direct access is unavailable or the user
asks for a fork:

```bash
gh pr create --title "<title>" --body-file "<body-file>" --label agent-generated
```

- Always add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.

## 9. Monitor the PR

Green CI is not an exit condition. Monitor every PR until it
merges or closes, the user tells you to stop, or a 12-hour wait times out.

Before the first wait, read every current issue comment, inline review comment,
and submitted review once, then address anything actionable. This is a one-time
inspection, not a monitoring loop. The comment and review arms establish a
baseline when they start and wake only for later new or edited feedback.

If Loom is available, set one honest status immediately before waiting:

```bash
loom status set --tag ok --message "waiting for PR #<N> events"
```

Do not refresh that status while nothing changes. Invoke one `wait_for.py`
process in the foreground and keep it attached until it exits:

```bash
uv run scripts/ci/wait_for.py --timeout 12h \
  "github.ci <N>" "github.pr <N>" \
  "github.pr_comment <N>" "github.review <N>"
```

The command's `--timeout 12h` is the only timeout. Do not background or detach
the process, wrap it in a shell `timeout`, or give the command runner or agent a
shorter deadline. A tool reporting that the command is still running does not
end the foreground wait. Give the process handle back to the runtime's blocking
wait/resume facility and continue waiting for that same process to exit. Repeat
the blocking resume as often as the execution interface requires.

`github.pr` covers terminal merged/closed state, merge conflicts,
ready-for-review transitions, and review-decision changes. It does not inspect
comment or review bodies, so it does not replace `github.pr_comment` or
`github.review`. Keep all three arms. Monitoring must not use a raw `poll` shell
expression, `gh pr checks --watch`, or repeated `gh pr view` calls.

The wait owns its exponential backoff. While it is running:

- remain silent until it returns an event, timeout, or error;
- do not poll GitHub manually;
- do not launch another `wait_for.py`;
- do not narrate unchanged CI, review, or merge state;
- resume a yielded process handle only with another blocking wait on that same
  handle; do not treat the yield as an event.

The command prints one JSON object and exits: `0` means an arm fired, `2` means
the overall timeout elapsed, and `1` means the wait failed. An event is not
always a successful verdict. Read `result.conclusion` for `github.ci` and
`result.reasons` for `github.pr`.

Act on the event, read any feedback that arrived concurrently, then re-arm:

1. **`github.ci`** — on failure, read the failing job log and fix the
   regression. A failure in an untouched file is not automatically
   pre-existing: confirm the same job fails on `main` independently before
   treating it as unrelated. Once CI finishes, omit `github.ci` from the next
   wait because its terminal result would fire immediately. Add it back after a
   push starts a new run.
2. **`github.pr`** — `merged` and `closed` are terminal. Resolve `conflicted`
   before re-arming; an unchanged conflict is intentionally reported again by a
   fresh wait. `ready_for_review` and `review_decision` describe review-state
   changes in the attached snapshots. When `review_decision` fires, inspect the
   submitted review because the lifecycle payload does not include its body.
3. **`github.pr_comment` / `github.review`** — address every actionable human
   and agent comment. Prefix agent-authored replies with `🤖` and resolve the
   thread. The default significant-comment filter ignores the authenticated
   user's comments, review-bot progress placeholders, clean verdicts, wrappers,
   Loom's exact `Working on this in loom: <session URL>` acknowledgement, and
   Loom access-control replies addressed to a bot.
4. **Timeout** — report the last statuses in the timeout payload and hand off.
   Do not replace the completed 12-hour block with manual polling.

Re-arm after every non-terminal event. Once CI finishes, omit `github.ci` so its
terminal result does not fire immediately; keep `github.pr`,
`github.pr_comment`, and `github.review` armed for later feedback. Read current
feedback once before each re-arm so simultaneous events are not absorbed into a
new baseline without being handled.

A question requiring user input pauses monitoring: raise `attention`, ask the
question, and resume after the answer. Otherwise the only exit conditions are a
merged or closed PR, an explicit request to stop, or a 12-hour timeout.
