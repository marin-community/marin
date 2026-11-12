---
name: manage-pr
description: Prepare, publish, update, and monitor a Marin pull request, including issue-driven fixes and requested commit or push checkpoints.
---

# Manage a pull request

Own the author-side workflow from a reviewable change through publication and
feedback. For an explicitly assigned GitHub issue, first read
[issue-driven fixes](references/issue-fixes.md); it covers diagnosis and duplicate
work before implementation. Other implementation tasks use their domain skill
until the change is ready to publish. `review-pr` owns independent assessment;
`simplify-code` owns an explicitly requested cleanup.

Before authoring a commit or PR title or body, read:

- `docs/dev-guide/writing-style/index.md`
- `docs/dev-guide/writing-style/pull-requests.md`
- `docs/dev-guide/writing-style/ai-writing-donts.md`

Clean and format the diff, commit it, then run advisory review and independent
validation concurrently. Resolve their results before pushing and publishing,
then monitor the PR. A WIP checkpoint may stop after clean-up, lint, staging,
commit, and push.

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

Finish automatic fixes before starting concurrent readers of the diff. This
entry point also runs Pyrefly when applicable; do not repeat an equivalent
standalone type check on unchanged inputs.

## 3. Stage changes

Review `git status` and `git diff`, then stage only this task's explicit files.
Never stage secrets or unrelated work.

## 4. Commit

Follow `docs/dev-guide/writing-style/pull-requests.md` for the commit subject and body.

Review the exact message before committing. After the commit, inspect it with
`git show -s --format='%s%n%n%b' HEAD`; do not push if a tool added attribution,
a session trailer, or other text that was not in the reviewed message.

If a hook fails, fix the issue and create a new commit. Do not amend unless the
user asks.

If there are no changes to commit, say so and stop.

## 5. Run advisory review and validation concurrently

After the initial commit, launch advisory review alongside affected tests.
Use separate command sessions or the runtime's concurrent tool calls; retain
each process handle, output, and exit status. Do not wait for test results to
start the review.

```bash
./infra/pre-commit.py --review --calling-agent=claude
```

The advisory review is required before every PR. It reviews the whole branch
diff against the merge base, including uncommitted work.

Set `--calling-agent` to your agent (`claude` or `codex`) and run the review
without asking for confirmation.

Run these checks while the review proceeds, concurrently where independent:

- `uv run --no-project infra/ci/run_tests.py` for affected safe unit tests.
  Do not override the repository's default marker expression.
- If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`.
- If the change is docs-heavy: `uv run mkdocs build --strict`.

While commands run, draft the PR title/body and gather issue links. Keep the
reviewed files and Git base stable until all readers finish: reviewers inspect
the live checkout throughout their run. Do not edit, auto-fix, rebase, or run
checks that rewrite reviewed files concurrently. The affected-test runner syncs
the checkout's `.venv`; serialize other commands that sync that environment.
For example, queue docs checks behind tests while review continues, or use
`uv run --no-sync` for docs checks once their dependencies are installed and
the environment sync has finished. Serialize commands that share mutable
outputs. Reuse existing environments and shared caches.

Reuse passing checks from implementation when their inputs remain unchanged.
Run only outstanding checks and those invalidated by subsequent edits. Prefer
the affected-test runner over a full suite unless the change requires broader
coverage.

Artifacts from advisory review are written under
`/tmp/marin-linter/<branch>/<timestamp>-<uniq>/`.

## 6. Collect results and resolve findings

Wait for every started command and inspect its output and exit status before
pushing or publishing. A launched process is not a completed check; advisory
review can exit successfully while reporting findings or an incomplete run.
Confirm it actually reviewed the branch and inspect its findings and failures.
Resolve every finding, checking the matching `infra/lint` rule, and fix failed
checks. Put fixes in a new commit after validating the affected behavior.

Do not recursively rerun `--review` after small, targeted touch-ups made in
response to its findings. Validate behavioral edits with the normal mechanical
checks and relevant tests. For a formatter-only mechanical edit, rerun formatting
and lint checks only. Rerun the advisory review only when the follow-up materially
changes the branch's implementation approach or scope, or when the user asks for
another pass. Apply the same rule to fixes prompted by tests or docs checks.

## 7. Push

Push when requested or when the branch has an upstream. Use
`git push -u origin HEAD` when needed. Stop on rejection; do not force-push
without explicit authorization.

## 8. Open or update the PR

The PR body becomes the squash-merge commit message. Follow
`docs/dev-guide/writing-style/pull-requests.md`.

Add `Fixes #NNNN` or `Part of #NNNN` only when an existing issue applies.

**Inspect the payload.** Draft the body in a uniquely named temporary file and
use `--body-file`. Re-open that file and apply the final compression pass before
publishing. After creating or editing the PR, fetch the
exact `title,body` with `gh pr view --json` and immediately correct text inserted
by a tool or stale template.

Push to the main repository unless direct access is unavailable or the user
asks for a fork:

```bash
gh pr create --title "<title>" --body-file "<body-file>" \
  --label agent-generated --label agentic-lint
```

- Always add the `agent-generated` label.
- Add `agentic-lint` only after advisory review completed and every finding was fixed or answered.
  The label records that lint pass; without it, CI launches a Loom lint review.
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
