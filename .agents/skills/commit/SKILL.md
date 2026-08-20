---
name: commit
description: Lint, test, commit, push, open or update a pull request, and monitor it through review. Use for any commit, push, or PR operation.
---

# Commit And PR

Read [the writing style](../writing-style/SKILL.md),
[pull-request guide](../writing-style/pull-requests.md), and
[AI-writing checklist](../writing-style/ai-writing-donts.md) before drafting text.
For changed tests, also read root `TESTING.md` and the nearest module testing
instructions.

## Workflow

Use this order for PR-ready work:

1. Review the complete branch diff. Remove dead code, weak tests, stale prose,
   debugging artifacts, and unrelated changes.
2. Run `./infra/pre-commit.py --changed-files --fix`.
3. Run relevant tests. The safe branch-wide default is
   `uv run --no-project infra/ci/run_tests.py`; do not override its marker
   expression. For docs path changes, run
   `uv run python infra/check_docs_source_links.py`; for docs-heavy changes,
   run `uv run mkdocs build --strict`.
4. Review `git status` and `git diff`, then stage only this task's explicit
   files. Never stage secrets or unrelated work.
5. Commit. Stop if there are no changes.
6. Run the advisory lint review once over the committed branch diff:
   `./infra/pre-commit.py --review --agent-command='<headless CLI>'`.
7. Resolve every finding. Put fixes in a new commit; do not amend unless the
   user asks. Rerun mechanical checks and affected tests, but rerun the
   advisory review only if the fixes materially change the approach or scope.
8. Push when requested or when the branch has an upstream. Use
   `git push -u origin HEAD` when needed. Stop on a rejected push; never force
   push without explicit authorization.
9. Open or update the PR, then monitor it as described below.

For a disposable WIP checkpoint, steps 1, 2, 4, 5, and 8 are sufficient. Weak
or exploratory tests must not survive into PR-ready work.

`./infra/pre-commit.py` is the repository lint entry point. Do not substitute
`uv run pre-commit`, skip hooks, or weaken checks.

## Commit

Use an imperative subject of at most 72 characters, optionally prefixed with a
scope such as `[iris]`. Do not use conventional-commit prefixes, emoji,
attribution, provider names, or session trailers. The optional body records the
behavior, reason, evidence, and material caveats; it does not inventory files or
tests.

Review the exact message before committing. Afterward inspect it with:

```bash
git show -s --format='%s%n%n%b' HEAD
```

Do not push if unreviewed text was inserted. If a hook fails, fix the problem
and create a new commit.

## Advisory Review

Run the review after the initial commit and before publishing the PR:

```bash
./infra/pre-commit.py --review --agent-command='codex exec'
```

The command reviews the whole branch diff against `main`, including
uncommitted changes, and writes its artifacts under `/tmp/marin-linter/`. It is
read-only and advisory. Inspect the `infra/lint` rule behind each reported code
and apply findings when they improve the change.

## Pull Request

The PR body becomes the squash-merge commit message. Follow the pull-request
guide and publish the shortest standalone account of changed behavior,
motivation, evidence, and caveats. Add `Fixes #NNNN` or `Part of #NNNN` only
when an existing issue applies.

Draft the body in a uniquely named temporary file, inspect the exact title and
body, and publish with `--body-file`. Add the `agent-generated` label. After
creating or editing the PR, fetch `title,body` with `gh pr view --json` and
correct any inserted or stale text. Do not credit the agent.

## Monitor

Opening or updating a PR includes monitoring until it merges or closes, the
user asks to stop, or a 12-hour wait times out. Green CI alone is not terminal.

Before the first wait, inspect all current issue comments, inline comments, and
submitted reviews once and address actionable feedback. Set one honest Weaver
status, then run one foreground wait:

```bash
weaver status ok "waiting for PR #<N> events"
uv run scripts/ci/wait_for.py --timeout 12h \
  "github.ci <N>" "github.pr <N>" \
  "github.pr_comment <N>" "github.review <N>"
```

Do not poll GitHub, poll a yielded process handle, start a second wait, or
narrate unchanged state while it runs. The command exits with one event: `0`
for an event, `2` for timeout, and `1` for failure. An event is not necessarily
successful: read `result.conclusion` for CI and `result.reasons` for PR events.

Handle the event and any concurrent feedback, then re-arm unless terminal:

- `github.ci`: fix failures. A failure in an untouched file is not automatically
  pre-existing; verify that the same job fails on `main` before treating it as
  unrelated. Once CI is terminal, omit this arm until a push starts a new run.
- `github.pr`: merged and closed are terminal. Resolve conflicts before
  re-arming. Inspect the submitted review when review state changes.
- `github.pr_comment` or `github.review`: address every actionable comment,
  prefix agent-authored replies with `🤖`, and resolve the thread.
- timeout: report the payload's last statuses and hand off.

Read current feedback once before each re-arm so simultaneous events are not
absorbed into a new baseline. A question requiring user input pauses monitoring;
otherwise keep the PR, comment, and review arms active until a terminal state
or timeout.
