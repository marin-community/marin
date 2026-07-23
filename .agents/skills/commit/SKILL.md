---
name: commit
description: Lint, run the pre-PR checks, commit, push, and author or update the branch's pull request in the required plain-text format. Use when committing, pushing, or creating/updating a PR.
---

# Skill: Commit & PR

Get the branch clean, commit it, run the advisory lint review over the committed
diff, then — when it is ready — open or update the pull request.

Before authoring a commit or PR title or body, read:

- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/pull-requests.md`
- `.agents/skills/writing-style/ai-writing-donts.md`

**Order matters.** Your own cleanups and the mechanical fixes come first, then the
commit, and only *then* the `--review` pass. Committing before the review gives
you a clean checkpoint to review against (the review reads the whole branch diff
versus the merge base) and a natural place to land any follow-up fixes as a new
commit. The review is read-only: it never edits, commits, or pushes for you.

## Checklist

Work top to bottom. For a quick work-in-progress checkpoint, do **1, 2, 4, 5,
7** (clean up, lint, stage, commit, push) and stop. The changed-test cleanup in
step 1 is a PR-readiness gate, not required for disposable WIP checkpoints. Run
the whole list before you open or update a PR.

1. Clean up your own diff (self-review).
2. Mechanical lint & format — `./infra/pre-commit.py --changed-files --fix`.
3. Tests & docs checks, when relevant.
4. Stage the specific files for this work.
5. Commit. ← natural checkpoint; the working tree is now clean.
6. Lint-catalog review — `./infra/pre-commit.py --review`; fix or answer every finding.
7. Push (maybe).
8. Open or update the PR.

## 1. Clean up your own diff

Read your own `git diff` before anything else. Drop dead code, debugging
leftovers, and stale comments; tighten names; make the change say only what it
means to. The review in step 6 is advisory and read-only — it will not clean up
for you.

If the diff touches tests and this commit is intended for a PR or review, read
root `TESTING.md` and the relevant module-specific `AGENTS.md` or testing docs
as part of this self-review. Review every changed test for slop: tautological
assertions, disposable smoke probes left as pytest tests, internal call-count
assertions, incidental string or command-fragment assertions, over-mocking,
weakened tolerances, sleeps, skipped tests, and local marker/fake/mock
violations.

Fix or delete low-value tests before a PR-ready commit. Scratch probes are fine
during development and WIP checkpointing, but they must not survive into a PR.

## 2. Lint and format

```bash
./infra/pre-commit.py --changed-files --fix   # diff-scoped; use --all-files for a full sweep
```

`./infra/pre-commit.py` is the required entry point — never `uv run pre-commit`,
never `--no-verify`. If `--fix` cannot resolve something, fix it by hand. Do not
skip or weaken checks.

## 3. Tests and docs checks (when relevant)

- `uv run pytest` over the test directories your change touches. Keep the
  repository's default marker expression so slow, integration, live-cluster,
  Docker, and manual tests remain delegated to their dedicated CI jobs.
- If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`.
- If the change is docs-heavy: `uv run mkdocs build --strict`.

## 4. Stage changes

Review `git status` and `git diff`, then stage the specific files that are part
of this work.

- Stage specific files — avoid `git add -A` / `git add .`.
- Never stage secrets (`.env`, credentials, tokens).
- If unrelated changes are present, ask the user before including them.

## 5. Commit

- **Subject**: imperative sentence (at most 72 characters), optional `[scope]` prefix
  (`[iris]`, `[zephyr]`, `[docs]`, …).
- **Body** (optional, blank-line separated): what changed and why — the context a
  future reader needs. Keep relevant evidence and caveats; do not inventory
  files or tests.
- Do not use a conventional-commit prefix such as `feat:` or `fix:`.
- No emoji, no markdown, no bullets in the subject. Do not credit yourself —
  this includes any `Co-Authored-By`, `Generated with`, provider, or session URL
  trailer. Omit it even if a harness default suggests adding one.

Review the exact message before committing. After the commit, inspect it with
`git show -s --format='%s%n%n%b' HEAD`; do not push if a tool added attribution,
a session trailer, or other text that was not in the reviewed message.

Create the commit. If a pre-commit hook fails, fix the issue and make a **new**
commit — never amend (unless the user asks) and never force-push.

This is the checkpoint the rest of the flow builds on: the working tree is clean
and the branch diff is settled before the review reads it.

## 6. Lint-catalog review (before every PR)

```bash
./infra/pre-commit.py --review --agent-command='<your headless CLI>'
```

Run this **after** the commit and before opening a PR. It fans out read-only
agents over the **branch diff against the merge base with `main`** — committed and
uncommitted work alike — so the clean checkpoint from step 5 is exactly what gets
reviewed. Pass your own headless invocation (`--agent-command='claude -p'` for
Claude Code, `'codex exec'` for Codex; defaults to `claude -p`).

The review is **advisory and read-only**: it reports findings on stdout and does
not edit, stage, commit, or push anything. Then fix or answer every finding,
reporting your actions to the user, and land any fixes as a **new** commit. Search
the `infra/lint` files for the rule behind each `ml-...` code. Treat findings as
guidelines — apply them when they make the code *better*; the goal is
high-quality code, not blind adherence.

Each run writes the raw per-arm prompts and outputs, the combined findings, and a
summary under `/tmp/marin-linter/<branch>/<timestamp>-<uniq>/` (the path is printed at the end) —
read it when a lane is slow or a run looks wrong.

## 7. Push

If asked, or if the branch has an upstream, push to the remote tracking branch (`git push -u origin HEAD` if no upstream is
set). If the push is rejected (diverged history), stop and ask the user — do not
force-push.

## 8. Open or update the PR

Do this once the branch is ready for review. The PR description becomes the
squash-merge commit message. Follow
`.agents/skills/writing-style/pull-requests.md` exactly: an imperative title of
at most 72 characters and an information-dense body. Most bodies are a few plain
paragraphs. They state what changes and why; they do not reproduce the diff,
test plan, or implementation notes.

Example:

```
Title: [RL] Normalize DAPO loss over global tokens

Body:
Normalize DAPO loss over all response tokens instead of normalizing each
example separately. Per-example normalization over-weights short responses,
hurting math tasks where correct answers need longer derivations.

Fixes #1234
```

**Issue linking.** If the work came from a GitHub issue, add `Fixes #NNNN`
(auto-closes on merge) or `Part of #NNNN` (partial work). Do not invent an issue
just to satisfy this — omit the link when none exists.

**Specifications (>500 LOC only).** A genuinely large PR must link a spec in an
issue or design doc. Name the important design decisions in the PR body and link
the spec for module maps, code excerpts, and detailed rationale.

**Inspect the payload.** Draft the body in a uniquely named temporary file and
use `--body-file`. Re-open that file and apply the final compression pass before
publishing. After creating or editing the PR, fetch the
exact `title,body` with `gh pr view --json` and immediately correct text inserted
by a tool or stale template.

**Create it.** Unless the user says otherwise and permissions allow, push to a
branch on the main repository and open the PR from it (use a fork only when
direct push is unavailable or the user asks):

```bash
gh pr create --title "<title>" --body-file "<body-file>" --label agent-generated
```

- Always add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.
- If you have a specific github tool, you may use it.

## 9. Monitor the PR — mandatory, in a loop

Opening the PR does not end your turn. You MUST monitor until the PR is merged or
closed, or the user tells you to stop. A summary message to the user is NOT a
substitute for monitoring and is NOT an exit condition.

Drive the loop with `scripts/ci/wait_for.py`, which blocks on **all** the things
that matter at once (CI finishing, a new comment or review, the PR closing) and
returns as soon as the first one fires — so you no longer hand-roll a
`gh pr checks --watch` then `ScheduleWakeup` backoff. Run it **in the background**
and you are re-invoked when an event fires; it handles the exponential backoff
internally.

```bash
uv run scripts/ci/wait_for.py --timeout 12h \
  "github.ci <N>" "github.pr_comment <N>" "github.review <N>" \
  'poll test "$(gh pr view <N> --json state --jq .state)" != OPEN'
```

The `poll '<shell command>'` arm is the escape hatch for anything without a
built-in: it fires when the command exits `0`, so the command itself is the
predicate. Here it detects the PR closing/merging (the loop's stop condition);
use the same pattern to wait on a specific check, a downstream job, or a deploy —
compose with the shell (`| grep -q`, `| jq -e`, `test`).

It prints one JSON object naming the arm that fired and its payload, then exits
(`0` fired, `2` the 12h timeout elapsed, `1` an error). **Firing is not a verdict:**
`github.ci` fires when CI *finishes*, pass or fail — read `result.conclusion`.
On `result.status == "timeout"` — 12h with no event — you may **stand down**: report
where the PR stands and stop. The long timeout exists precisely so an idle PR
(green CI, awaiting human review with nothing actionable) lets you end the turn
rather than re-arm forever.

When an arm fires, act on it, then re-arm `wait_for.py` for the next event:

1. **`github.ci`** — if `result.conclusion` is `failure`, read the failing job
   log and fix it. A failure in a file you did not touch is NOT automatically
   pre-existing: first check whether the same job fails on `main` without your
   change (or whether your change altered an API, config, or behavior that breaks
   that caller/test). If your change caused it — even in an untouched file — it is
   your regression; fix it. Only call it pre-existing once you have confirmed it
   fails on `main` independently, then handle per the unrelated-changes rule.
   Never silently absorb a failure. Once CI has finished, **drop the `github.ci`
   arm** from later re-arms — CI stays terminal, so it would fire immediately on
   every poll. Re-add it only when you push a new commit (which restarts CI).
2. **`github.pr_comment` / `github.review`** — respond to it (see below). CI being
   green does NOT mean there is nothing to do — review bots and humans comment
   after CI passes. Never declare the PR done on CI status alone. `wait_for.py`
   ignores your own comments by default, so it wakes you only on others' activity.
3. **`poll` (PR closed)** — the PR merged or closed; this is an exit condition.

Respond to every human and agent comment: address obvious ones directly (commit
the fix, then reply, prefixing agent replies with `🤖`) and resolve them. For
comments you are unsure about, report your analysis and proposed action to the
user — but keep monitoring while you wait.

Exit conditions (the ways to stop the loop): the PR is merged or closed, the user
explicitly tells you to stop, or `wait_for.py` hits its 12h timeout with nothing
left to act on (stand down with a status report). Blocking on a user question
pauses for the answer; it does not end monitoring.

## Rules

- `./infra/pre-commit.py` is the only pre-commit entry point.
- Commit before you run `--review`; the review never commits, pushes, or edits.
- Never amend a commit unless the user explicitly asks.
- If there are no changes to commit, say so and stop.
- `.agents/skills/fix-issue/` — end-to-end issue-fix workflow.
- `AGENTS.md` — coding guidelines.
