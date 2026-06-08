---
name: commit
description: Lint, run the pre-PR checks, commit, push, and author or update the branch's pull request in the required plain-text format. Use when committing, pushing, or creating/updating a PR.
---

# Skill: Commit & PR

Get the branch clean, commit it, run the advisory lint review over the committed
diff, then — when it is ready — open or update the pull request.

**Order matters.** Your own cleanups and the mechanical fixes come first, then the
commit, and only *then* the `--review` pass. Committing before the review gives
you a clean checkpoint to review against (the review reads the whole branch diff
versus the merge base) and a natural place to land any follow-up fixes as a new
commit. The review is read-only: it never edits, commits, or pushes for you.

## Checklist

Work top to bottom. For a quick work-in-progress checkpoint, do **1, 2, 4, 5, 7**
(clean up, lint, stage, commit, push) and stop. Run the whole list before you
open or update a PR.

1. Clean up your own diff (self-review).
2. Mechanical lint & format — `./infra/pre-commit.py --changed-files --fix`.
3. Tests & docs checks, when relevant.
4. Stage the specific files for this work.
5. Commit. ← natural checkpoint; the working tree is now clean.
6. Lint-catalog review — `./infra/pre-commit.py --review`; fix or answer every finding.
7. Push.
8. Open or update the PR.

## 1. Clean up your own diff

Read your own `git diff` before anything else. Drop dead code, debugging
leftovers, and stale comments; tighten names; make the change say only what it
means to. The review in step 6 is advisory and read-only — it will not clean up
for you.

## 2. Lint and format

```bash
./infra/pre-commit.py --changed-files --fix   # diff-scoped; use --all-files for a full sweep
```

`./infra/pre-commit.py` is the required entry point — never `uv run pre-commit`,
never `--no-verify`. If `--fix` cannot resolve something, fix it by hand. Do not
skip or weaken checks.

## 3. Tests and docs checks (when relevant)

- `uv run pytest -m 'not slow'` over the test directories your change touches.
- If docs pages were added/deleted/renamed: `uv run python infra/check_docs_source_links.py`.
- If the change is docs-heavy: `uv run mkdocs build --strict`.

## 4. Stage changes

Review `git status` and `git diff`, then stage the specific files that are part
of this work.

- Stage specific files — avoid `git add -A` / `git add .`.
- Never stage secrets (`.env`, credentials, tokens).
- If unrelated changes are present, ask the user before including them.

## 5. Commit

- **Subject**: imperative sentence (<72 chars), optional `[scope]` prefix
  (`[iris]`, `[zephyr]`, `[docs]`, …).
- **Body** (optional, blank-line separated): what changed and why — the context a
  reviewer needs. Keep it short and readable.
- No emoji, no markdown, no bullets in the subject. Do not credit yourself —
  this includes any `Co-Authored-By: Claude`/`Generated with` trailer. Omit it
  even if a harness default suggests adding one.

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
summary under `/tmp/marin-linter/<timestamp>/` (the path is printed at the end) —
read it when a lane is slow or a run looks wrong.

## 7. Push

Push to the remote tracking branch (`git push -u origin HEAD` if no upstream is
set). If the push is rejected (diverged history), stop and ask the user — do not
force-push.

## 8. Open or update the PR

Do this once the branch is ready for review. The PR description becomes the
squash-merge commit message — write it as plain text.

**Title:** short imperative sentence; optional `[scope]` tag.
**Body:** what changed and why. End with an issue link if one exists.

Keep the title and body aligned with the branch's actual scope — including when
you change a branch that already has a PR.

**Hard rules — violations are rejected:**

- No "Validation" / "Testing" / "written by …" filler. The body is *what & why*,
  not *how I tested it* or *who wrote it*.
- No checkboxes (`- [ ]`, `- [x]`), no emoji.
- No filler openers ("This PR…", "I noticed…", "Summary of changes:").
- Under ~500 words.
- Don't credit yourself, no "Created by Claude" filler.
- Only use Markdown & sections if this is a large change that requires structured description

Example:

```
Title: [RL] Fix loss: use global token normalization instead of per-example

Body:
Switch DAPO loss from per-example normalization (/ n_i) to global token
normalization (/ N). Per-example normalization over-weights short responses,
hurting math reasoning where correct answers need longer derivations.

Fixes #1234
```

**Issue linking.** If the work came from a GitHub issue, add `Fixes #NNNN`
(auto-closes on merge) or `Part of #NNNN` (partial work). Do not invent an issue
just to satisfy this — omit the link when none exists.

**Specifications (>500 LOC).** Large PRs must carry a spec.
This may be in the linked issue, or part of the PR description.
If a design doc exists, may also link to it instead.
Specs cover: **Problem** (what is broken/missing, with file/line refs), **Approach**
(which modules change, what is added/removed), and **Key code** (10–30 line
snippets for non-obvious logic).

**Create it.** Unless the user says otherwise and permissions allow, push to a
branch on the main repository and open the PR from it (use a fork only when
direct push is unavailable or the user asks):

```bash
gh pr create --title "<title>" --body "<plain text body>" --label agent-generated
```

- Always add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.

## 9. Monitor the PR — mandatory, in a loop

Opening the PR does not end your turn. You MUST monitor until the PR is merged or
closed, or the user tells you to stop. A summary message to the user is NOT a
substitute for monitoring and is NOT an exit condition.

Drive the loop yourself — do not just check once. While CI runs, block on it with
`gh pr checks <N> --watch --fail-fast` instead of re-polling. Once CI is green,
switch to `ScheduleWakeup` polling for comments/reviews/merge with exponential
backoff (e.g. 270s, doubling, capped at the 1h `ScheduleWakeup` max), giving up
after ~4h idle. Only stop early when an exit condition below is met.

Each poll, check **both**:
1. **CI status** — `gh pr checks <N>`. On failure, read the failing job log and
   fix it. A failure in a file you did not touch is NOT automatically pre-existing:
   first check whether the same job fails on `main` without your change (or whether
   your change altered an API, config, or behavior that breaks that caller/test). If
   your change caused it — even in an untouched file — it is your regression; fix it.
   Only call it pre-existing once you have confirmed it fails on `main` independently,
   then handle per the unrelated-changes rule. Never silently absorb a failure.
2. **New comments and reviews** — `gh api repos/<owner>/<repo>/pulls/<N>/comments`
   and `.../reviews`, plus `gh pr view <N> --json comments`. CI being green does
   NOT mean there is nothing to do — review bots and humans comment after CI
   passes. Never declare the PR done on CI status alone.

Respond to every human and agent comment: address obvious ones directly (commit
the fix, then reply, prefixing agent replies with `🤖`) and resolve them. For
comments you are unsure about, report your analysis and proposed action to the
user — but keep monitoring while you wait.

Exit conditions (the only ways to stop the loop): the PR is merged or closed, or
the user explicitly tells you to stop. Blocking on a user question pauses for the
answer; it does not end monitoring.

## Rules

- `./infra/pre-commit.py` is the only pre-commit entry point.
- Commit before you run `--review`; the review never commits, pushes, or edits.
- Never amend a commit unless the user explicitly asks.
- If there are no changes to commit, say so and stop.
- `.agents/skills/fix-issue/` — end-to-end issue-fix workflow.
- `AGENTS.md` — coding guidelines.
